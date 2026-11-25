# Parallel N-body (Gaia DR3): plan for the MPI version

**Group:** Pelle, Rasmus, David · **Course:** 02616 Large-Scale Modelling · **Drafted:** 9 Nov 2026
**Plenum slide:** 18 Nov · **Hand-in:** Fri 4 Dec 2026, 12:00 (PDF report and ZIP of the code)
**Status:** the N-body project is **approved** as our Project 2 (confirmed 9 Nov).
**Code in the project:** `code/SerialImplementation.py`, `code/ParallelImplementation.py` (v1, implemented 9 Nov), `code/batch.sh`, `code/plot_results.py`, `code/GaiaSubsetSelection.py`, and the visualisation scripts `code/DataVisualize.py`, `code/plot_space.py`, `code/plot_space_2.py`. How the parallel script and the job script work is described in `claude/ParallelImplementation_and_batch_summary.md`.

### Where we are (updated 10 Nov)

- **v1 is implemented and validated.** `ParallelImplementation.py` follows this plan (§2) and uses only weeks 1–6. Its results are **bit-identical** to the serial code for P = 1, 2, 3, 4, 7, with and without merging (tested on synthetic data), and on the real nearest-5000 Gaia subset on the HPC.
- **Decisions taken:** `Allgatherv` (option B, §2.4); merges are checked every step, as in the serial code; the stop flag is agreed every step (`--stop-check-every 1`); all runs go into **one** `results.csv`, protected by a file lock.
- **Found while building v1:**
  - pandas gave column-major position arrays, which `Bcast` scrambled on ranks ≥ 1. `load_subset` now forces C order.
  - A SIGTERM sent to `mpirun` kills the ranks before they can save. `batch.sh` therefore always sets `--time-limit` 2 min below the walltime.
- **First results (N = 5 000, 100 steps, 8 ranks, Xeon E5-2650 v4):**
  - Speed-up 7.76–7.79×, i.e. about 97 % efficiency. Communication is 0.04 % of the run time.
  - The step time grows as N² (5k → 10k → 15k).
  - The kernel runs at about 0.5 % of the 35.2 GFLOP/s per-core peak. Absolute times differ by about 16 % between nodes with the same CPU model, but the speed-up is the same on both.
- **Next:** the E2 scaling runs (P = 1…24) on a fully reserved node, then E3–E5, then the 18 Nov plenum slide.

This plan covers:

1. **v1**, a first parallel version of `SerialImplementation.py` that uses **only material from weeks 1–6**.
2. The **hooks** v1 leaves so that weeks 7, 8 and 9 can be added later as separate, measurable iterations.

Every design choice is tagged with the lecture it comes from, e.g. `[W4 §3.8]`, which points to the condensed week notes. Items marked **(decide)** are open choices for the group. Items marked **(check)** are assumptions that must be checked against the code or by measurement before we rely on them.

> This started as a plan only. At the group's request v1 was then implemented with AI help (9 Nov), so the code must be read through by the whole group and mentioned in the AI declaration. Every member should be able to explain it at the oral exam.

---

## 0. Ground rules for v1

| Allowed in v1 (weeks 1–6) | Saved for later iterations |
|---|---|
| `MPI.COMM_WORLD`, rank and size, blocking `Send`/`Recv` and deadlock awareness `[W1]` | `Clone()`, `Split`, groups, `Create_cart`, `Shift` `[W7]` |
| Compute-bound vs. memory-bound, rank placement with `span[ptile]`, `--map-by`, `--bind-to`, `--report-bindings`, InfiniBand vs. Ethernet `[W2]` | Formal Amdahl/Gustafson fits, strong vs. weak scaling plots `[W8]` |
| Timing anatomy (t0…t3), the model `T(s) = μ + s/B` `[W3]` | Custom datatypes: `Create_contiguous/indexed/struct/resized` `[W8]` |
| `mpi4py` with **upper-case buffer API** for anything hot, lower-case only for one-off Python objects `[W4 §3.5]` | MPI-IO (`File.Open`, `Write_at`, `Scan`, `Write_ordered`) `[W9]` |
| `Bcast`, `Scatter`, `Gather`, `Reduce`, `Allgather`, `Allreduce`, `Isend`/`Irecv` with `Wait` `[W4 §3.7–3.8]` | cProfile per rank, mpiP `[W9]` |
| Static vs. dynamic scheduling and chunk-size lessons from Project 1 `[W5]` | |

**One structural rule we take on early:** every parallel function takes `comm` as a **parameter** and never uses `MPI.COMM_WORLD` directly. In v1 the caller simply passes `MPI.COMM_WORLD`. This uses no week 7 MPI feature, only a function argument, but it means week 7 can later pass in a clone or a sub-communicator without any rewrite.

---

## 1. Starting point: how each piece of the serial code fits into the parallel version

Every part of `SerialImplementation.py` (`code/SerialImplementation.py`; overview in `claude/LSM_Project2_session_summary.md`) takes one of four roles in the parallel version:

- **Owned**: each rank holds only its block of stars.
- **Replicated**: every rank holds the full array.
- **Rank 0 only**: done once, on rank 0.
- **Collective**: all ranks must call it together.

| Serial piece | Parallel role in v1 | Notes |
|---|---|---|
| `load_subset`, `parse_spectral_classes`, `guess_spectral_classes` | **Rank 0 only**, then `Bcast` | Rank 0 reads the CSV. Positions and masses go out with `Bcast` on NumPy buffers. Classes (strings) go once with lower-case `bcast`, which is acceptable because it isn't a hot path `[W4 §3.5]`. |
| `Stars.pos` (N×3) | **Replicated** (refreshed every step by `Allgather`) | Needed by every rank to compute forces on its own block. |
| `Stars.vel` (N×3) | **Owned** (block) | Only gathered when merging or saving. |
| `Stars.mass`, `members`, `owners`, `classes` | **Replicated** | These only change during a merge, which happens on every rank at the same time (see §2.5). |
| `compute_accelerations` | **Owned i-range × all j** | **Key refactor:** add `i_start, i_stop` arguments. The serial version calls it with `0, N`, so serial and parallel run the *same kernel*. Inside the loop, `i` must stay the **global** index, because `dist_sq[i] = np.inf` (no self-force) and `raw_dist_sq[i + 1:]` (pairs with j > i) both rely on it. The function then returns accelerations only for the own block, shape `(i_stop − i_start, 3)`. |
| `timed_accelerations` | **Owned** | `perf.interactions += stars.count * (stars.count - 1)` counts all N rows. Per rank it must become `n_own * (N − 1)`, and the total comes from a `Reduce(SUM)`. Otherwise `interactions_per_s` is overstated P times. |
| `group_close_stars`, `merge_stars` | **Replicated, deterministic** | Every rank runs the same union-find on the same global pair list (§2.5). |
| `step` | **Collective** | Contains the `Allgather`, and also the `while close_pairs:` merge loop. Every rank must call both together, every step (§2.5). |
| `mean_distance_to_centre_of_mass`, `total_momentum` | **Owned partial + `Allreduce`/`Reduce`** | Sum over our own block, then a reduction `[W4 §3.8]`. |
| `Perf`, `peak_memory_mb` | **Per rank**, then `Reduce`/`Gather` to rank 0 | Report max, min and mean over ranks (§2.9). |
| `LivePlot`, `plot_start_to_end`, `save_figure` | **Rank 0 only** | Needs a gather of all positions. Off by default in benchmarks. Never `plt.show()` in a batch job. |
| `StopRequest`, `run_mode` | **Collective decision** | All ranks must agree to stop on the *same* step, or the next `Allgather` deadlocks (§2.7). |
| `build_results_row`, `append_results` | **Rank 0 only** | The `results.csv` columns for ranks are already there. |

**Suggested file layout (decide):** keep the physics in `SerialImplementation.py`, possibly moved into an importable module such as `nbody_core.py`. Add a `ParallelImplementation.py` that imports the same kernel functions and adds only distribution, communication and timing. If serial and parallel share one kernel, any differences we measure come from the parallelisation, not from two slightly different force loops.

---

## 2. v1 design

### 2.1 Decomposition: replicated positions, block-owned stars

This is the "Allgather" plan from our Assignment 2 description:

- Stars are split into **P contiguous blocks** with `np.array_split` logic: block sizes differ by at most 1 `[W3 §3.3]`.
- Rank *r* **owns** stars `[i_start(r), i_stop(r))`. It updates their velocities and positions.
- Every step, all ranks **Allgather** the positions of all stars, so each rank can sum the forces on its own block from **all** N stars `[W4 §3.8]`.

**Why this first:**

- It is the simplest correct split. The force loop already runs "for each i, vectorised over all j", which is exactly a per-rank i-range (session summary, *Force loop*).
- **Load balance is good by construction.** With direct summation, every i interacts with all N j, so equal block sizes mean equal work. This is the opposite of Mandelbrot `[W5 §3.2]`, so we should not expect to need a dynamic scheduler in v1. That only becomes relevant later (§7.3).
- **Memory is not the limit.** Positions for all 626k stars are 626 000 × 3 × 8 B ≈ 15 MB per rank. Replicating them is cheap, so the hard limit is the O(N²) compute (§3).

### 2.2 One time step on one rank

This mirrors the serial `step` exactly. The serial order is: forces + close pairs → `while close_pairs:` (merge, then compute forces + pairs again) → kick → drift. Merges therefore use pairs found **in the same step** from the current positions, and the full O(N²) force computation is **repeated** after every merge round.

```
1. [comm]     Allgather(own positions → all positions)       # collective
2. [compute]  acc_own, pairs_own = compute_accelerations(pos_all, mass_all,
                                       i_start, i_stop, softening, merge_radius)
3. [merge?]   n_pairs = Allreduce(SUM, len(pairs_own))         # collective, 1 int
   while n_pairs > 0:                                          # same decision on every rank
       global merge (see §2.5), recompute block boundaries
       repeat 1–2 and the Allreduce
4. [update]   vel_own += acc_own * dt ;  pos_own += vel_own * dt
5. [stop?]    agree on stop flag (see §2.7)                   # collective
6. [diag]     every k steps: Reduce/Allreduce momentum, mean distance, (energy)
```

- The loop condition **must be a collective result**. In the serial code `while close_pairs:` is a local test. In parallel, one rank may find pairs while another finds none, and if only the first one goes back into the `Allgather`, every rank hangs.
- `pos_own` should be a **view** into a slot of the replicated `pos_all` buffer. The local update then writes straight into the send region, with no extra copy. **(check)** that `Allgather` accepts a send buffer that overlaps the receive buffer. If it doesn't, use a small separate send buffer. The "in-place" option is not part of the lectures.

### 2.3 Data layout and buffers

- Store data as **structure of arrays (SoA)**: `pos` (N, 3) float64, `vel` (N, 3) float64, `mass` (N,) float64. All are C-contiguous, so the x, y, z of one star sit next to each other `[W8 §2]`. This is also the layout a week 8 `Create_contiguous(3)` "xyz" type expects (§8.2).
- **Pre-allocate** every communication buffer once, outside the loop, and reuse it. No `np.concatenate` or new arrays inside the step loop.
- **Upper-case only** inside the loop (`Allgather`, `Allreduce`, `Bcast`). The lower-case versions pickle the data, which defeats the purpose of measuring the interconnect `[W4 §3.5]`.
- Set `OMP_NUM_THREADS=1` (and `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) in job scripts. That way one rank uses exactly one core, and P ranks do not each spawn hidden BLAS threads and oversubscribe the node `[W1 §3.3, W2 §3.9]`.

### 2.4 When N is not divisible by P (decide)

Plain `Allgather` needs **equal** chunk sizes on every rank. Two options:

| Option | How | Pro | Con |
|---|---|---|---|
| **A. Zero-mass padding** | Pad N up to a multiple of P with dummy stars that have mass 0, placed far away | Uses only week 4 collectives | Merge detection must skip padding. Changes the length of the j-arrays, so NumPy's pairwise summation can group terms differently: results are close to serial but **not bit-identical** |
| **B. `Allgatherv`** | Pass counts and offsets per rank | No dummy stars. Same j-array as serial, so results **can be bit-identical** | Not shown in the lectures, though it is the obvious "v" variant of `Allgather` in the `mpi4py` docs |

**Recommendation:** use **B** if the group accepts one small documented step beyond the lecture list. It makes validation (§4) much stronger, because we can test exact equality instead of tolerances. Otherwise use **A**, and validate with tolerances and conserved quantities. Either way, write down the choice and the reason in the report.

### 2.5 Merging in parallel (the hardest part of v1)

Merges are **rare global events**, so v1 handles them with a global re-sync, which is slow but simple:

1. Each rank finds close pairs in its own i-block during the force loop. No extra cost; this is the same trick as in the serial code.
2. **Agree whether any merge is needed.** `Allreduce(SUM)` (or `MAX`) on a 1-element "pair count" buffer `[W4 §3.8]`. If it is 0 everywhere, skip the rest. This is the common case and costs one tiny collective.
3. If there are pairs, **gather the pair lists to every rank.** They are small and irregular, so lower-case `allgather` of Python lists is acceptable, because merges are rare and not on the hot path.
4. **Gather velocities** (`Allgather` of N×3) so that every rank has the full state for one step.
5. **Every rank runs the same `group_close_stars` and `merge_stars`** on the same inputs. This gives the same result everywhere with no further communication.
6. Recompute block boundaries for the new N. Then run the force pass again (the `while` loop in §2.2).

What the real code already guarantees (checked in `SerialImplementation.py`):

- **No duplicate pairs.** The kernel only reports pairs with j > i (`raw_dist_sq[i + 1:]`), and every i is owned by exactly one rank. Each pair is therefore found exactly once, by the owner of the smaller index, also when j lies in another rank's block. Concatenating the per-rank lists gives the same set as the serial code, with no deduplication needed.
- **Union-find doesn't depend on pair order.** `group_close_stars` always attaches the larger root to the smaller one (`parent[max] = min`), so every group's root ends up being its smallest index, whatever order the pairs arrive in. `np.unique(..., return_inverse=True)` then numbers the groups in index order. If ranks are concatenated in rank order, the list is even in exactly the serial order.
- **Merged class choice is deterministic.** `np.lexsort((masses, real, labels))` is a stable sort, so ties between equally heavy stars resolve by original index, the same on every rank.
- **`owner` stays consistent.** `owner = labels[stars.owner]` is a pure function of the replicated arrays, so it can stay replicated.

Still to watch:

- **Cost of a merge round:** each round repeats the full O(N²) force pass. If clusters merge in chains over several rounds in one step, that step costs several normal steps. That's harmless for correctness, but keep `merge_calls`/`forces_calls` in the results so outlier steps can be explained.
- **Milestone order:** get v1 working and validated **with `--merge-radius 0` first**, then switch merging on. Note that the default `--merge-radius` is 0.01, so merging is **on by default**. Benchmark scripts should always set it explicitly.

### 2.6 Start-up

- Rank 0: parse the arguments, load the CSV, guess the classes with the fixed seed. Then `Bcast` N, positions, masses (and velocities, if `pmRA`/`pmDE`/`RV` starting velocities are added later), plus classes and owners with lower-case `bcast`.
- Alternative: every rank reads the CSV itself. That is simpler, but P ranks reading the same file at once is a file-system load that week 9 talks about. Keep "rank 0 reads, then `Bcast`" for v1, and **time it separately** as `t_init`.
- Argument parsing: either every rank parses `sys.argv` (identical, so harmless), or rank 0 parses and broadcasts. **Decide.** The first option is simpler.

### 2.7 Stopping together: time limits, Ctrl+C and SIGTERM

A collective (`Allgather`) runs every step. If **one** rank stops and the others don't, the others wait forever. This is the week 1 deadlock lesson in a new form `[W1 §3.6]`.

- **Time limit:** only **rank 0's clock** decides. Every step (or every k steps, **decide**), rank 0 `Bcast`s a 1-element stop flag. Alternatively, all ranks `Allreduce(MAX)` their local flags. The second version also catches a signal that only one rank received.
- **Signals:** keep `StopRequest` on every rank, but treat it only as a *local request*. It becomes a stop only through the collective flag above. The second Ctrl+C in `StopRequest._handle` raises `KeyboardInterrupt` immediately. In the parallel version this can happen on one rank in the middle of a collective, so after the first request, ranks should wait for the collective flag instead of exiting on their own. (Its docstring also says SLURM; the DTU cluster uses LSF.)
- **Time limit in the serial code:** the deadline check uses `perf.step_times[-1]` as the expected duration of the next step. In parallel, step times differ slightly between ranks, so only rank 0 should do this check.
- **(check)** how `mpirun` and LSF pass SIGTERM/SIGINT to the ranks (`bkill` sends a sequence of signals). Test it: start a short job, `bkill` it, and confirm that the results row is still written.
- The flag check costs one small collective per step. Measure it. It should be latency-sized, i.e. µs `[W3 §3.10]`.

### 2.8 Output

- `results.csv`: written **by rank 0 only**. Fill in the existing columns for ranks, nodes and communication bytes. Add the new columns listed in Appendix B.
- Final state (`--save-final`): `Gather` positions and velocities to rank 0, which writes the CSV (the same padding/`Gatherv` choice as §2.4). This rank-0 funnelling is exactly what week 9 replaces (§8.3).
- Plots: rank 0 only, after a gather. `--plot-iter` forces a gather every N steps, so it is **off in all benchmark runs**. Matplotlib must use the `Agg` backend in batch jobs.

### 2.9 Timing and metrics

`Perf` already has the sections `load`, `forces`, `merge`, `integrate`, `communication`, `plotting` and `output`, plus `comm_bytes` and `step_times`. `build_results_row` already accepts `implementation="mpi"`, `n_ranks` and `load_imbalance`. Reuse these names rather than inventing new ones, so serial and MPI rows in `results.csv` line up column for column. The table shows how the plan's sections map onto them, with clean timer boundaries `[W3 §3.8–3.9]`:

| Plan section | `Perf` key | Contents |
|---|---|---|
| init | `load` (+ new `distribute`) | MPI start-up excluded. Load on rank 0, then `Bcast` and buffer allocation, timed separately |
| comm | `communication` | `Allgather` of positions (per step, summed). Add bytes to `comm_bytes` |
| compute | `forces` | `compute_accelerations` on the own block |
| update | `integrate` | kick and drift |
| merge | `merge` | the whole merge path, including its gathers. The repeated force passes stay under `forces` |
| sync | new `sync` | the stop-flag and pair-count collectives, and diagnostics |
| final | `output`, `plotting` | gathers, saving, plotting |

Note that `other_sim_time_s` in `build_results_row` is computed as simulation time minus forces, merge, integrate and communication. Any new `sync` time therefore lands there unless it gets its own column.

- Use **wall-clock time** `[Project2 notes]`, via `time.perf_counter()` or `MPI.Wtime()`.
- Before starting the step-loop timer, synchronise all ranks with one tiny `Allreduce`. Otherwise early ranks measure the time they spend waiting for late ones.
- At the end, `Reduce` each section with MAX, MIN and SUM to rank 0. **The slowest rank defines the step time.** The gap between max and min is our first, cheap measure of load imbalance (week 9 profilers will confirm it later). This also answers the week 4 question "which rank's timings should you plot?".
- **Throughput metric:** interactions per second = N² × steps / t_loop. This plays the role that Mlup/s plays in the Poisson text. It converts to GFLOP/s with the existing model of 20 FLOP per pair, which we compare with the **theoretical peak per core** of the actual CPU `[W2 §3.1]`.
- **Record the CPU model** (`lscpu` or `/proc/cpuinfo` from inside the job) and the host names in every results row `[Project2 notes, W2 lab 1]`.

---

## 3. Expectations, written down *before* we run anything

The project text asks us to state expectations first and compare afterwards. Here is a simple model to fill in with measured constants.

**Per-step time on P ranks:**

```
T_step(P) ≈ N²/(P·R)                         compute: R = interactions/s per core (measure, P=1)
          + T_allgather(N, P)                communication
          + c_py · N/P                       Python loop overhead per i-row
          + T_sync                           stop-flag collective (≈ latency)
```

with, from `T(s) = μ + s/B` `[W3 §3.10]` and a ring-style allgather (a guess, since the MPI library picks the algorithm):

```
T_allgather ≈ (P−1)·μ + ((P−1)/P) · 24·N / B
```

**What this predicts:**

- **Compute grows as N², communication as N.** For large N the compute term dominates and speed-up should be close to ideal. For small N (1k stars) the per-row Python overhead and the latency terms dominate, and adding ranks quickly stops helping. **Expect a 1k-star run to scale poorly. That is a result, not a bug.**
- **Single node vs. two nodes:** the Allgather moves from shared memory to InfiniBand `[W2 §3.7]`. With large N this hardly matters, because compute dominates. With small N it should be clearly visible. Running over **Ethernet** should make the small-N case much worse.
- **Is the kernel really compute-bound? (check)** On paper N-body is compute-bound. But the NumPy kernel creates **temporary arrays of length N** (dx, dy, dz, r², …) for every i. For N ≳ 100k each temporary is about 1 MB, which falls out of L2. The kernel then streams through memory, and adding ranks on one socket may saturate memory bandwidth, just like the Poisson example in week 2 `[W2 §3.5–3.6]`. **Test:** compare `--map-by ppr:n:package` (spread over sockets) against compact placement at the same P. If spread is clearly faster, the kernel is memory-bound in practice.
- **How far can we go?** With R measured at P = 1, one step of the full 626k set costs 626 000² / (P·R) seconds. Calculate this number early. It decides whether "full Gaia" means a handful of steps or is out of reach for v1, and it feeds straight into the "as many stars as possible" goal of the assignment.

**Run budget:** Project 2 suggests that the serial reference run should take **≤ 3–4 minutes**. Choose the step count per N from the measured R:

```
steps(N) ≈ 200 s · R / N²
```

Then keep it **fixed** for all P at that N. That makes it strong scaling; the formal treatment comes in week 8.

---

## 4. Validation (before any timing is trusted)

Ordered from cheapest to most expensive:

1. **Rank-count independence.** Run the same input on P = 1, 2, 3, 4, 7 for a few steps, deliberately including P values that don't divide N. Compare the final positions with the serial code:
   - with option B (§2.4): expect **exact equality** for merge-free runs. Every row of the kernel works on the same full-length j-array in the same order as in the serial code, so the floating-point operations are identical. The existing `final_position_checksum` column should then match to the last digit;
   - with option A: expect agreement to ~1e-12 relative after a few steps.
   - **Note:** N-body systems are **chaotic**. Tiny round-off differences grow exponentially over many steps, so compare after **few** steps, and use conserved quantities for long runs.
2. **Conserved quantities** (already in the code): total momentum ≈ 0 (it starts at rest), the centre of mass stays fixed, total mass is exact, including through merges.
3. **Energy:** E = K + U. The potential U is O(N²), so compute it in parallel: each rank sums over its own i-block, then `Reduce(SUM)`. Do this only every k steps. Track the relative drift |ΔE/E₀| against dt.
4. **Two-body test** (from the assignment): circular orbit, compare the period and radius with the analytic solution. This runs with P = 2 as well (one star per rank).
5. **Plummer sphere** (from the assignment): generate one, check that the density profile stays stable for a few crossing times.
6. **Merges:** same input, P = 1 vs. P = 4, with merging on. Check that the merge count, final N, total mass and momentum are identical.

**Always keep a small "golden" test set** (e.g. N = 200, 20 steps) that runs in seconds on a laptop with `mpirun --oversubscribe` `[W4 §3.10]`. Use it **only for correctness, never for timing**.

---

## 5. Experiments for v1

All runs are batch jobs on the `hpcintro` queue `[W1 §3.8]`, with `--report-bindings` on to check placement `[W2 lab 2]`.

| # | Question | N | P | Placement |
|---|---|---|---|---|
| E1 | Serial baseline: R, c_py, the steps(N) budget | 1k, 5k, 20k, 50k (check feasibility) | 1 | 1 node |
| E2 | Single-node scaling: where does adding ranks stop helping? | same | 1, 2, 4, 8, 12, 16, 24 | `span[hosts=1]` |
| E3 | Compact vs. spread over sockets: memory bandwidth (§3) | 20k, 50k | 4, 8, 12 | `--map-by ppr:n:package` vs. compact |
| E4 | One node vs. two nodes (InfiniBand) | 1k, 20k | 8, 16, 24 | `span[ptile=…]` |
| E5 | InfiniBand vs. Ethernet (small N, where communication matters) | 1k, 5k | 8 over 2 nodes | select the network as in the week 2 OSU scripts |
| E6 | Cost of merging and of the stop-flag sync | 5k with merging on/off | 8 | 1 node |

- **Planning arithmetic:** count *configurations × repeats × ~4 min* before submitting anything `[W5 §3.6]`. Run at least 3 repeats per point and report the median and spread.
- Write every run to `results.csv` with `completed = True/False`. Only compare serial and MPI on completed runs, as decided yesterday.
- The **18 Nov plenum slide** should be essentially §3 and this table: our expectations, which experiments test them, and how we will tell the explanations apart.

---

## 6. Shortfalls and risks

| Risk | Effect | Mitigation |
|---|---|---|
| The general requirements in the Poisson text (Cartesian map, Mlup/s, bandwidth vs. hardware) may still be what the examiners look for | Missing an expected discussion point | The project is approved. We cover the spirit anyway: Cartesian map → §8.1, Mlup/s → interactions/s, bandwidth vs. hardware → the §3 test |
| Replicated positions: every rank receives all N positions every step | Communication volume per rank doesn't shrink with P, which limits strong scaling for small N | Accept for v1 and measure it. The week 7 ring or 2-D force decomposition fixes it (§8.1) |
| Forces are computed twice per pair (i on j and j on i) | 2× more compute than necessary | Possible later improvement (§7.2). It brings back load imbalance and needs a `Reduce` of forces |
| Python loop over i | Overhead per row dominates for small N/P | Tile the i-loop (§7.1). Measure c_py first |
| NumPy temporaries | The kernel turns memory-bound for large N | Test E3. Tiling j into blocks that fit in cache. Numba (§7.1) |
| Ranks disagree about stopping | Deadlock in `Allgather` | Collective stop flag (§2.7) |
| Merge results differ between ranks | Silent divergence: ranks simulate different universes | The serial merge code is already order-independent (§2.5), so the main rule is that every rank gets the same full pair list. Assert that N and total mass are equal across ranks after every merge (cheap `Allreduce` check, debug mode) |
| Chaos | Long-run serial vs. MPI comparison is meaningless | Short-run comparison plus conserved quantities (§4) |
| All stars start at rest | Cold collapse: close encounters and merges become frequent and time-step errors grow | Keep for benchmarks (fixed dt). Starting velocities from `pmRA`/`pmDE`/`RV` are still a TODO, physics rather than parallelism |
| Fixed dt during close encounters | The "bouncing" seen yesterday | Softening is already in place. An adaptive dt would break run-to-run comparability (as decided) |
| Hidden BLAS threading | P ranks × T threads oversubscribe the node, which ruins measurements | `OMP_NUM_THREADS=1` etc. in every job script |
| Laptop vs. HPC | Timing on laptops with `--oversubscribe` is meaningless | Laptop for correctness only, HPC for every number in the report |
| Full 626k stars | Possibly minutes per step even on a full node | Calculate early (§3). Present "largest N per time budget" as a result |

---

## 7. Possibilities (not v1, but worth knowing about)

### 7.1 Kernel speed (independent of MPI)

- **Tile the i-loop:** compute a block of i-rows × all j at once with 2-D broadcasting. Fewer Python iterations, but larger temporaries (block × N), so the tile size must balance Python overhead against cache size.
- **Numba** `@njit` for the pair loop: removes the temporaries entirely and may turn the kernel back into a compute-bound one. Project 2 explicitly allows hybrid parallelism (multiprocessing or parallel Numba) **if** we explain how it uses the cores. That would be a separate iteration with its own measurements.
- Keep the plain NumPy kernel as the reference, so we can show which change caused which speed-up `[W5 §3.5 "code progress"]`.

### 7.2 Using symmetry (Newton's third law)

- Compute each pair only once (j > i), which halves the flops. But row i now has N − i − 1 pairs, so **equal blocks become unequal work**: the triangle problem. This is Project 1's imbalance all over again, and its fixes apply: cyclic or small-chunk distribution of rows `[W5 §3.5]`.
- The reaction forces on stars owned by other ranks then need summing: an `Allreduce` of an N×3 force array, or a reduce-scatter pattern. Measure whether the halved compute pays for the extra communication.

### 7.3 Interaction radius, load imbalance, dynamic scheduling (from the assignment)

- If we introduce a gravitational cut-off radius, the work per star depends on local density (Gaia stars are clustered near the Sun). **This is when** the Project 1 dynamic scheduler becomes useful: a manager hands out chunks of i-rows on request, using `Status.source`/`tag` `[W5 §3.8]`.
- **Physics warning:** a hard cut-off changes the answer (gravity doesn't truly vanish beyond a cut-off). Any speed-up must be reported together with the error it introduces compared with direct summation.

### 7.4 Overlapping communication with compute (allowed in v1 if time permits)

- Split the step: start `Iallgather`, or `Isend`/`Irecv` to every rank, then compute the forces from **own-block j** first (already local), then `Wait`, then the remaining j `[W4 §3.7]`. Every request must get a `Wait` or `Test` `[Project2 notes]`.
- `Iallgather` isn't shown in the lectures, but the `Isend`/`Irecv` version is. Expect a gain only where communication is a real share of the step time (small N, multiple nodes).

---

## 8. Iteration roadmap: weeks 7, 8 and 9

Each iteration = one change, measured against the previous version on the same experiment set, plus a short code snippet in the report showing whether it helped or hurt `[W5 §3.5]`.

### 8.1 Week 7: communicators and Cartesian topologies

| Iteration | What changes | Hook already in v1 |
|---|---|---|
| **7a. Clone** | `comm = MPI.COMM_WORLD.Clone()` in `main`, passed down | `comm` is a parameter everywhere |
| **7b. Ring algorithm on a 1-D periodic Cartesian communicator** | `Create_cart([P], periods=[True])`, `Shift(0, 1)` gives the neighbours. Each rank keeps only its own block and passes a travelling block of j-positions around the ring P−1 times, adding up forces as it goes. Memory drops from O(N) to O(N/P), and with `Isend`/`Irecv` the next block can travel while the current one is computed | Block ownership (§2.1). The kernel takes a j-array argument |
| **7c. 2-D force decomposition** (the block mat-vec of the week 7 lecture) | P = r × c grid. `row_comm = Split(row)`, `col_comm = Split(col)`. Each rank computes forces of i-block(row) on j-block(col), then `Allreduce` along the row. The communication volume per rank falls from ~N to ~N/√P | Kernel with separate i-range and j-range |
| **7d. Node-aware** | `Split_type(COMM_TYPE_SHARED)` gives one communicator per node, for a hybrid layout (one rank per node with Numba threads, or node leaders doing the inter-node exchange) | `OMP_NUM_THREADS` already controlled in the job scripts |
| **7e. Ensembles** | Run several N or seeds at once on sub-communicators. Useful when a single N stops scaling | `main` accepts a `comm` |

- **7b or 7c** also covers the "use the Cartesian map" requirement in Project 2, with a real reason for it.
- Use `MPI.PROC_NULL` and `COMM_NULL` checks wherever a grid doesn't fit P exactly `[W7 §3.7]`.

### 8.2 Week 8: scaling laws and custom datatypes

- **Strong scaling:** E2/E4 re-plotted as speed-up and efficiency, with a fit of Amdahl's serial fraction s. Explain any decline with the §3 model.
- **Weak scaling for an O(N²) problem:** to keep the *work per rank* constant, **N must grow as √P**, not as P. Doubling the ranks means N × 1.41. Getting this wrong is a classic mistake. Write it explicitly in the report.
- **Custom datatypes:**
  - `MPI.DOUBLE.Create_contiguous(3)`: one "xyz" element. Send `(pos, n_stars, XYZ)` so that counts are in stars, not doubles.
  - `Create_struct` for a **star record** (pos, vel, mass, id, class code). The assignment mentions "a unique data structure"; this is where it pays off, e.g. for the merge re-sync or for migrating stars between ranks in 7b/7c. Compare it with packing into a temporary buffer, as the week 8 lab does (bandwidth plot).
  - Always `Commit()` and `Free()`. Verify every transfer (reset the buffer, receive, assert) `[W8 §3.12]`.

### 8.3 Week 9: parallel IO and profiling

- **Snapshots with MPI-IO** instead of a gather to rank 0:
  - Each rank writes its block of `pos`/`vel` with `Write_at_all` at an offset from an **exclusive** prefix sum of block sizes. That is `Scan` minus the rank's own size `[W9 §3.6]`.
  - Or use `Write_ordered`.
  - Delete the file before every run `[W9 §3.7]`.
  - Verify against the rank-0 CSV from v1.
  - Write a small header (N, step, time), or a separate metadata file, so plots can be made afterwards from the snapshots instead of with `--plot-iter`.
- **Profiling:**
  - Run cProfile per rank (`mpirun --output dir=…`) to check that the `Perf` sections agree with a real profile and to look for surprises such as time in merging, logging or `np.isfinite`.
  - Run mpiP to see the MPI% per rank.
  - Compare both with our own max/min timers (§2.9). Remember that mpiP's MPI times are flagged as unreliable under Python `[W9 §3.12]`.

---

## 9. Milestones and checklist

**M0: refactor (serial, no MPI)**
- [x] `compute_accelerations(positions, masses, softening, merge_radius, i_start, i_stop)`; the serial version uses the default range (all stars)
- [x] Pass `comm` as a parameter to every parallel function
- [x] Validation by comparing final-state CSVs: built into `batch.sh` (`MODE=validate`)
- [ ] A small "golden" test set (N = 200, 20 steps) kept in the repository for laptop checks

**M1: v1 without merging (`--merge-radius 0`)**
- [x] Rank 0 loads the data and `Bcast`s it; block boundaries come from `array_split` logic
- [x] `Allgatherv` (option B, §2.4) and the per-block kick/drift
- [x] Collective stop flag; tested with the time limit and SIGTERM. A SIGTERM sent to `mpirun` kills the ranks before they save, so `batch.sh` uses `--time-limit`
- [x] Timers reduced to rank 0 (max, min, sum); `results.csv` row with ranks, nodes, CPU model, host names
- [x] Validation step 1 of §4 passes for P = 1, 2, 3, 4, 7 (bit-identical). Conserved quantities (step 2) match the serial run. Energy (step 3) is not implemented yet

**M2: v1 with merging**
- [x] Global merge re-sync (§2.5), with the `while` condition decided collectively (§2.2)
- [x] Validation step 6 of §4 on synthetic data (290 of 300 stars merged, identical results)
- [ ] The same check on real Gaia data with a merge radius that actually produces merges

**M3: physics validation**
- [ ] Two-body, Plummer sphere and energy drift (§4, steps 3–5)

**M4: experiments E1–E6 and the 18 Nov plenum slide**
- [x] First runs: serial vs. 8 ranks at N = 5 000 on two nodes; 8 ranks at N = 10 000 and 15 000
- [x] `plot_results.py`: step time vs. N, serial vs. MPI per node, and speed-up vs. ranks (once E2 exists)
- [ ] E2 (P = 1…24, on a fully reserved node), then E3–E6
- [ ] Plenum slide

**M5+: iterations from §8**, one at a time, each with before/after numbers

How the group splits the milestones between members is for the group to decide. A natural split: one person on M0/M1, one on validation (§4), one on job scripts and experiment logistics (§5, Appendix A).

---

## 10. Open questions for the group (and the teachers)

1. ~~§2.4: padding (A) or `Allgatherv` (B)?~~ Decided: `Allgatherv` (B).
2. ~~Do we merge every step, or only every k steps?~~ Decided: every step, as in the serial code.
3. Starting velocities from `pmRA`/`pmDE`/`RV`: before or after the first benchmarks? (Recommendation: after. Benchmarks don't need realistic physics, but the report should say so.)
4. How large an N can we realistically show? Answer this after E1 (§3).
5. Which week 7 path do we take: the ring (7b) or the 2-D force decomposition (7c)? Doing both deeply is likely too much ("deep analysis of a few" `[W5 §3.6]`).
6. Which coordinate frame is "the" frame? `GaiaSubsetSelection.py` and `plot_space*.py` use equatorial (ICRS) x, y, z, while `DataVisualize.py` converts to galactic coordinates. The physics doesn't care, but plots in the report should all use one frame.

---

## Appendix A: LSF job script skeleton

This was the skeleton written before v1 existed. The job script in use is `code/batch.sh` (see `claude/ParallelImplementation_and_batch_summary.md`).

```bash
#!/bin/bash
#BSUB -J nbody_P8_N20k
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -q hpcintro
#BSUB -n 8
#BSUB -R "span[hosts=1]"          # or span[ptile=4] for two nodes
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 00:15

module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8
module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11   # only if needed

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg

lscpu | grep "Model name"          # record the CPU type for the report
mpirun --report-bindings python3 ParallelImplementation.py \
       --data subsets/gaia_nearest_20000.csv --steps <from §3> --merge-radius 0
```

For the placement experiment E3, reserve the full node and start fewer ranks with an explicit `-np` and `--map-by ppr:n:package` `[W2 §3.9]`.

## Appendix B: `results.csv` columns for v1

**Already in `build_results_row`, just fill them:** `implementation` (`"mpi"`), `n_ranks`, `n_threads`, `communication_time_s`, `comm_bytes`, `comm_bandwidth_GBps`, `load_imbalance` (max/mean of `forces` time over ranks), plus the validation columns `final_mean_dist_com_pc`, `final_momentum_norm` and `final_position_checksum`.

Some existing columns need care in parallel:

- `hostname` holds only rank 0's host. Add `hosts` with all of them.
- `cpu_count` is `os.cpu_count()`, which is the node's core count, not the cores LSF gave the job.
- `peak_memory_mb` is per process. Report the max over ranks.
- `interactions_per_s` must use the summed interaction count (§1) divided by the slowest rank's `forces` time.

**New:** `n_nodes`, `hosts`, `cpu_model`, `decomposition` (`allgather_block`, later `ring`/`force2d`), `allgather_variant` (`pad`/`v`), min/max over ranks for `forces`, `communication`, `merge` and `sync`, `bytes_allgather_per_step`, and `git_commit` (or a hash of the script), so every row can be traced back to the exact code version it came from.

## Appendix C: small things noticed in the uploaded scripts

None of these affect the parallel plan, but they are cheap to fix before the code goes into the hand-in ZIP:

- `GaiaSubsetSelection.py`, line 27 (docstring): a shell command got pasted into the middle of a word: `(chanecho "path/to/bigstuff" >> .gitignorege with --plots-dir)`. It should read "(change with --plots-dir)".
- `DataVisualize.py`, lines 19–34: the branches pick a `cmap` and `clim` per colour source (e.g. log Teff with `[log10 3000, log10 20000]`), but `add_points` always passes `cmap="RdYlBu_r", clim=[-0.5, 3]`. The Teff and spectral-class branches are therefore drawn with the wrong colour scale. The unused `COLOUR` variable (line 8) is left over from an earlier version.
- Distances: `GaiaSubsetSelection.py` uses `1000 / Plx`, while `plot_space*.py` use the `Dist` column and `DataVisualize.py` keeps only `Plx > 1` (within 1 kpc). The visualisations may therefore not show exactly the stars the simulation uses. Worth one sentence in the report.
- `SerialImplementation.py`, line 116: the comment refers to `record_results`, but the function is called `build_results_row`.
