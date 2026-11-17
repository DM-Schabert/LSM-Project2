# Communication schemes: how the positions get to the ranks

Every star feels every other star, so in every time step each rank needs positions it does not own. This folder compares ways of moving them, one at a time, against the same baseline and with the same measurements.

**Question.** How does the communication scheme change the strong scaling of the direct-summation N-body code, for small and large N, on one node and on two?

What we know from the Amdahl campaign (`amdahl/results.html`): on one node and with large N, communication costs almost nothing (N = 5 000, P = 24: about 0.3 ms per step against about 120 ms of computing). With small N it is a large part of each step (N = 50: about a quarter of the step loop). So differences between the schemes will show **at small N, across nodes and at large P**, and hardly at all on one node with large N. Finding that is a result too.

## Layout

| Folder | What goes in |
|---|---|
| `../ParallelImplementation.py` | **all three schemes**, chosen with `--decomposition allgather` (v1, default), `ring` (Pelle) or `2d`. One program, so every scheme shares loading, timers, the stop logic and the results row |
| `2d_decomposition.html` | interactive explanation of the 2D decomposition, with pseudo code, and at the bottom the C2 measurements (expectation against measurement, speed-up, where the time of a step goes); open it in a browser |
| `make_page.py` | writes the C2 numbers from `../results.csv` into the data block of `2d_decomposition.html` |
| `tests/` | C1: `validate.sh` runs every scheme and compares its final state with the serial run (`compare_states.py`) |
| `bench/` | communication alone, no physics: time per call against message size and P, to fit `T = μ + s/B` |
| `jobs/` | LSF job scripts (one node, two nodes) |
| `results/` | raw CSV rows of every run (in git) |
| `logs/` | LSF `.out`/`.err` of the cluster jobs (not in git) |
| `out/` | plots and tables, made again from `results/` (not in git) |

## Rules

1. **Expectation before measurement.** For every scheme, write down below what it should cost (data per rank per step, messages per step, memory per rank) before running it. Afterwards, compare.
2. **Correct before fast.** A scheme is only timed once its accelerations match v1 (`np.allclose`) for several N and P, including N not divisible by P.
3. **One change per iteration.** Each scheme, and each improvement of a scheme (e.g. overlap), is its own iteration with its own label, measured against the previous one on the same experiments.
4. **Everything else fixed.** Same force kernel, same N, steps, `dt`, softening, merging off, one thread per rank, same rank placement (`--report-bindings` in the `.err` file), 3 repeats, median and spread. Serial baseline and all P of one sweep on the same node.
5. **Traceable.** Label in the results = git tag of the code (`v1-numpy`, `v2-ring`, …). Every results row also carries the code fingerprint.
6. **Raw data in git, plots not.** `results/` is committed; `out/` is made again by the analysis.

## Schemes and what we expect

N stars, P ranks, 24 bytes per position (3 doubles). "Received" is per rank and time step.

| Scheme | Label | Received | Messages per step | Positions held per rank |
|---|---|---|---|---|
| Allgather (v1) | `v1-numpy` | 24 N (P − 1)/P | one collective, algorithm chosen by MPI | N |
| Ring, blocking (1D periodic Cartesian, `Sendrecv`) | `v2-ringblk` | 24 N (P − 1)/P | P − 1 point-to-point | 3 N/P (own block + two travelling buffers) |
| Ring with overlap (`Isend`/`Irecv`, Pelle's default) | `v2-ring` | as ring | as ring, hidden behind computing | 3 N/P |
| 2D force decomposition (p_r × p_c grid, row and column communicators) | `v3-2d` | about 24 N (p_r − 1)/P positions (`Allgatherv` in the column) plus 24 N/p_r forces (`Allreduce` in the row) | one `Allgatherv` in a column of p_r ranks, one `Allreduce` in a row of p_c ranks | N/p_r + N/p_c |

Expectations to check:

- On one node, all schemes are about equally fast for large N.
- For small N the ring is slower than Allgather, because it pays P − 1 latencies instead of one collective.
- Overlap helps only where communication is a real share of the step: small N, or two nodes.
- 2D needs a grid close to square to pay off: at P = 24 (6 × 4) each rank receives about 0.2 N positions plus 0.17 N forces instead of N positions; at P = 64 (8 × 8) about 0.11 N plus 0.13 N.
- Memory is not the limit at our sizes: all 626 000 Gaia stars are 15 MB of positions per rank.

**Numbers from C0, written down before measuring the schemes** (one exchange per time step, single-call timing, median, jobs 30045015 and 30045016):

| N | 1 node, P = 24: Allgatherv / ring / 2D (µs) | 2 nodes, P = 48: Allgatherv / ring / 2D (µs) |
|---|---|---|
| 50 | 17 / 43 / 12 | 26 / 92 / 16 |
| 1 000 | 41 / 57 / 27 | 83 / 111 / 30 |
| 10 000 | 190 / 168 / 52 | 579 / 217 / 109 |

- **Ring, 1 node, N = 50:** about 26 µs more per step than v1, on a step loop of about 156 µs. Without other effects the efficiency at P = 24 should drop from 38 % to about 33 %.
- **2D, small N:** about 5 µs less communication per step (1 node), 10 µs less on 2 nodes. In addition, 2D splits the work more finely: at N = 50, P = 24 (6 × 4) the largest tile has 9 × 13 = 117 pairs against a mean of 104 (1.12), while v1's largest block of 3 stars against a mean of 2.08 is 1.44. So 2D should also wait less for the slowest rank at small N.
- **Large N (≥ 1 000), 1 node:** all three within a few percent; the waiting for the slowest rank (2.5–8 % of the step, measured in v1) is larger than any difference in communication.
- **Fair comparison:** compare the schemes with the same kernel, `--tile-rows 0`. With `--tile-rows 1`, the ring loops over its rows P times per step and 2D p_c times as often as v1, so the Python loop, not the communication, would decide.

## Experiments

| # | What | N | P | Nodes |
|---|---|---|---|---|
| C0 | Communication alone: `Allgatherv`, ring of `Sendrecv`, `Bcast` + `Allreduce`. Fit latency μ and bandwidth B | message sizes of 50 to 100 000 stars | 2 to 24 (48) | 1 and 2 |
| C1 | Correctness of every scheme against the serial run (`tests/validate.sh`) | 5, 100, 1 000 (5 < P: empty blocks) | 1, 2, 3, 4, 6, 7, 8, 12, 16, 24 | 1 |
| C2 | Strong scaling per scheme (with the Amdahl framework, one label per scheme) | 50, 100, 1 000, 5 000 | 1 … 24 | 1 |
| C3 | The same across two nodes | 1 000, 5 000, 20 000 | 8 … 48 | 2 |
| C4 | Overlap on and off (ring) | 50, 1 000 | 8, 24 | 1 and 2 |
| C5 | Memory per rank against N (peak RSS) | up to the largest that fits | 24 | 1 |

## Running C0 and getting the results off the cluster

We cannot push from the cluster, so every job writes **one CSV file** that is copied to the laptop and committed there.

On the cluster, from the repository root:

```bash
git pull
bsub < communication/jobs/bench_1node.sh     # 24 ranks on one node, about 2 min
bsub < communication/jobs/bench_2nodes.sh    # 2 x 24 ranks on two nodes, about 3 min
bjobs                                        # PEND = waiting, RUN = running
ls communication/results/                    # bench_<jobid>.csv, one per job
```

On the laptop, from the repository root (user name and repository path on the cluster as yours):

```bash
scp <user>@<login node>:<repo on the cluster>/communication/results/bench_*.csv communication/results/
python3 communication/bench/analyse_bench.py   # needs pandas and matplotlib
git add communication/results/bench_*.csv && git commit -m "C0 results"
```

The analysis writes `communication/out/bench/`: `pingpong` (latency and bandwidth with the fit T = μ + s/B), `schemes_N` and `schemes_P` (time of one exchange of each scheme against N and against P), `fits.csv` and `report.txt`. The report also predicts the ring from the pingpong fit, (P − 1)(μ + b/B), next to the measured times: expectation against measurement.

What the benchmark does and how it times is described at the top of `bench/comm_bench.py`. In short: warm-up calls first; every repeat starts after a `Barrier` and counts with the slowest rank's time (`time.perf_counter`); median and quartiles over the repeats. Collectives are timed one call per repeat (`timing_mode = single`, the main measure, as OSU times collectives) and, for comparison, back to back (`batched`). Back-to-back collectives overlap and look faster than in a time step, where computing separates two exchanges. Pingpong is only timed back to back, as usual. The job's `.err` file shows where the ranks were bound (`--report-bindings`).

The first C0 runs (jobs 30040944 and 30040946, 13 Nov 2026) timed the collectives only back to back. Their pingpong numbers are fine; their collective numbers are optimistic for small messages.

Laptop test, no cluster needed (pretends there are two nodes):

```bash
mpirun -np 4 python3 communication/bench/comm_bench.py --fake-nodes 2 --sizes "50 1000" --target-s 0.02 --out /tmp/bench_test.csv
python3 communication/bench/analyse_bench.py /tmp/bench_test.csv --out /tmp/bench_out
```

## Running C1, C2 and C3 on the cluster

**C1, correctness first** (one job, about 10 min; result `communication/results/validate_<jobid>.csv`):

```bash
bsub < communication/tests/validate.sh
```

It runs the serial program and every scheme (allgather, allgather tiled, ring overlap/blocking/tiled, 2d, 2d tiled, 2d with the other grid shape) for N = 5, 100, 1 000 at P = 1 … 24, and compares the final states. v1 must be bit-identical, the others within 1e-9 relative. On a laptop: `LOCAL=1 P_LIST="1 2 3 4 6" bash communication/tests/validate.sh`.

**C2 and C3, strong scaling with the Amdahl framework.** One campaign per scheme, all with the same kernel (`--tile-rows 0`); `PROGRAM_ARGS` goes to the MPI runs only, the serial baseline stays the same. For a quicker comparison, three sweeps are enough (`SWEEPS` below: N = 50, 1 000, 5 000):

```bash
export SWEEPS="50 20000 3 15
1000 300 3 15
5000 30 3 30"
LABEL=v1-tile         PROGRAM_ARGS="--tile-rows 0"                                         bash amdahl/run_all.sh
LABEL=v2-ring-tile    PROGRAM_ARGS="--decomposition ring --tile-rows 0"                    bash amdahl/run_all.sh
LABEL=v2-ringblk-tile PROGRAM_ARGS="--decomposition ring --ring-comm blocking --tile-rows 0" bash amdahl/run_all.sh
LABEL=v3-2d-tile      PROGRAM_ARGS="--decomposition 2d --tile-rows 0"                      bash amdahl/run_all.sh
# C3: the same with NODES=2 in front (two full nodes, P up to 48), e.g.
NODES=2 LABEL=v3-2d-tile-2n PROGRAM_ARGS="--decomposition 2d --tile-rows 0" bash amdahl/run_all.sh
```

Every campaign also starts its own analysis job. To compare the schemes, copy `results.csv` and `amdahl/launch_times.csv` to the laptop and run, for example, `python3 amdahl/amdahl_analysis.py --label v1-tile v2-ring-tile v2-ringblk-tile v3-2d-tile --out amdahl/out/schemes`: the per-N plots then show all schemes together. The `decomposition` column in `results.csv` says which scheme and grid each MPI row used (`allgather_block`, `ring_overlap`, `ring_blocking`, `2d_6x4`, …).

For the page, `python3 communication/make_page.py` (default: the four labels above) takes the latest job per label and N and writes the medians into `2d_decomposition.html`. The page compares every scheme with its own run at P = 1, because each sweep ran on a different node and the nodes differ by up to 21 % (serial run at N = 50).

## Status

C1 on the cluster (job 30082644, `results/validate_30082644.csv`): 228 runs, N = 5, 100, 1 000, P = 1 … 24, 20 steps, all passed. The positions agree exactly after 20 steps (they move too little to show a difference); the velocities carry the difference. At N = 1 000 every scheme that sums in a different order ends at the same 1.3e-13: rounding differences of about 1e-16 grow over the 20 steps, most likely through a close pair of stars, and the growth is the same whatever the summation order. That is four orders of magnitude below the tolerance.

| Scheme | Expectation written | Implemented | Correct (C1) | Measured, 1 node | Measured, 2 nodes |
|---|---|---|---|---|---|
| Allgather (v1) | yes (`docs/Parallel_Plan_v1(1).md`, §3; C0) | yes | **cluster: bit-identical** to serial, 30/30 runs (job 30082644) | yes (`amdahl/`, `--tile-rows 1`) | no |
| Ring, blocking and overlap | yes (above, from C0) | yes (Pelle, 15 Nov) | **cluster: 90/90 passed**, ≤ 1.3e-13 relative (job 30082644) | C2 (jobs 30083334–40): N = 50 collapses, about 70 µs per kernel call × P calls per step; N ≥ 1 000 12–21 % below v1 | no |
| 2D force decomposition | yes (above, from C0) | yes (16 Nov) | **cluster: 78/78 passed**, bit-identical for p_c = 1, else ≤ 1.3e-13 relative (job 30082644) | C2 (jobs 30083342–44): N = 50 14 % better than v1, N ≥ 1 000 5–8 % below | no |

C2 on one node (jobs 30083330–30083344, `--tile-rows 0`, `v1-tile` as the reference) is shown at the bottom of `2d_decomposition.html`. The C0 prediction for the ring missed by a factor of about 90 at N = 50: C0 timed the messages only, but the ring calls the force kernel P times per step and each call has a fixed cost of about 70 µs in Python. At large N the differences between the schemes also come from the kernel (summed kernel time of all ranks at N = 5 000, P = 24, relative to P = 1: v1 1.29, 2D 1.37, ring 1.63), not from the bytes sent.
