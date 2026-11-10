# `ParallelImplementation.py` and `batch.sh`: how they work

Summary of the MPI version (v1) of the Gaia N-body simulation and its LSF job script, as of 10 November 2026. The design reasoning is in `claude/Parallel_Plan_v1.md`. v1 uses only material from weeks 1–6.

## The idea in one paragraph

Every rank owns a contiguous block of stars and only moves those. Before every force calculation, all ranks share their positions with one `Allgatherv`, so each rank holds all N positions. It then computes the forces on its own stars from every star, using exactly the serial force function. Merging, stopping and output are collective decisions, so all ranks always take the same steps together. The results are bit-identical to `SerialImplementation.py`.

## How to run it

```bash
# laptop (correctness only, never timing)
mpirun --oversubscribe -np 4 python3 ParallelImplementation.py --data gaia_subsets/gaia_nearest_1000.csv --steps 20

# HPC: edit the settings at the top of batch.sh, then
bsub < batch.sh
```

`ParallelImplementation.py` imports `SerialImplementation.py`, so the two must be in the same folder.

## Changes to `SerialImplementation.py` that the MPI version relies on

- **`compute_accelerations(..., i_start=0, i_stop=None)`** computes forces only on stars `i_start` to `i_stop - 1`, from all stars. The serial code uses the default (all stars), so its results are unchanged. Inside the loop `i` stays the global star index, because the self-force exclusion and the j > i pair search depend on it.
- **`load_subset`** returns positions in row-major (C) order. pandas gave column-major order, which MPI (raw memory) would scramble and which also changed the summation order.
- **`results_lock`** and **`append_results`** let several jobs write to one `results.csv` at the same time without losing rows.

## Functions in `ParallelImplementation.py`

### Distribution

- **`Blocks`** stores which stars each rank owns: rank r owns stars `displs[r]` up to `displs[r] + counts[r]`.
- **`Blocks.range_of`** returns the first and one-past-last star index of a rank.
- **`make_blocks`** splits N stars into P contiguous blocks whose sizes differ by at most one, like `np.array_split`. With fewer stars than ranks, some ranks own nothing but still take part in every collective.
- **`ParallelState`** is the simulation state as one rank sees it:
  - all positions, replicated (only valid right after an allgather)
  - its own positions and velocities
  - the replicated masses, member counts, owner map and spectral classes
  - the block layout and its rank
- **`state_from_stars`** splits a full `Stars` object, identical on every rank, into this rank's `ParallelState`. It is used at start-up and after every merge.
- **`allgather_rows`** gathers every rank's (n_own, 3) block into the full (N, 3) array on every rank with one `Allgatherv`. Counts are in doubles, 3 per star. This is the function to change in week 8 for a custom "xyz" datatype.
- **`gather_full_state`** is a collective that gives every rank the complete current positions and velocities. It is used for the live plot and the final save.

### Start-up

- **`load_and_distribute`**:
  - Rank 0 reads the CSV and draws the guessed classes, so only one rank touches the file system.
  - Positions, velocities, masses, member counts and owners go to all ranks with upper-case `Bcast`; the class letters go once with lower-case `bcast`.
  - If rank 0 cannot load the file, it broadcasts the error and every rank exits, so nobody is left waiting.
  - `--data` chooses the subset CSV. `--seed` sets the random classes, drawn once on rank 0.

### Physics (the parallel parts)

- **`parallel_accelerations`** is a collective. It refreshes all positions with `allgather_rows` (timed as `communication`), then calls the serial `compute_accelerations` on the own block (timed as `forces`). It counts only its own interactions, n_own × (N − 1); rank 0 later sums them.
- **`close_pairs_anywhere`** is a collective `Allreduce(SUM)` of the number of close pairs. It replaces the serial `while close_pairs:` so that every rank makes the same merge decision.
- **`parallel_merge`** is a collective:
  - It gathers all pair lists (lower-case `allgather`) and all velocities (`Allgatherv`) to every rank.
  - Every rank then runs the serial `merge_stars` on identical data and rebuilds its blocks for the new N.
  - Each pair is found exactly once, by the owner of the smaller index, so the joined list equals the serial one.
- **`parallel_step`** runs one time step in the serial order: forces, then (merge, forces again) until no close pairs remain, then kick and drift of the own block.
  - `--dt` sets the time step.
  - `--softening` and `--merge-radius` are passed to the force function. With `--merge-radius 0` the merge check is skipped on every rank.

### Stopping together

- **`ParallelStopRequest`** turns Ctrl+C and SIGTERM into a *request* to stop. Unlike the serial version, a second Ctrl+C does not stop the rank at once, because one rank leaving in the middle of a collective would leave the others waiting forever.
- **`agree_on_stop`** is a collective `Allreduce(MAX)` of each rank's stop code. The result is one of: carry on, interrupted, window closed, or time limit. All ranks therefore stop after the same step.
  - Only rank 0 checks the clock and the plot window.
  - `--quick`, `--time-limit` and the normal 5-minute limit set the deadline; stepping stops at 95 % of it to leave time for saving.
  - `--stop-check-every K` agrees on stopping every K steps instead of every step.

### Results

- **`cpu_model`** reads the CPU model name from `/proc/cpuinfo`, for the report.
- **`code_fingerprint`** returns a short hash of `ParallelImplementation.py` and `SerialImplementation.py`, so every results row can be traced to a code version.
- **`reduce_perf`** is a collective that combines every rank's timings on rank 0:
  - Section times and step times become the **MAX** over ranks, because the slowest rank decides when a step is done.
  - Interaction counts and communication bytes are **summed**; peak memory is the max.
  - It adds min and mean columns per section, `load_imbalance` (max/mean of `forces`), hosts, number of nodes, CPU model, LSF job ID and the code fingerprint.
- **`save_final_state`** writes the final state with the same columns as the serial `--save-final`, so the two files can be compared directly.
  - `--save-final` turns it on.

### Command line and main

- **`parse_args`** has the same options, names and defaults as the serial script, plus `--stop-check-every`, so the same command line works for both.
- **`main`**:
  1. Prints where the ranks run and loads and distributes the stars.
  2. Synchronises once with `Barrier`, then runs steps until `--steps`, the time limit or a stop request.
  3. Gathers the final state and reduces the timings.
  4. On rank 0 only: prints the summary, saves `--save-final`, the `--plot` figure and the results row.
  - `--plot` makes the start → end plot on rank 0 after the run.
  - `--plot-iter N` makes a live plot every N steps, at the cost of a gather each time; never use it in benchmarks.
  - `--results` chooses the results CSV and `--no-results` skips writing.
  - `--run-name` replaces the automatic name, `mpi_<mode>_<data>_<P>ranks`.
- **`stars_at_start`** wraps the replicated arrays as a `Stars` object for the start-up printout (mean distance to the centre of mass).
- **The `__main__` block** calls `MPI.COMM_WORLD.Abort` on any unhandled error, so a crash on one rank ends the whole job instead of leaving the others waiting until the walltime.

### Timed sections

Each section is timed per rank; `results.csv` shows the max over ranks.

| Section | What it contains |
|---|---|
| `load` | rank 0 reading the CSV |
| `distribute` | broadcasting the initial state |
| `communication` | `Allgatherv` of positions before every force calculation |
| `forces` | `compute_accelerations` on the own block |
| `merge` | the whole merge path, including its gathers |
| `integrate` | kick and drift of the own block |
| `sync` | stop-flag and pair-count `Allreduce`s. These mostly measure faster ranks waiting for the slowest one |
| `plotting`, `output` | rank 0, plus the gathers they need |

## `batch.sh`

### LSF settings (`#BSUB` lines)

Job name `nbody_mpi`, queue `hpcintro`, 8 slots on one node (`span[hosts=1]`), 2 GB memory per slot, 15 minutes walltime. Output goes to `nbody_mpi_<jobid>.out` and `.err`.

- **Changing ranks or placement:** edit these lines, or override them on the command line, e.g. `bsub -n 16 < batch.sh`. LSF reads them before the script runs, so shell variables don't work in them.
- **Placement recipes from week 2:**
  - one node: `span[hosts=1]`
  - two nodes: `span[ptile=P/2]`
  - spread over sockets: reserve the full node (`-n 24`) and set `NP` and `MAP_OPTS` below.

### Settings at the top of the script

| Variable | Default | Meaning |
|---|---|---|
| `MODE` | `benchmark` | `benchmark`: one MPI run. `validate`: serial run, MPI run, then compare the final states |
| `DATA` | `gaia_subsets/gaia_nearest_5000.csv` | subset CSV |
| `STEPS` | `100` | number of steps; choose from the serial baseline, steps ≈ 200 s × (interactions/s) / N² |
| `DT`, `SOFTENING` | `0.01`, `0.01` | as in the simulation |
| `MERGE_RADIUS` | `0` | merging off for benchmarks (the program's own default is 0.01) |
| `WALLTIME_MIN` | `15` | must match `#BSUB -W` |
| `TIME_LIMIT` | walltime − 120 s | the program stops itself and saves before LSF kills the job |
| `NP` | empty | empty means one rank per reserved slot; set it to start fewer ranks than reserved |
| `MAP_OPTS` | empty | e.g. `--map-by ppr:4:package --bind-to core` (needs `NP`) |
| `RESULTS` | `results.csv` | the one shared results file |

### What the script does

1. Loads the course modules: pandas, matplotlib and mpi4py with Open MPI 5.0.8.
2. Sets `OMP_NUM_THREADS=1` (and the OpenBLAS/MKL equivalents) so each rank uses exactly one core, `MPLBACKEND=Agg` for plotting without a display, and `PYTHONUNBUFFERED=1` so output appears in order.
3. Records the setup in the `.out` file: job ID, slots and hosts, CPU model, all settings and the loaded modules.
4. Stops with a hint if the data file is missing.
5. Runs `mpirun --report-bindings`, so the `.err` file shows which core each rank was pinned to.
   - **benchmark:** runs `ParallelImplementation.py` once.
   - **validate:** runs `SerialImplementation.py` and `ParallelImplementation.py` with the same options, saves both final states, and prints `Validation: N stars, identical=..., max abs difference=...`. Without merging, `identical=True` is expected. From the week 7 iterations on, the forces are added in a different order, so judge the max difference against a small tolerance instead.

### Why the time limit matters

In testing, a SIGTERM sent to `mpirun`, as `bkill` or the walltime limit may do, killed the ranks before they could save their results. A SIGTERM that reaches the ranks themselves is handled correctly. The reliable stop is therefore the program's own `--time-limit`, which `batch.sh` always sets below the walltime.

## Typical workflow

1. Make the subsets once: `python3 GaiaSubsetSelection.py --csv gaia_cleaned.csv --no-plot --save`.
2. After every change to the parallel code: `MODE=validate` with a small N.
3. Once per N, step count and CPU type: a serial baseline (validate mode gives one).
4. Timing runs: `MODE=benchmark`, at least 3 repeats per point.
5. Plots and speed-up table: `python3 plot_results.py`. Speed-ups are only computed between serial and MPI runs on the same node.
