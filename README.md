# isoeff: adaptive Amdahl → Gustafson campaign

Run this once for every code version that might be an improvement. It finds out how
far the version scales and how much N has to grow to keep adding ranks worthwhile.
Different versions can then be compared on one plot.

```bash
cd ~/.../code                                          # the folder with ParallelImplementation.py
chmod +x isoeff/start.sh                               # once
./isoeff/start.sh v1-numpy --note "baseline, Allgatherv"
python3 isoeff/controller.py status --campaign isoeff/out/v1-numpy --log
```

## The rule

1. **Amdahl (strong scaling).** Keep N fixed and step P up: 1, 2, 4, 8, 12, 16, 24, then whole nodes (48, 72, …).
2. Each step `P_prev → P` must be **cost-effective**:
   - marginal efficiency `(S(P)/S(P_prev)) / (P/P_prev) ≥ 0.8` (doubling P must give at least 1.6× more speed-up), **and**
   - parallel efficiency `E = S/P ≥ 0.5`.

   `S` is measured against the serial program, run in the same job on the same node.
3. **Gustafson.** When a step fails, `P_prev` is that N's threshold. N then grows (×2 by default), and the same step is tested again. If it now passes, P keeps going up.
4. **Stop conditions:**
   - the largest P is still cost-effective (Amdahl's limit lies beyond the hardware; add nodes),
   - the next N does not fit in a 7-minute job,
   - or a point fails twice.

A failing step is measured a second time before it is accepted, so one noisy job cannot set a threshold.

## How it runs

- One campaign = one chain of 7-minute LSF jobs. Each job measures one (N, P) point:
  - the serial baseline,
  - then 3 MPI repeats,
  - each run bounded by the program's own `--time-limit`.

  At the end, the job runs the controller, which decides the next point and `bsub`s it.
- **Subsets are made automatically** (`gaia_subsets/gaia_nearest_<N>.csv`):
  - `make_subset.py` writes exactly the same file as `GaiaSubsetSelection.py`, but needs no plotly. It uses `GaiaSubsetSelection` itself when that can be imported.
  - The controller makes the next subset before it submits the job, so no walltime is spent on it.
- **Code snapshot.** `init` copies all `*.py` from the code folder into the campaign. You can keep editing the next version while a campaign runs.
- **Time plan.** The time limits come from the campaign's own serial cost fit, `T(N) = a·N + b·N²`. If N gets large, fewer MPI repeats are run. With 7 minutes, the largest N on one node is about 16 000 (the serial baseline is the limit there).
- **Resumable.** The decision is replayed from `results.csv` every time, so `next` never repeats a point that has already been measured. Use it to:
  - resume after a broken chain (it detects jobs killed at the walltime and resubmits),
  - extend a finished campaign (`set`, then `next`).

## Commands

| What | Command |
|---|---|
| start | `./isoeff/start.sh <label> [--note ...] [--n0 500] [--growth 2] [--max-nodes 1] [--marginal 0.8] [--efficiency 0.5]` |
| progress | `python3 isoeff/controller.py status --campaign isoeff/out/<label> [--log]` |
| resume | `python3 isoeff/controller.py next --campaign isoeff/out/<label>` |
| more nodes later | `python3 isoeff/controller.py set --campaign isoeff/out/<label> max_nodes=2`, then `next` |
| compare versions | `python3 isoeff/plot.py isoeff/out/v1-numpy isoeff/out/v2-ring` |
| test without LSF | `init ... --executor local --mpirun mpirun` (runs every job right there) |
| the ring (v2) | `./isoeff/start.sh v2-ring --mpi-args "--decomposition ring"` |

`--mpi-args` goes to `ParallelImplementation.py` only (the serial baseline doesn't know these options). Write a single option as `--mpi-args=--tile-rows=0`, so the shell doesn't read it as an option of its own.

**Comparing versions fairly:** the serial baseline always uses the v1 kernel, which handles one star per loop iteration. `--tile-rows 0` speeds up the kernel itself (about 2× at small N in testing). A campaign with tiling therefore reports speed-ups that include this kernel gain, not only the parallel gain. To isolate the ring's effect, compare runs that use the same kernel:
- `v1` (allgather, tile 1) vs. `v2-ring` (ring, tile 1)
- `v1-tile` (allgather, tile 0) vs. `v2-ring-tile` (ring, tile 0)

Other settings, all stored in `config.json` and changeable with `set`:

- `at_max_p=grow`: keep growing N at the largest P (pure weak scaling)
- `mpi_reps`, `min_steps`, `reserve_s`, `queue`, `map_opts` (e.g. `"--map-by core --bind-to core"`)
- `merge_radius`: 0 by default, so N and the work per step stay constant during a run

## Output (`isoeff/out/<label>/`)

- `points.csv`: one row per (N, P): serial and MPI ms/step, S, E, marginal efficiency, Karp–Flatt serial fraction, verdict, hosts
- `thresholds.csv`: the last cost-effective P for every N where a step failed
- `decisions.log`: every decision in words, with the cost-model fit
- `plots/`:
  - speed-up vs. P and efficiency vs. P, one line per N
  - the isoefficiency path: for each P, the N at which P became cost-effective
- `results.csv`, `logs/`, `jobs/` (the exact job scripts), `jobs.json`, `DONE` (why it stopped)

## Before trusting a campaign

- Run `MODE=validate` with `batch.sh` on the new version first. This campaign only measures speed, not correctness.
- Check `--report-bindings` in the first job's `.err` file.
- The chain needs `bsub` to work from inside a job. If the first job's `.out` ends in "bsub failed", run `next` from a login node after each job. The resume detection handles that.
