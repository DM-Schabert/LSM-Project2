# Amdahl framework: strong scaling on one node

Measures how the run time falls with the number of MPI ranks P at **fixed work** (same N, same steps) and fits Amdahl's law to it. It reuses the existing code unchanged: `SerialImplementation.py` and `ParallelImplementation.py` do the runs and write their usual row to the shared `results.csv`; this folder only adds a job script that sweeps over P and an analysis script.

| File | What it does |
|---|---|
| `amdahl_batch.sh` | LSF job: reserves a whole node, runs the serial baseline and MPI at P = 1, 2, 4, 8, 12, 16, 24, three times each |
| `amdahl_analysis.py` | reads the `amdahl_` rows of `results.csv` plus `launch_times.csv`, fits Amdahl, makes plots and a LaTeX table |
| `launch_times.csv` | created by the job: wall time of every whole launch (see below) |
| `logs/` | LSF `.out`/`.err` files. The `.err` file has the `--report-bindings` output |
| `out/` | created by the analysis: plots (PNG + PDF), `amdahl_summary.csv`, LaTeX tables |

## Run it

From the repository root on the HPC:

```bash
bsub < amdahl/amdahl_batch.sh                                   # default: N = 5 000, 30 steps, label v1-numpy
DATA=gaia_subsets/gaia_nearest_1000.csv STEPS=300 bsub < amdahl/amdahl_batch.sh   # other N, no editing needed
python3 amdahl/amdahl_analysis.py                               # after the job(s) finished
```

Every setting at the top of `amdahl_batch.sh` can be set from outside like this, because LSF passes the environment on.

| Setting | Default | Meaning |
|---|---|---|
| `LABEL` | `v1-numpy` | name of the code version. **Change it for every new parallelisation** (e.g. `v2-numba`, `v3-ring`) |
| `DATA` | `gaia_nearest_5000.csv` | subset |
| `STEPS` | `30` | the step time hardly varies (±0.05 %), so 30 steps per point are enough |
| `P_LIST` | `1 2 4 8 12 16 24` | rank counts; every P runs on the same reserved node |
| `REPEATS` | `3` | the analysis uses the median per P and shows min–max |
| `SERIAL` | `1` | run `SerialImplementation.py` in every repeat. This is T(1) |
| `MERGE_RADIUS` | `0` | merging off, so the work per step stays constant |
| `WALLTIME_MIN` | `30` | must match `#BSUB -W`. Runs stop themselves before the walltime |
| `RESULTS` | `results.csv` | the shared file. Rows are found again by `run_name = amdahl_<label>_j<jobid>_r<repeat>` |

**Time budget for the defaults** (Xeon E5-2650 v4, N = 5 000): serial ≈ 2.5–2.9 s per step, so 75–90 s per serial run; all MPI runs of one repeat together take about 2.1 × that. One repeat ≈ 4–5 min, three repeats ≈ 13–15 min plus start-up, inside the 30 min walltime. If `hpcintro` does not allow 30 min, lower `-W`, `WALLTIME_MIN` and `REPEATS` together.

**Laptop test** (correctness of the pipeline only; laptop timings say nothing about the cluster):

```bash
LOCAL=1 DATA=gaia_subsets/gaia_nearest_100.csv STEPS=20 P_LIST="1 2 4" REPEATS=1 \
  RESULTS=/tmp/test_results.csv LAUNCH=/tmp/test_launch.csv bash amdahl/amdahl_batch.sh
python3 amdahl/amdahl_analysis.py --results /tmp/test_results.csv --launch /tmp/test_launch.csv --out /tmp/amdahl_out
```

## What is measured

Amdahl: `S(P) = T(1) / T(P) = 1 / (s + (1 − s) / P)`, with `s` the part that does not get faster with more ranks. The analysis uses three wall-clock times:

| Measure | Source | Contains |
|---|---|---|
| step loop | `simulation_time_s` | the time steps only (slowest rank per step) |
| program | `total_time_s` | + loading, distributing, output (rank 0) |
| whole launch | `launch_wall_s` in `launch_times.csv` | + Python start-up, imports, MPI start-up |

The programs cannot time their own start-up, so the job script times every launch from outside. For Python this is often the largest serial part, and it is part of what a user waits for. The main plots use the whole launch.

## Reading the output

- **`speedup_N<n>`**: measured speed-up per version (median, min–max bars), the Amdahl fit (dotted) and the ideal line. The label at the end of each line gives the fitted `s`.
- **`efficiency_N<n>`**: E = S / P.
- **`karp_flatt_N<n>`**: `e(P) = (1/S − 1/P) / (1 − 1/P)`, the serial fraction that each P implies on its own. **Flat** means a fixed serial part, and Amdahl describes the curve. **Rising with P** means an overhead that grows with P (communication, waiting for the slowest rank, shared memory bandwidth), which Amdahl's model does not contain. The fitted `s` is the least-squares value of these points.
- **`measures_<label>_N<n>`**: speed-up of the step loop, the program and the whole launch side by side. The gap between them is what the parts outside the loop cost.
- **`lost_time_<label>_N<n>`**: where the efficiency goes. Each bar is `P·T(P) − T(serial)` in % of the serial run, i.e. `1/E − 1`, split into: extra compute, communication, waiting, the parts outside the step loop, and start-up. Serial parts grow linearly with P here, because all P cores wait while one does them.
- **`amdahl_table_<label>_N<n>.tex`**: booktabs table (P, T, S, E, e) with the fit in the caption.
- The terminal report also gives the gain of every doubling of P (`8→16: ×1.6`) and the first P where the efficiency drops below 70 %. That answers the question in our project description: when does doubling the cores stop paying off?

## Comparing parallelisations

Run the same sweep for every new version with its own `LABEL`, the same `DATA`, `STEPS` and placement. The analysis puts all versions of one N into the same plots. Expect `s` to **grow** when the kernel gets faster (e.g. Numba): the parallel part shrinks, the serial part does not. That is Amdahl's law in one picture.

## Rules that keep the numbers comparable

- **One node per sweep, serial included.** Nodes of the same type differed by about 16 %. Speed-ups are only computed inside one job.
- **Fixed work, not fixed wall time.** With a time limit, larger P simply do more steps. Fixed wall time belongs to Gustafson (weak scaling).
- **Placement stays fixed:** `--map-by core --bind-to core` fills socket 0 first (P ≤ 12 on one socket). Check the `.err` file. A different placement is a different experiment (E3 in the plan).
- No `--plot-iter`, merging off, one thread per rank (`OMP_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`).
- Runs that hit the time limit (`completed = False`) and failed launches are ignored by the analysis.
- With few active cores, the CPU may run at a higher turbo frequency than with all 24 busy. That lowers the efficiency at large P without any serial part. Mention it if e(P) rises at P = 16–24.
