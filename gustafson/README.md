# Gustafson framework: weak scaling on one node

Measures what happens when the **problem grows with the number of MPI ranks P**, so that every rank keeps the same amount of work as the base problem with N0 stars. Gustafson's law describes this: the scaled speed-up S(P) = s + (1 − s)·P grows linearly in P instead of saturating as in Amdahl's law. Like `amdahl/`, it reuses the existing code unchanged: `SerialImplementation.py` and `ParallelImplementation.py` do the runs and write their usual row to the shared `results.csv`. This folder only adds the job scripts, a size planner and an analysis.

| File | What it does |
|---|---|
| `run_all.sh` | the whole campaign: **the experiment list** (`SWEEPS`, one sweep per line, each as its own job) plus an analysis job that starts when all have ended |
| `gustafson_batch.sh` | LSF job for one sweep: reserves a whole node; for every P = 1, 2, 4, 8, 12, 16, 24 it runs MPI on N(P) stars with P ranks and the serial program on the same N(P) stars, three times |
| `plan.py` | N(P) for both growth rules, the time estimate of a sweep, and the fit of the serial cost model (standard library only, runs on the login node) |
| `gustafson_analysis.py` | reads the `gustafson_` rows of `results.csv` plus `launch_times.csv`: scaled speed-up, weak efficiency, Gustafson fit, plots, LaTeX tables |
| `results.html` | **the results page**, same style as `amdahl/results.html`. Empty until `make_page.py` has filled it |
| `make_page.py` | puts the numbers of the analysis (and optionally the Amdahl campaign, for the strong-vs-weak comparison) into `results.html` |
| `launch_times.csv` | written by the jobs: wall time of every launch. Copy it from the cluster together with `results.csv` |
| `logs/`, `out/` | not in git (see `.gitignore`): LSF output, the submitted job scripts and `jobs.txt`; plots, CSVs, LaTeX tables and `report.txt` |

## Run it

From the repository root on the HPC login node:

```bash
DRY_RUN=1 bash gustafson/run_all.sh     # shows N(P) of every sweep and which subsets are missing
python3 GaiaSubsetSelection.py --n <the list it prints> --save --no-plot   # where gaia_cleaned.csv is
bash gustafson/run_all.sh               # 6 sweeps + the analysis, results in gustafson/out/<campaign>/
```

The sweeps need about 40 new subset sizes (`gaia_subsets/gaia_nearest_<N>.csv`, the same nested nearest-N selection as before). `run_all.sh` stops with the exact `GaiaSubsetSelection.py` command if any are missing. `GaiaSubsetSelection.py` imports plotly, so run it where you made the existing subsets and copy the new files to `gaia_subsets/` on the cluster if that was your laptop.

As with Amdahl, the cluster does not pass your environment on to a job, so `run_all.sh` writes each sweep's settings as `export` lines into a copy of the batch script (kept in `gustafson/logs/<campaign>/`). For one sweep by hand:

```bash
mkdir -p gustafson/logs                 # LSF needs it for the .out/.err files (run_all.sh makes it)
bsub < gustafson/gustafson_batch.sh     # defaults: rule time, N0 = 1 000, 150 steps
sed -e 's|N0:-1000|N0:-300|' -e 's|STEPS:-150|STEPS:-1200|' gustafson/gustafson_batch.sh | bsub
```

Then the page, after the analysis job has finished:

```bash
python3 gustafson/make_page.py --dir gustafson/out/<campaign> --amdahl amdahl/out/v1-numpy
```

`--amdahl` is the Amdahl analysis folder of the same code version (`by_N_<label>.csv` and `amdahl_summary.csv`). With it, the page adds the "same N, fixed or grown with P" comparison for every N that both campaigns ran (50, 100, 1 000, and 300 once the Amdahl N = 300 sweep exists). The page loads d3 from cdnjs, so it needs internet.

## The experiment list

`SWEEPS` at the top of `run_all.sh`. Every sweep runs about 15–16 min (estimated with `plan.py` from the Amdahl times), with 30 min walltime:

| rule | N0 | N at P = 1, 2, 4, 8, 12, 16, 24 | steps (MPI and serial base) | why |
|---|---|---|---|---|
| time | 50 | 50, 87, 146, 233, 303, 361, 461 | 14 000 | overhead-dominated base problem; Amdahl gave 38 % at N = 50 |
| time | 100 | 100, 165, 262, 403, 512, 605, 761 | 6 000 | Amdahl gave 51 % at N = 100 |
| time | 300 | 300, 458, 683, 1 004, 1 251, 1 460, 1 810 | 1 200 | between the small and large regime |
| time | 1 000 | 1 000, 1 454, 2 096, 3 005, 3 703, 4 292, 5 279 | 150 | compute-dominated; N(24) close to the Amdahl N = 5 000 |
| pairs | 100 | 100, 141, 200, 283, 346, 400, 490 | 6 000 | the N0·√P rule, to show why it misleads at small N |
| pairs | 1 000 | 1 000, 1 414, 2 000, 2 828, 3 464, 4 000, 4 899 | 150 | the N0·√P rule where it is nearly right |

Steps are chosen so that every run lasts about 20 s. The serial runs on the scaled problems use max(10, ⌈steps / P⌉) steps, so they also last about 20 s; only times per step are compared. `python3 gustafson/plan.py --n0 300 --steps 1200 --walltime 30` prints the table and estimate for one line.

## How N grows with P

One step of direct summation costs a·N + b·N² on one core: **a** per star (NumPy's work per row of the force loop) and **b** per pair. Fitted to the four serial step times of the Amdahl campaign: a = 23.5 µs, b = 113.5 ns, within ±3.4 % of every point. At N = 100 the per-star part is two thirds of the step, at N = 5 000 under 4 %.

- **time** (main rule): N(P) solves a·N + b·N² = P·(a·N0 + b·N0²). Every rank's share then costs the serial time of the base problem, which is Gustafson's assumption ("the total parallel time in the scaled problem is p"). Ideal: E = 1 and S = P.
- **pairs**: N = N0·√P, the textbook rule for O(N²) work and the one `code/plot_results.py` uses. It keeps the pair interactions per rank fixed, but each rank gets fewer, longer rows than the base problem, which costs less time. For N0 = 100 the serial work at P = 24 grows only ~11× instead of 24×, so the cost model predicts a weak efficiency near 190 % without any parallel effect. The two `pairs` sweeps are there to show this.

A new code version has its own a and b. Fit them from its Amdahl campaign and pass them on:

```bash
python3 gustafson/plan.py --fit amdahl/out/<campaign>/by_N_v2-numba.csv      # prints --a-us .. --b-ns ..
LABEL=v2-numba COST_A_US=.. COST_B_NS=.. bash gustafson/run_all.sh
```

## What is measured

Per time step (`simulation_time_s / iterations`, the slowest rank per step), medians over the repeats:

| Symbol | Measured as | Meaning |
|---|---|---|
| T(P) | MPI program, N(P) stars, P ranks | parallel time of the scaled problem |
| τ(P) | serial program, N(P) stars | serial time of the scaled problem; τ(1) = T(1), the serial program on N0 |
| E(P) = T(1) / T(P) | | **weak efficiency** (week 8, slide 5). Ideal 100 % |
| S(P) = τ(P) / T(P) | | **scaled speed-up** (Gustafson). Ideal P |
| g(P) = τ(P) / τ(1) | | how much the serial work really grew. Ideal P; S = g · E |
| s(P) = (P − S) / (P − 1) | | serial fraction implied by one P: the weak-scaling counterpart of Karp–Flatt |

The Gustafson fit is the least-squares s of P − S = s·(P − 1) over all P > 1; the same fit to the most and least favourable pairing of repeats gives its range. Gustafson's s is a share of the **parallel** run's time; Amdahl's s is a share of the **serial** run's time, so the two numbers are not the same thing.

E is also given for the program (`total_time_s`) and the whole launch (`launch_wall_s`), comparing whole MPI runs with the whole serial base run (same number of steps). They include the start-up and the loading and distributing of the stars, which grow with N.

## Reading the output

In `gustafson/out/<campaign>/` (all as PNG and PDF):

- **`scaled_speedup_<label>_<rule>`**: S(P) per N0 with the Gustafson fit (dotted) and S = P. The label at each line end gives s.
- **`weak_efficiency_<label>_<rule>`**: E(P) per N0, min–max bars.
- **`gustafson_fraction_<label>_<rule>`**: s(P). Flat = a fixed serial part per step, as Gustafson assumes. Rising = a cost that grows with P (Allgather of N(P) positions, waiting for the slowest rank).
- **`growth_<label>`**: τ(P)/τ(1) for every sweep; shows how far the pairs rule is from P times the work.
- **`rules_<label>_N0_<n>`**: both rules side by side for the same N0.
- **`lost_time_<label>_<rule>_N0_<n>`**: extra time per step against the base problem, T(P) − T(1) in % of T(1) (= 1/E − 1), split into compute of the own rows, communication and waiting. With the time rule the compute part should stay near zero.
- **`measures_<label>_<rule>_N0_<n>`**: E of the step loop, the program and the whole launch.
- **`gustafson_table_<label>_<rule>_N0_<n>.tex`**, **`by_N0_<label>.tex/.csv`**: booktabs tables for the report.
- **`report.txt`**: everything above as text, per sweep and summarised.

## Rules that keep the numbers comparable

- **One node per sweep, serial included.** Serial and MPI runs of the same P run back to back on the same node. E and S are only computed inside one job.
- **Fixed work per rank, not fixed wall time:** N(P) and the steps are set before the run; runs that hit the time limit (`completed = False`) and failed launches are ignored.
- Same placement as the Amdahl sweeps (`--map-by core --bind-to core`, socket 0 first), merging off, one thread per rank, no `--plot-iter`.
- The memory per process stays small: the force loop works row by row (O(N) temporaries), so N = 5 279 needs no extra memory request.
- With few busy cores the CPU may turbo higher than with all 24 busy. In weak scaling that lowers E at large P without any parallel overhead. Mention it if E drops at P = 16–24 while communication stays small.

## Laptop test

Correctness of the pipeline only (laptop timings say nothing about the cluster):

```bash
LOCAL=1 N0=20 STEPS=200 P_LIST="1 2 4" REPEATS=1 \
  RESULTS=/tmp/test_results.csv LAUNCH=/tmp/test_launch.csv bash gustafson/gustafson_batch.sh
python3 gustafson/gustafson_analysis.py --results /tmp/test_results.csv --launch /tmp/test_launch.csv --out /tmp/gustafson_out
```

It needs `gaia_nearest_20/37/66.csv` (`python3 gustafson/plan.py --n0 20 --p 1 2 4 --sizes` lists them).

## Not covered

More than one node, N0 above 1 000 (N0 = 5 000 needs N(24) ≈ 25 000 stars, ~70 s per serial step and ~45 min per repeat; `SERIAL=0` would skip the serial runs of the scaled problems, leaving only E), and other parallelisations (run the same list with a new `LABEL`).
