# Amdahl framework: strong scaling on one node

Measures how the run time falls with the number of MPI ranks P at **fixed work** (same N, same steps) and fits Amdahl's law to it. It reuses the existing code unchanged: `SerialImplementation.py` and `ParallelImplementation.py` do the runs and write their usual row to the shared `results.csv`; this folder only adds a job script that sweeps over P and an analysis script.

| File | What it does |
|---|---|
| `run_all.sh` | the whole campaign: one sweep per N (50, 100, 300, 1 000, 5 000), each as its own job, plus an analysis job that starts when all have ended |
| `amdahl_batch.sh` | LSF job for one N: reserves a whole node, runs the serial baseline and MPI at P = 1, 2, 4, 8, 12, 16, 24, three times each |
| `amdahl_analysis.py` | reads the `amdahl_` rows of `results.csv` plus `launch_times.csv`, fits Amdahl, makes plots and LaTeX tables |
| `results.html` | **the results page**: every chart and table of the latest campaign, opens in any browser (loads d3 from cdnjs, so it needs internet) |
| `make_page.py` | puts the numbers of a new analysis into `results.html` (see below) |
| `launch_times.csv` | written by the jobs: wall time of every whole launch (see below). Copy it from the cluster together with `results.csv` |
| `logs/` | not in git. LSF `.out`/`.err` files. The `.err` file has the `--report-bindings` output. `logs/<campaign>/` keeps the exact job scripts `run_all.sh` submitted and `jobs.txt` |
| `out/` | not in git, made again by the analysis: plots (PNG + PDF), `amdahl_summary.csv`, LaTeX tables, `report.txt` |

## Run it

From the repository root on the HPC login node:

```bash
bash amdahl/run_all.sh                 # everything: 5 sweeps + the analysis, results in amdahl/out/<campaign>/
DRY_RUN=1 bash amdahl/run_all.sh       # only write and show the job scripts
LABEL=v2-numba bash amdahl/run_all.sh  # the same campaign for a new code version
```

Each sweep runs on its own node with its own serial baseline, so the jobs can run at the same time. When all of them have ended, the analysis job writes `amdahl/out/<campaign>/report.txt` and the plots. The N, steps, repeats and walltimes are the `SWEEPS` table at the top of `run_all.sh`.

**The DTU cluster does not pass your environment on to a job**, so `DATA=... bsub < amdahl/amdahl_batch.sh` silently runs the defaults. `run_all.sh` therefore writes the settings as `export` lines into a copy of `amdahl_batch.sh` (as the team's `submit.sh` does for `batch.sh`). For a single sweep by hand, edit a copy on the fly:

```bash
bsub < amdahl/amdahl_batch.sh          # one sweep with the defaults: N = 5 000, 30 steps
sed -e 's|gaia_nearest_5000|gaia_nearest_1000|' -e 's|STEPS:-30|STEPS:-300|' amdahl/amdahl_batch.sh | bsub
python3 amdahl/amdahl_analysis.py --job <jobid> ...   # analyse by hand (needs the pandas and matplotlib modules)
```

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

**Time budget** (Xeon E5-2650 v4): one repeat costs about 3 × the serial run (serial, MPI at P = 1, then halving). The campaign's sweeps take 6–15 min each (N = 300 with 10 000 steps about 25 min, hence 2 repeats). With `-W` 00:30 the jobs ran fine on `hpcintro`.

**Choosing the steps:** fixed work per N, with the serial run lasting about 30–90 s. The efficiency of the step loop does not depend on the number of steps (work and per-step overhead both scale with it); only the one-time start-up shrinks relative to longer runs. Our 10 000 steps are used where they fit (N ≤ 300); at N = 5 000 they would take 8 h serially.

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

The programs cannot time their own start-up, so the job script times every launch from outside. **The tables and main plots use the step loop.** Its efficiency does not depend on how many steps we run. The start-up is a fixed cost of about 2 s per run, so its share depends on the run length: in our 30–90 s sweeps it costs a lot, in a 10 000-step run nothing. It is therefore shown on its own (`startup_<label>`), and the whole launch appears only in `measures_<label>_N<n>`.

## Reading the output

- **`speedup_N<n>`**: measured speed-up of the step loop per version (median, min–max bars), the Amdahl fit (dotted) and the ideal line. The label at the end of each line gives the fitted `s`.
- **`efficiency_N<n>`**: E = S / P. At small N it starts above 100 %: the MPI program at P = 1 is up to 10 % faster than the serial program (same force kernel, cause not investigated), and S is relative to the serial program.
- **`karp_flatt_N<n>`**: `e(P) = (1/S − 1/P) / (1 − 1/P)`, the serial fraction that each P implies on its own. **Flat** means a fixed serial part, and Amdahl describes the curve. **Rising with P** means an overhead that grows with P (communication, waiting for the slowest rank, shared memory bandwidth), which Amdahl's model does not contain. The fitted `s` is the least-squares value of these points.
- **`measures_<label>_N<n>`**: speed-up of the step loop, the program and the whole launch side by side. The gap between them is what the parts outside the loop cost.
- **`lost_time_<label>_N<n>`**: where the efficiency of the step loop goes. Each bar is `P·T(P) − T(serial)` in % of the serial run, i.e. `1/E − 1`, split into extra compute, communication, and waiting plus other loop work. Serial parts grow linearly with P here, because all P cores wait while one does them.
- **`startup_<label>`**: start-up per launch (whole launch minus the program's own time) against P, one line per N. It does not depend on N; the dotted line is a straight-line fit.
- **`amdahl_table_<label>_N<n>.tex`**: booktabs table (P, T, S, E, e) with the fit in the caption.
- **`by_N_{speedup,efficiency,karp_flatt}_<label>`**: the step loop of every N in one plot. This is where strong scaling breaks down: the work per step grows like N², the overhead per step (latency, Python, uneven split of the stars) hardly depends on N.
- **`by_N_<label>.tex/.csv`**: one row per N with the serial step time, S and E at the largest P, the Amdahl `s` with its range (the same fit to the fastest and to the slowest runs; if it reaches below 0, `s` is noise), `s · T_step` (the time per step the fit treats as serial; about 0.1 ms for small N) and Karp–Flatt at P = 4 and the largest P. Rising Karp–Flatt = an overhead that grows with P, which Amdahl's model does not contain.
- The terminal report also gives the gain of every doubling of P (`8→16: ×1.6`) and the first P where the efficiency drops below 70 %. That answers the question in our project description: when does doubling the cores stop paying off?

## Results so far: v1-numpy (Allgather), 11 Nov 2026

Open **`results.html`** for the charts. Campaign `v1-numpy_20261111_0949`, jobs 29992752–29992755, one Xeon E5-2650 v4 node per N. Step loop, medians of 3 repeats:

| N | serial ms/step | S(24) | E(24) | Amdahl s (range over runs) | Karp–Flatt P = 4 → 24 |
|---|---|---|---|---|---|
| 50 | 1.43 | 9.1 | 38 % | 5.1 % (4.2–7.8 %) | 3.2 → 7.1 % |
| 100 | 3.55 | 12.2 | 51 % | 3.0 % (1.2–4.5 %) | 1.6 → 4.2 % |
| 1 000 | 142 | 21.9 | 91 % | 0.07 % (−0.08–0.67 %) | −0.5 → 0.4 % |
| 5 000 | 2 871 | 23.3 | 97 % | 0.19 % (0.07–1.3 %) | 0.1 → 0.1 % |

- About 0.1 ms per step does not get faster with more ranks, whatever N is. The work per step grows like N², so s falls like 1/N². For N ≥ 1 000, s is below the run-to-run noise.
- For N ≤ 100, Karp–Flatt rises with P: an overhead that grows with P, so Amdahl's law alone does not describe the curve. The causes are the uneven split of the stars (at N = 50 and P = 24 some ranks hold 3 stars, the average is 2.1, which alone limits E to 69 %), the Allgather latency (communication 0.18 s → 0.71 s over 20 000 steps, mean over the ranks), and the fixed Python work per step.
- Start-up per launch: 1.3–1.4 s for the serial program, 1.80 s + 28 ms per rank with mpirun, the same for every N.
- The first runs (10 Nov, other nodes) gave the same S and E within ±4 points: speed-ups are reproducible, absolute times differ by node.
- Not yet measured: N = 300 (between 51 % and 91 %), more than one node, other parallelisations.

To update `results.html` after a new campaign (e.g. with N = 300), from the repository root:

```bash
python3 amdahl/amdahl_analysis.py --label v1-numpy --job <one job per N> --out amdahl/out/v1-numpy
python3 amdahl/amdahl_analysis.py --label v1-numpy --out amdahl/out/v1-numpy-all   # every job: reproducibility table
python3 amdahl/make_page.py --dir amdahl/out/v1-numpy --all amdahl/out/v1-numpy-all
```

Only the data block of the page changes; numbers in its text are filled in from the data. Sentences that describe a trend (e.g. "for N = 50 it climbs") are fixed text, so read them again after a new campaign.

## Comparing parallelisations

Run the same sweep for every new version with its own `LABEL`, the same `DATA`, `STEPS` and placement. The analysis puts all versions of one N into the same plots. Expect `s` to **grow** when the kernel gets faster (e.g. Numba): the parallel part shrinks, the serial part does not. That is Amdahl's law in one picture.

## Rules that keep the numbers comparable

- **One node per sweep, serial included.** Nodes of the same type differed by about 16 %. Speed-ups are only computed inside one job.
- **Fixed work, not fixed wall time.** With a time limit, larger P simply do more steps. Fixed wall time belongs to Gustafson (weak scaling).
- **Placement stays fixed:** `--map-by core --bind-to core` fills socket 0 first (P ≤ 12 on one socket). Check the `.err` file. A different placement is a different experiment (E3 in the plan).
- No `--plot-iter`, merging off, one thread per rank (`OMP_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`).
- Runs that hit the time limit (`completed = False`) and failed launches are ignored by the analysis.
- With few active cores, the CPU may run at a higher turbo frequency than with all 24 busy. That lowers the efficiency at large P without any serial part. Mention it if e(P) rises at P = 16–24.
