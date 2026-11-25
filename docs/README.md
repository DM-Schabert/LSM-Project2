# Where to find what

The repository holds the two programs, one folder per experiment, and the presentation. This page lists them in the order of the report: each question, the folder that answers it, the job that measures it and the files that hold the result.

## Start here

| You want | Open |
|---|---|
| every result, as slides | `presentation.html` (its numbers are refreshed by `make_presentation.py`) |
| every results page, with one line per finding | `overview.html` |
| the programs | `SerialImplementation.py` and `ParallelImplementation.py` (`--decomposition allgather`, `ring` or `2d`; `--tile-rows K`) |
| the stars | `gaia_cleaned.csv`; the subsets `gaia_subsets/gaia_nearest_<N>.csv` are made by `GaiaSubsetSelection.py` |
| the row of every run | `results.csv` |

## The main line of the report

Each folder has a `README.md` with the question, the expectations written before the measurement, the commands and the result.

| | Question | Folder | Jobs | Results in git | Page and plots |
|---|---|---|---|---|---|
| 1 | Do more ranks give more time steps? Strong scaling and the Amdahl fit | `amdahl/` | `run_all.sh` starts one `amdahl_batch.sh` per N | rows `amdahl_*` in `results.csv`, `launch_times.csv` | `amdahl/results.html`; plots in `amdahl/out/<campaign>/` |
| 2 | Are the three schemes correct, and is the network the limit? | `communication/` | `validate.sh` (correctness), `bench_1node.sh` and `bench_2nodes.sh` (messages alone); the scheme comparison runs through `amdahl/run_all.sh` | `communication/out/` | `communication/results.html`, `2d_decomposition.html`; plots in `communication/out/plots/` |
| 3 | Does overlap help the ring? | `communication/overlap/` | `overlap_batch.sh` | `communication/overlap/out/` | `communication/overlap/results.html` |
| 4 | Is the memory size the limit? | `memory/` | `memory_batch.sh` | `memory/out/` | `memory/results.html` |
| 5 | Why do the same pairs cost more on a full node? The cache (K0 to K5) | `cache/` | `cache_node.sh`, `cache_node20k.sh` (K0 to K2), `cache_bign.sh` (K3), `cache_comm.sh` (K4), `counters.sh` (K5) | `cache/out/`: the tables `k0` to `k5`, the reports, the raw rows of every job | plots in `cache/out/plots/` |
| 6 | What did each change buy, in time steps per 5 minutes? | `bench5/` | `bench5_measure.sh`, then `bench5_final.sh` | `bench5/out/`: `steps5.csv`, `winners.txt`, `report.txt` | plots in `bench5/out/plots/` |

## Side results

| Question | Folder | By |
|---|---|---|
| How many stars fit when ranks are added? | `isoeff/`, plots in `plots/` | Pelle |
| Weak scaling (Gustafson), same structure as `amdahl/` | `gustafson/` | Pelle |
| The ring scheme when it was added | `ring_v2/`, `Ring_v2_summary.md` | Pelle |
| Does redistributing the stars after merges pay off? | `experiments/` | Rasmus |
| Does a derived MPI datatype for (x, y, z) help? | `data_types/` | Rasmus |
| Runs with many ranks on one and two nodes | `n_pilot/` | Rasmus |

## How an experiment folder is laid out

The folders of the main line share one layout:

```
<experiment>/
  README.md      the question, the expectations, the commands, the result
  *.sh           LSF jobs; start them from the repository root:  bsub < <experiment>/<job>.sh
  *.py           the benchmark programs, the analysis, make_page.py
  out/           results: CSVs and reports are in git; out/plots/ is made again by the analysis
  logs/          LSF output, not in git (memory/ and overlap/ keep the logs of the one job they report)
  results.html   the results page, where there is one
```

- Files are named after their job: `<what>_<LSF job id>.csv`.
- Most jobs also run on a laptop with tiny sizes, as `LOCAL=1 bash <experiment>/<job>.sh`. This only checks that the scripts run and is never a measurement. Its files carry `local` in their name and git ignores them.
- `amdahl/out/` is not in git at all: `amdahl_analysis.py` makes it again from `results.csv` and `launch_times.csv`.

## This folder

| File | What it is |
|---|---|
| `LSM_Assignmnt_2.pdf` | our proposal and the report template |
| `Gaia N-body two HPC project descriptions.md` | the project description |
| `Parallel_Plan_v1.md` | the plan for the parallel program, written before v1 (9 and 10 Nov) |
| `ParallelImplementation_and_batch_summary.md` | how v1 and its job script work, function by function (10 Nov) |
| `LSM_Project2_session_summary_2026-11-09_10.md` | notes of the work on 9 and 10 Nov |
| `mpi_parallelization_map.html` | the serial program with every place the parallel version changes |

The three notes from November use the folder names of that time (`code/`, `claude/`).
