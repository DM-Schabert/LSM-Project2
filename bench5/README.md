# bench5: how many time steps fit into 5 minutes, for every configuration we built

**Goal.** Our assignment says: *"Our overall goal is to simulate with as many of the stars as possible … within reasonable computational time"*, and the program's normal run is limited to 5 minutes (`NORMAL_TIME_LIMIT_S = 300` in `SerialImplementation.py`). This benchmark plays through every configuration the project went through, from the serial program to the cache-sized tiles, and measures for several groups of stars how many time steps each one gets into 5 minutes.

Nothing in the program changes: every configuration is a set of options of `ParallelImplementation.py` (`--decomposition`, `--tile-rows`) and of `mpirun` (one node or two), or the serial program itself.

```bash
# on the cluster, from the repository root
# phase 1: every configuration, one job per group of stars (10-25 min each)
for n in 1000 5000 20000 100000; do sed "s/^N=.*/N=$n/" bench5/bench5_measure.sh | bsub; done

# on the laptop, after copying bench5/out and bench5/logs back
python3 bench5/analyze.py          # table, plot, report; writes bench5/out/winners.txt

# phase 2, on the cluster with winners.txt in bench5/out: the winner of every group runs 5 real minutes (~25 min)
bsub < bench5/bench5_final.sh
python3 bench5/analyze.py          # now with the real 5-minute runs

# check that the scripts run (laptop, tiny sizes, never a measurement)
LOCAL=1 bash bench5/bench5_measure.sh && python3 bench5/analyze.py --local && LOCAL=1 bash bench5/bench5_final.sh
```

## Configurations (`configs.txt`, in the order the project introduced them)

| Label | Ranks | What | Since |
|---|---|---|---|
| `serial` | 1 | the serial program | 8 Nov |
| `v1-24` | 24 | v1 (Allgatherv), untiled kernel (the serial row loop) | 9 Nov |
| `v1-tiles-24` | 24 | v1 with automatic tiles (~2^18 pairs, 20 MB) | 15 Nov |
| `ring-24`, `2d-24` | 24 | the ring (overlap) and the 2D grid, automatic tiles | 15/16 Nov |
| `v1-tiles-48`, `ring-48`, `2d-48` | 48 | the same on two nodes | 17 Nov |
| `v1-small-24/48` | 24, 48 | v1 with 2-row tiles (`cache/`) | 24 Nov |
| `ring-small-24/48` | 24, 48 | the ring with pieces of ~13 000 pairs (~1 MB) | 24 Nov |
| `2d-small-24/48` | 24, 48 | 2D with tiles of ~13 000 pairs | 24 Nov |

Tile rows for the "small" rules are computed per N and P in `common.sh` (at least 2 rows: 1 row would switch the program to its slow serial row loop). One node means 24 ranks on the first node of the two-node job (`--map-by core`), two nodes 24 + 24 (`--map-by ppr:24:node`).

**Groups of stars:** N = 1 000, 5 000, 20 000 and 100 000 (`gaia_subsets/gaia_nearest_<N>.csv`, made by the job if missing).

## Method

- **Phase 1 (`bench5_measure.sh`):** every configuration runs for about 15–20 s (at least 3 steps; at N = 100 000 at least 2), 2 repeats (1 at N = 100 000). From the time per step of the step loop, `analyze.py` computes
  - **steps in 5 minutes of computing:** 300 s / time per step, the number that ranks the configurations, and
  - **steps in a 5-minute job:** (0.95 · 300 s − start-up) / time per step, because the program counts its start-up into the time limit and stops 5 % early.

  The serial program is left out at N = 100 000 (one step would take about 17 minutes).
- **Phase 2 (`bench5_final.sh`):** the best configuration of every group runs with `--time-limit 300` and as many steps as it can. Its iterations are the time steps a real 5-minute job gets: a measurement, not an extrapolation.

## Expectations (written before the first measurement)

`analyze.py` applies these rules and prints yes / no / n.a.

| # | Expectation (from `communication/` and `cache/`) | Rule for "yes" |
|---|---|---|
| B1 | From N = 5 000 on, the best configuration uses two nodes (×1.98 from 24 to 48 ranks in C3). | The winner of every N ≥ 5 000 has 48 ranks |
| B2 | Small tiles beat the automatic ones on a full node (K1: 23 % shorter steps at N = 5 000). | `v1-small-24` > `v1-tiles-24` at every N ≤ 20 000 |
| B3 | At N = 1 000 the ring's P kernel calls per step make it lose to v1 (C3: ×1.2–2.2 at 24–48 ranks). | Best v1 > best ring at N = 1 000 |
| B4 | At N = 100 000 the ring with cache-sized pieces wins (K3: 6–8 % shorter steps). | The winner at N = 100 000 is `ring-small-24` or `ring-small-48` |
| B5 | Phase 1 predicts a real 5-minute job. | Iterations of every phase-2 run within 90–105 % of the expectation from phase 1 for its time limit |

## Files

| File | What it does |
|---|---|
| `configs.txt` | the configurations: label, ranks, placement, scheme, tile rule, date, description |
| `common.sh` | sourced by the jobs: modules, placement, tile rules, one run of a configuration, copying the job's rows |
| `bench5_measure.sh` | phase 1, one group of stars per job (set `N=` with `sed`, as above) |
| `bench5_final.sh` | phase 2, the winners from `out/winners.txt` for 5 minutes each (`ONLY="5000 20000"` for a subset) |
| `analyze.py` | `out/steps5.csv`, `out/winners.txt`, `out/report.txt`, `out/plots/steps5.png` |

Every run also writes its usual row to `results.csv`, named `bench5_<config>_N<N>_r<repeat>_j<job>` (phase 1) or `bench5final_<config>_N<N>_j<job>` (phase 2); each job copies its rows to `out/runs_<job>.csv`. Files of `LOCAL=1` runs carry `local` in their name and are ignored by git.
