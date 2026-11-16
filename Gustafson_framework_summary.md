# Gustafson (weak-scaling) framework — summary (12 Nov 2026)

New folder `gustafson/` next to `amdahl/`, same structure: `run_all.sh` (experiment list + analysis job), `gustafson_batch.sh` (one sweep per node), `plan.py` (N(P), time estimate, cost-model fit; stdlib only), `gustafson_analysis.py`, `make_page.py`, `results.html` (same style as `amdahl/results.html`, empty until filled), `README.md`. Uses the unchanged Serial/ParallelImplementation.py; run_name = `gustafson_<label>_<rule><N0>_j<job>_r<rep>`.

## Design decisions
- Per P, one job runs MPI on N(P) stars with P ranks (T(P), fixed steps) and the serial program on the same N(P) (τ(P), max(10, ceil(steps/P)) steps). Only per-step times are compared.
- Metrics: E = T(1)/T(P) (weak efficiency, ideal 1); S = τ(P)/T(P) (scaled speed-up, ideal P); g = τ(P)/τ(1); s(P) = (P−S)/(P−1); Gustafson fit P − S = s(P − 1).
- Serial cost per step fitted to the v1-numpy Amdahl campaign: T(N) = a·N + b·N², a = 23.47 µs, b = 113.5 ns (±3.4 %).
- Main rule `time`: N(P) solves T(N) = P·T(N0), so each rank keeps the base problem's serial time (Gustafson's assumption). Comparison rule `pairs`: N = N0·√P (as in code/plot_results.py). With pairs, NumPy's per-row overhead makes each rank's share cheaper than the base problem; the model predicts ~190 % weak efficiency at N0 = 100, P = 24.

## Experiment list (P = 1, 2, 4, 8, 12, 16, 24; 3 repeats; ~16 min each, 30 min walltime)
| rule | N0 | steps | N(24) |
|---|---|---|---|
| time | 50 | 14 000 | 461 |
| time | 100 | 6 000 | 761 |
| time | 300 | 1 200 | 1 810 |
| time | 1 000 | 150 | 5 279 |
| pairs | 100 | 6 000 | 490 |
| pairs | 1 000 | 150 | 4 899 |

Needs ~40 new nearest-N subsets (run_all.sh prints the GaiaSubsetSelection.py command). After the campaign: `python3 gustafson/make_page.py --dir gustafson/out/<campaign> --amdahl amdahl/out/v1-numpy`.

Status: pipeline tested locally with MPICH on synthetic subsets and under the cluster's pandas 2.3.0 / numpy 2.3.1 / matplotlib 3.10.3; not yet run on the HPC.
