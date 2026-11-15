# isoeff scaling framework — summary (14–15 Nov 2026)

Folder `code/isoeff/`, next to `amdahl/` and `gustafson/`: `controller.py` (stdlib only), `make_subset.py`, `plot.py`, `make_page.py`, `results.html`, `start.sh`, `README.md`. Run once per parallel improvement.

## What we measure (agreed 15 Nov): capacity mode, the default
- **Fixed per run:** 10 000 steps and a 7-minute job (the run must finish within 300 s, as in `batch.sh`). MPI runs only.
- **Serial and correctness are separate:** serial was the proof of concept; correctness is checked by us now and then with `batch.sh MODE=validate`.
- **Per version, the question is:** how many more ranks/nodes are still worth adding, and how many more stars fit.
- **Rule:**
  1. At P, grow N along a ×√2 grid (300, 424, 600, 849, 1 200, 1 697, 2 400, 3 394, …) until 10 000 steps no longer finish in time. The last N that fits is N_max(P).
  2. Step up to the next P (1, 2, 4, 8, 12, 16, 24, then whole nodes) and time it on N_max(P_prev). It pays off when (T(P_prev)/T(P))/(P/P_prev) ≥ 0.8.
  3. If it pays off, find N_max of the new P. If not (measured twice; runs can land on different nodes, ±16 %), P_prev is the last useful P and the campaign ends.
- **Continuing:** `--continue-from <previous campaign>` copies the grid, steps, rule and run settings, and starts 3 grid steps (`--overlap`) below the previous version's largest N, at the smallest P that held that N there. The new version repeats the last points for a direct comparison instead of starting at N = 300, P = 1 again.
- **Merges** are fine: all runs at one N do the same steps, so they follow the same trajectory.
- **Usage:**
  ```
  ./isoeff/start.sh v1 --merge-radius 0.001
  ./isoeff/start.sh v2-ring --continue-from isoeff/out/v1 --mpi-args "--decomposition ring"
  python3 isoeff/make_page.py isoeff/out/v1 isoeff/out/v2-ring
  ```
- **Output:** `capacity.csv` (P, N_max, time, marginal efficiency of the step into P, verdict), `cap_points.csv` (every run), `thresholds.csv`, `decisions.log`, plots (largest N vs P; marginal efficiency per step), and the campaign's own `results.html`.
- **Results page:** capacity sections (tiles with useful ranks and the gain over the previous version; largest N vs P for all versions; marginal efficiency per step; version table with ranks and stars gained; every run of a chosen version). The older time-mode sections only appear for campaigns of that kind.

## Shared machinery
- **Self-chaining LSF jobs.** Full nodes reserved (`-n 24 span[hosts=1]`; above 24, `span[ptile=24]`). Each job ends with `controller.py next`, which `bsub`s the next point.
- **Stateless replay.** The rule is replayed from `results.csv` every time: resumable (killed jobs detected via `bjobs`) and extendable (`set max_nodes=2` + `next`) without remeasuring.
- **Subsets** are made on demand by `make_subset.py`: byte-identical to `GaiaSubsetSelection.py`, without plotly.
- **Code snapshot** per campaign.
- **`mpi_args`:** options for `ParallelImplementation.py` only (e.g. `--decomposition ring`, `--tile-rows 0`).

## Older modes (14–15 Nov, superseded but still replayed)
- **Time mode:** speed-up against a serial run in the same 7-min job; P steps up while marginal ≥ 0.8 and E ≥ 0.5; N doubles at a threshold (isoefficiency path).
- **Step mode (`--steps0`):** the same, with fixed steps per N.
- **Why superseded:** both need a serial run that fits in the job. With 10 000 steps in 7 min that holds only up to N ≈ 420, which doesn't fit the agreed fixed steps + fixed walltime.

## First HPC campaigns (15 Nov, time mode, N0 = 500, 1 node)
- **v1-numpy:** still cost-effective at P = 24 with N = 500 (E 76 %, marginal 16 → 24 81 %, S = 18.2).
- **"v2-ring":** did **not** run the ring. It was created by the old controller without `mpi_args`, so its numbers match v1. Check `results.csv` for `ring_overlap`.

## Status
Capacity mode was tested:
- **On simulated data through the real rule:** three versions on 2 nodes; a continuation; a slower version that has to step N down; a failing 24 → 48 step.
- **Locally with the real programs and MPICH:** a campaign and a continuation of it with the ring; the results page with capacity-only, mixed and empty data.

Not yet run on the HPC in capacity mode.
