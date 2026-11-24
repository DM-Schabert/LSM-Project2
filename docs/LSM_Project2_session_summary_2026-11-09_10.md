# LSM Project 2: session summary (9–10 November 2026)

Follows on from `claude/LSM_Project2_session_summary.md` (8 November).

## What we did

- Read the course material for weeks 1–9 and the Project 2 texts, and wrote `claude/Parallel_Plan_v1.md`.
  - The plan covers a first MPI version (v1) that uses only weeks 1–6.
  - It also lays out how weeks 7, 8 and 9 can be added later as separate iterations.
- Added the project scripts to the project under `code/`: `SerialImplementation.py`, `GaiaSubsetSelection.py`, `DataVisualize.py`, `plot_space.py` and `plot_space_2.py`.
- Checked the plan against the real serial code and updated it. The main finding is that the serial merge loop (`while close_pairs:`) must become a collective decision in the MPI version.
- Wrote `ParallelImplementation.py`, the MPI version (v1), and `batch.sh`, the LSF job script with a benchmark mode and a validate mode.
- Made two small changes to `SerialImplementation.py`:
  - `compute_accelerations` takes an optional star range, so serial and MPI run the same force code.
  - `load_subset` now returns positions in row-major (C) order.
- Found and fixed a bug: pandas returned positions in column-major (Fortran) order. `Bcast` sends raw memory, so ranks ≥ 1 received scrambled x, y, z values (errors of about 2 pc after one step).
- Tested the MPI version locally:
  - It is **bit-identical** to the serial version on 1, 2, 3, 4 and 7 ranks, with and without merging (290 of 300 stars merged in one test).
  - Fewer stars than ranks works.
  - The time limit, SIGTERM, a missing data file, `--plot` and `--plot-iter` all work.
- Found that a SIGTERM sent to `mpirun` (as `bkill` or the walltime limit may do) kills the ranks before they can save. `batch.sh` therefore always stops the program itself, 2 minutes before the walltime.
- Explained how to run `GaiaSubsetSelection.py` on the HPC to make the subset CSVs.
- Ran the first jobs on the HPC and analysed them:
  - Serial vs. 8 ranks at N = 5 000 on two nodes.
  - 8 ranks at N = 10 000 and 15 000.
- Made a LaTeX table of the first speed-up result.
- Changed result saving so that every run (serial and MPI, any job) goes into **one** `results.csv`, protected by a file lock. Tested with 60 processes writing at the same moment: no rows were lost.
- Wrote `plot_results.py`, which makes the report plots from `results.csv`:
  - step time vs. N
  - serial vs. MPI per node
  - speed-up vs. ranks, once the scaling runs exist

## Choices we made

- **Project:** the Gaia N-body project is approved as our Project 2, instead of the Poisson problem.
- **v1 scope:** only material from weeks 1–6. There are three small steps beyond the exact lecture list: `Allgatherv`, one `Barrier` before the step loop, and passing `comm` as a function argument.
- **Decomposition:** each rank owns a contiguous block of stars (sizes differ by at most one). All positions are replicated with `Allgatherv` before every force calculation.
- **`Allgatherv`, not padding:** this works for any N without dummy stars, and keeps the results bit-identical to the serial code.
- **Shared kernel:** the MPI version imports all physics, plotting and result code from `SerialImplementation.py`. Speed differences can then only come from the parallelisation.
- **Merging:**
  - Ranks agree whether there are close pairs anywhere with an `Allreduce`.
  - The pair lists and velocities are then gathered to every rank, and every rank runs the serial `merge_stars` on identical data.
  - No deduplication or sorting is needed: each pair is found once, and the union-find result doesn't depend on pair order.
- **Stopping:**
  - The time limit, Ctrl+C/SIGTERM and closing the plot window are only local requests. Ranks stop together through an `Allreduce(MAX)`.
  - Only rank 0's clock decides the time limit.
  - A second Ctrl+C no longer stops one rank on its own.
- **Crashes:** an unhandled error on any rank calls `Abort`, so the other ranks don't wait until the walltime runs out.
- **Timing:**
  - Every section is timed on every rank and reduced to rank 0.
  - `results.csv` shows the MAX over ranks (the slowest rank decides the step time), plus min and mean columns.
  - Interactions are counted per rank (own rows only) and summed.
- **Communication API:** upper-case `Allgatherv`/`Allreduce`/`Bcast`/`Reduce` on NumPy arrays in the hot path. Lower-case pickled calls are used only for small one-off objects (pair lists, class letters, error messages, host names).
- **Results file:**
  - One shared `results.csv` with a file lock (`results.csv.lock`).
  - Serial rows leave the MPI-only columns empty.
  - New columns are merged in automatically.
- **Benchmarks vs. validation:**
  - Validate (serial + MPI + comparison) after every change to the parallel code, using a small N.
  - Benchmark (MPI only) for timing runs.
  - The serial baseline is run once per N, step count and CPU type.
- **Speed-up:** only computed between a serial and an MPI run on the **same node**, because nodes with the same CPU model differed by about 16 %.
- **Plot style:** no titles (the file name is the title); serial is always orange and MPI always blue; PNG for viewing and PDF for LaTeX.

## First results (Xeon E5-2650 v4 @ 2.20 GHz, `hpcintro`)

| | N = 5 000, node -75 | N = 5 000, node -77 | N = 10 000 | N = 15 000 |
|---|---|---|---|---|
| Serial time per step | 2 499 ms | 2 913 ms | — | — |
| 8 ranks, time per step | 322 ms | 375 ms | 1 419 ms | 3 129 ms |
| Speed-up / efficiency | 7.76× / 97 % | 7.78× / 97 % | — | — |

- **Validation:** MPI and serial final states are identical on real Gaia data. `final_position_checksum` is 6483251.584928703 in both.
- **Communication:** 0.04 % of the run time.
- **Sync time:** this is mostly faster ranks waiting for the slowest one; the gap between the fastest and slowest rank's `forces` time is about 3 %.
- **Step time grows as N²:** 5k → 10k is 4.1× slower (theory 4×); 10k → 15k is 2.2× slower (theory 2.25×).
- **The kernel is slow:** it reaches about 0.2 GFLOP/s per core, roughly 0.5 % of the 35.2 GFLOP/s peak. That is why communication is negligible and the speed-up is close to ideal.
- **Time budget:** the full 626k set would take about 12 h per step serially, or about 30 min per step on a full 24-core node with perfect scaling.

## Files

- `claude/Parallel_Plan_v1.md`: the plan, with a status update for 10 Nov.
- `claude/ParallelImplementation_and_batch_summary.md`: how the MPI script and the job script work, function by function.
- `code/ParallelImplementation.py`, `code/batch.sh`, `code/plot_results.py`: new.
- `code/SerialImplementation.py`: updated (kernel range, C-order positions, locked results file).

## Next steps

- Run E2: N = 5 000, 100 steps, P = 1, 2, 4, 8, 12, 16, 24, on a fully reserved node (`-n 24`, then set `NP` in `batch.sh`). Do at least 3 repeats per point.
- Check the `--report-bindings` output in the `.err` files to see how the ranks were placed on the sockets.
- Run E3 (compact vs. spread over sockets), then E4/E5 (two nodes, InfiniBand vs. Ethernet).
- Validate merging on real Gaia data with a merge radius that actually produces merges.
- Add an energy-conservation check (M3).
- Prepare the 18 Nov plenum slide.
- Start the week 7 iteration: either the ring algorithm on a 1-D Cartesian communicator, or the 2-D force decomposition. Results will then no longer be bit-identical, so validate against a tolerance instead.
- Fix the small issues in Appendix C of the plan before the hand-in ZIP.
