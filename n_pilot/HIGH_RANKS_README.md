# Higher MPI rank-count experiment (non-destructive extension)

These are **new files only**; the group's `ParallelImplementation.py`, `batch.sh`, and your existing `n_pilot/run_pilot.py` are **not modified**.

## Purpose

Select N based on measured simulator throughput and MPI communication share at higher rank counts, with a controlled one-node / two-node comparison. Tests `N=50,100`, **50,000 complete simulation steps**, 3 repetitions, merging disabled via your existing `config.sh`.

- `jobs/high_ranks_one_node.lsf`: 16 allocated cores on **one node**; test `P=2,4,8,16`.
- `jobs/high_ranks_two_nodes.lsf`: 32 allocated cores in groups of 16 per node; test `P=8,16,32` using `mpirun --map-by node` to spread smaller configurations across both nodes.
- `run_high_ranks.py`: thin wrapper calling **the original** `run_pilot.py`; adds unique per-job directories and optional node spreading.
- `analyze_high_ranks.py`: extracts `Simulated`, `forces`, `communication`, `sync` from the original simulator output, summarises medians and ranges, and creates two PNG figures.

### Before submitting (from `n_pilot/`)

Ensure `config.sh` has your **working** `PILOT_SIM_COMMAND` and `PILOT_DATA_TEMPLATE`, which should still refer to `../ParallelImplementation.py` and `gaia_subsets/gaia_random_{N}.csv` (relative to the `gaia/` directory when the simulator is launched by `run_pilot.py`). Confirm merging is disabled (`--merge-radius 0`) and `--no-results` is used.

```bash
source ./config.sh
python3 run_high_ranks.py --n 50 --ranks 16 --steps 10 --repeats 1 --dry-run
python3 run_high_ranks.py --spread --n 50 --ranks 32 --steps 10 --repeats 1 --dry-run
bsub < jobs/high_ranks_one_node.lsf
bsub < jobs/high_ranks_two_nodes.lsf
bjobs
```

Notes: dry runs create a log directory but execute **no simulation**. The first job may pend if no available single node has 16 cores; the second may pend waiting for two nodes with 16 slots each. Before comparing, check the LSF job headers and the `allocation.txt` file saved next to each group of logs: CPU models, host count, rank placement and background load can affect results.

### Analyse when complete

```bash
python3 analyze_high_ranks.py
```

Outputs in `results/`:
- `high_ranks_raw_phases.csv` (one row per simulation)
- `high_ranks_summary.csv` (medians and min/max by N, P and node topology)
- `high_ranks_throughput.png`
- `high_ranks_communication.png`

Original complete logs remain in `logs/high_ranks_one_node_<jobid>/` and `logs/high_ranks_two_nodes_<jobid>/`. Original 10,000-step experiments under `logs/N*_long_R*.log` remain intact. The plots use the NEW 50,000-step experiments only, so their run-length comparisons remain consistent.

The communication percentage is a ratio of the simulator's slowest-rank communication phase timer to simulated runtime; it does not include all MPI synchronization overhead and the individually reported maximum phase times may not add up. Run `python3 analyze_high_ranks.py` again after transferring the logs back to your local `n_pilot/` folder.
