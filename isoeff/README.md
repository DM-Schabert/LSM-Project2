# isoeff: how many ranks, nodes and stars each version handles

Run this once for every parallel improvement. Each run is 10,000 time steps of the MPI program in one 7-minute job. The campaign finds out how many ranks are still worth adding, and how many stars fit in the time budget at each number of ranks. Serial runs and correctness checks are not part of it; do those yourself (`batch.sh MODE=validate`).

```bash
cd ~/.../code                                   # the folder with ParallelImplementation.py
chmod +x isoeff/start.sh                        # once

# the first version
./isoeff/start.sh v1 --merge-radius 0.001 --note "allgather"

# every next version continues where the previous one stopped
./isoeff/start.sh v2-ring --continue-from isoeff/out/v1 --mpi-args "--decomposition ring" --note "ring"

python3 isoeff/controller.py status --campaign isoeff/out/v2-ring --log
python3 isoeff/make_page.py isoeff/out/v1 isoeff/out/v2-ring      # comparison page: isoeff/results.html
```

## The rule (capacity mode, the default)

- **Every run:** `--steps` (10,000) steps in a `--walltime-min` (7) minute job. A run *fits* when it finishes all steps within its own time limit: the walltime minus `--run-reserve-s` (120 s), so 300 s, as in `batch.sh`.
- **N grid:** 300, 424, 600, 849, 1,200, 1,697, 2,400, … (×√2), so N_max is known to within about 20 %.
- **P list:** 1, 2, 4, 8, 12, 16, 24, then whole nodes (48, 72, …) up to `--max-nodes`.

1. At P, grow N along the grid until the run no longer fits. The last N that fits is **N_max(P)**: the most stars this P can simulate in the time budget. If the very first N is already too big, N steps down instead.
2. Step up to the next P and time it on N_max of the previous P. The step pays off when the extra ranks deliver at least 80 % of the ideal extra speed-up: **(T(P′) / T(P)) / (P / P′) ≥ 0.8** (`--marginal`). Only MPI runs are compared.
3. If the step pays off, find N_max of the new P, and so on. If it doesn't, the step is measured once more (another node). If it still fails, P′ is the **last useful P** and the campaign ends.

Each version's result: the last useful P (and its nodes), and N_max at every P up to it. The page and the plots compare that across versions: "v2 goes from 24 to 48 ranks and fits 1.4× more stars".

**Continuing.** `--continue-from <previous campaign>` copies the grid, steps, rule and run settings, and starts `--overlap` (3) grid steps below the previous version's largest N, at the smallest P that handled that N. The first few points of the new version then repeat points of the old one, for a direct comparison, without remeasuring everything from N = 300 and P = 1. Anything you pass explicitly (e.g. `--max-nodes 2`) overrides the copied settings.

**Merges** are fine: every run at one N does the same steps, so all runs at that N follow the same trajectory, merges included.

**Noise.** Each run is its own job, so the two runs of a step can land on different nodes, which differed by up to about 16 %. That's why a failing step is measured twice; the hosts are in `cap_points.csv`. A run just at the 300-second edge can come out either way; the √2 grid keeps that to one grid step.

## How it runs

- **One job per run.** Each job runs the MPI program once and then runs `controller.py next`, which decides the next point and `bsub`s it.
- **Subsets are made automatically** (`gaia_subsets/gaia_nearest_<N>.csv`):
  - `make_subset.py` writes exactly the same file as `GaiaSubsetSelection.py`, but needs no plotly.
  - The controller makes the next subset before it submits the job.
- **Code snapshot.** `init` copies all `*.py` from the code folder into the campaign, so you can keep editing the next version while a campaign runs.
- **Resumable.** The decision is replayed from `results.csv` every time, so `next` never repeats a point that has already been measured. Use it to:
  - resume after a broken chain (it detects jobs killed at the walltime and resubmits),
  - extend a finished campaign (`set ... max_nodes=2`, then `next`).
- **A point that fails twice** (crashes, not "too big") ends the campaign; see `logs/`.

## Commands

| What | Command |
|---|---|
| start | `./isoeff/start.sh <label> [--continue-from isoeff/out/<previous>] [--mpi-args "..."] [--note "..."] [--max-nodes 2]` |
| progress | `python3 isoeff/controller.py status --campaign isoeff/out/<label> [--log]` |
| resume | `python3 isoeff/controller.py next --campaign isoeff/out/<label>` |
| more nodes later | `python3 isoeff/controller.py set --campaign isoeff/out/<label> max_nodes=2`, then `next` |
| results page | `python3 isoeff/make_page.py isoeff/out/v1 isoeff/out/v2-ring`, then open `isoeff/results.html` |
| plots | `python3 isoeff/plot.py isoeff/out/v1 isoeff/out/v2-ring` (in `isoeff/plots/`) |
| test without LSF | `init ... --executor local --mpirun mpirun --steps 300 --walltime-min 1 --run-reserve-s 30` |

`--mpi-args` goes to `ParallelImplementation.py` (e.g. `"--decomposition ring"`, `"--decomposition ring --tile-rows 0"`). Write a single option as `--mpi-args=--tile-rows=0`, so the shell doesn't read it as an option of its own.

## Output (`isoeff/out/<label>/`)

- `capacity.csv`: per P, N_max, the time of its run, the marginal efficiency of the step up to it, and a verdict
- `cap_points.csv`: every (N, P) that was run: runs, fits, time, ms per step, verdict, hosts
- `thresholds.csv`: the step that did not pay off, if any
- `decisions.log`: every decision in words
- `plots/`: largest N against P and the marginal efficiency of each step (PNG + PDF)
- `results.html`: this campaign's own filled results page, written when it finishes
- `results.csv`, `logs/`, `jobs/` (the exact job scripts), `jobs.json`, `DONE` (why it stopped)

## Results page

`isoeff/results.html` is in the same style as `amdahl/` and `gustafson/results.html`. It stays empty until `make_page.py` fills its data block. Pass one or more campaign folders; their order sets the colours and the "gained" columns (each version against the one before it).

- **Tiles:** useful ranks per version, the most stars there, and what it gained over the previous version.
- **Charts:** the largest N against P for all versions (rings mark steps that didn't pay off), and the marginal efficiency of every step against the 80 % rule.
- **Version table:** MPI options, useful P, nodes, largest N, ranks and stars gained, which version it continues, why it stopped.
- **One version in detail:** every run and the decision log.

Older (time-mode) campaigns can sit on the same page; their sections appear only when they're included. The charts load d3 from cdnjs, so the page needs an internet connection.

## Older modes

Campaigns made before capacity mode still replay and still show on the page. You can also start new ones with `--mode time` or `--steps0 S`:

- **Time mode:** speed-up against a serial run in the same job, P steps up while marginal ≥ 0.8 and E ≥ 0.5, and N doubles at a threshold.
- **Step mode (`--steps0`):** like time mode, but with a fixed number of steps per N.

Both need a serial run that fits in the job, which limits N.
