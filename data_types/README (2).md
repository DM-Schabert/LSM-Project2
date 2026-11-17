# MPI datatype experiments (isolated folder)

**Purpose.** Test whether a derived MPI datatype for `(x, y, z)` coordinates
improves the communication step of our N-body project.

**Safety.** Nothing in this folder imports or changes `ParallelImplementation.py`
or any other existing project file. Only `data_types/results/` and
`data_types/logs/` receive generated outputs. Synthetic `(N, 3)` `float64`
coordinates reproduce the relevant Allgatherv memory layout without requiring
`gaia_cleaned.csv`. This is **only a communication microbenchmark**, not yet
an end-to-end speedup measurement for the main simulation.

## Files

- `benchmark_allgatherv.py`: MPI microbenchmark + exact correctness check.
- `layout.py`, `test_layout.py`: partitioning helper and tests that need no MPI.
- `hpc_env.sh`: dedicated environment configuration (if needed).
- `jobs/smoke.lsf`: quick 1/2/4-rank correctness checks, including uneven blocks.
- `jobs/benchmark.lsf`: 2/4/8-rank timing sweep for N = 1k/5k/20k.
- `analyze_results.py`: summarizes the raw CSVs and optionally makes a PNG plot.
- `results/`: machine-readable output CSVs and plots (safe to commit).
- `logs/`: LSF `.out`/`.err` files (ignored by Git).

## 1. First, in VS Code on your computer

1. Open a terminal and clone the existing repo once:

   ```bash
   git clone https://github.com/DM-Schabert/LSM-Project2.git
   cd LSM-Project2
   git switch -c experiment/data-types
   code .
   ```

   If the repository is already cloned, just open that folder in VS Code,
   run `git status`, `git pull --ff-only` (only if your working tree is clean),
   then `git switch -c experiment/data-types`.

2. Extract the supplied ZIP **into the repository root**. It should create
   exactly `LSM-Project2/data_types/`. If you already have a folder named
   `data_types`, stop rather than overwriting it.

3. Sanity-check the new folder (Python standard library only):

   ```bash
   python data_types/test_layout.py
   git status --short
   ```

## 2. On DTU HPC (LSF-based DTU/DCC cluster)

Work on the HPC's login node to submit jobs, not to run benchmarks directly.
Make the repository available there by cloning your new branch (after pushing
it to GitHub) or transferring `data_types/` using your usual HPC transfer
process. Then go **inside `data_types/`**:

```bash
cd /path/to/LSM-Project2/data_types
# Check whether MPI, NumPy and mpi4py are available in this environment:
source ./hpc_env.sh
# Submit ONLY the small correctness test first:
bsub < jobs/smoke.lsf
bjobs
```

`source ./hpc_env.sh` may report missing MPI/Python: add the exact `module load`
and environment activation commands used by the existing project to **this new
`hpc_env.sh` file only**. This is the one configuration item that might need
adjustment: we could not inspect the current repository's HPC module setup
from this session. Make sure Python's `mpi4py` matches the loaded MPI library.

When the smoke job has **finished successfully** (`bjobs -a`), inspect its log:

```bash
ls -lt logs/
cat logs/smoke_<JOB_ID>.out
```

Look for **`SMOKE TEST PASSED`** and `PASS` on all cases. Once verified:

```bash
bsub < jobs/benchmark.lsf
bjobs
```

The second job creates `results/bench_<JOB_ID>.csv` and `logs/bench_<JOB_ID>.out`.
Results are appended per benchmark combination by rank 0 only; separate job
IDs mean jobs do not overwrite each other's results.

**Note:** These are DTU/DCC **LSF** scripts (`bsub`, queue `hpc`), not Sophia
Slurm scripts. Use the scheduler your project is actually running on. All
experiments are designed for a single node to keep hardware comparable.

## 3. Analyse outputs

With the CSV results in `data_types/results/`, run:

```bash
python data_types/analyze_results.py
```

from the repo root, or `python analyze_results.py` from inside `data_types/`.
Generates a `summary.csv` and, when matplotlib is available, a comparison PNG.

## 4. Only commit YOUR new folder

From the repository root:

```bash
# Confirm your branch, and inspect what changed.
git branch --show-current
git status --short
# There must be NO previously staged files before you continue.
git diff --cached --name-only
# Stage ONLY the new folder.
git add data_types/
# Verify the staged paths. Only data_types/... should appear!
git diff --cached --name-only
# Commit and push only this branch.
git commit -m "Add isolated MPI datatype experiments"
git push -u origin experiment/data-types
```

Create a Pull Request from `experiment/data-types` into `main` when ready.
Do not click Merge until you've reviewed the changed-files tab. No existing
repo file needs to be edited or staged. `git add .` is deliberately avoided.

### What exactly are we comparing?

**Standard**: `MPI.DOUBLE`; counts and displacements measure individual
`float64` elements. Every star has three of them.

**XYZ derived datatype**: `MPI.DOUBLE.Create_contiguous(3)`; one MPI element
means three `float64` values, i.e. an `(x, y, z)` record. Counts and
displacements now measure stars.

Both methods transfer identical bytes. Derived datatypes are not compression.
Here the data are already contiguous, so major improvements are **not**
expected; finding no benefit is a valid experiment result.

### Benchmark interpretation

- Values represent max across ranks of average time per blocking `Allgatherv`
  call over repeated iterations; barrier and reduction overheads are excluded.
- Each approach receives its own separate output buffer and warmup calls.
- The benchmark alternates which approach goes first in successive repeats.
- Performance of the actual simulation cannot be inferred solely from this
  microbenchmark: only integrate after checking total runtime separately.
- To compare fairly, run both variants on the same nodes under similar load.
