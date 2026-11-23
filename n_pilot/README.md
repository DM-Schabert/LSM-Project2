# N-selection pilot (isolated; existing simulation unchanged)

## Question
For which **fixed N** does our existing MPI N-body simulator become communication-sensitive at 2, 4 and 8 ranks? This first pilot measures end-to-end runtimes and preserves simulation logs. It does **not** implement the proposed fixed-time throughput experiment.

## Safe layout
Place the entire `n_pilot/` folder beside `ParallelImplementation.py`, `batch.sh`, and `data_types/`. No original project files are modified or copied.

## Required one-time configuration
The GitHub code was not accessible when this folder was packaged, so the exact simulator CLI **has not been verified**. Open `../batch.sh` and look at the definition of `ARGS` and its invocation of `ParallelImplementation.py`. Then set `PILOT_SIM_COMMAND` in `config.sh` to that **same working invocation**, replacing ONLY the rank count by `{P}`, number of stars (or data path) by `{N}` (or `{DATA}`), and number of simulation timesteps by `{STEPS}`. Keep all physics, integration, output and merge flags equivalent to the existing run. Example in the config is *not a verified command*.

To inspect the existing command (read-only):
```bash
grep -nE '^(N|DATA|ARGS|STEPS|MPIRUN)=|ParallelImplementation.py|add_argument' ../batch.sh ../ParallelImplementation.py | head -80
```
If `DATA` is a subset file, set `PILOT_DATA_TEMPLATE` to its actual path template. Use `GaiaSubsetSelection.py`/the existing subset workflow to prepare *one fixed, deterministic file for each N* and reuse it for every rank count.

## Run on DTU HPC
```bash
cd ~/courses/02616_LSM/gaia/n_pilot
source ./config.sh
python3 run_pilot.py --dry-run
# Review ALL commands, then:
bsub < jobs/pilot.lsf
bjobs
```
If the existing simulator does not have a safe flag for a fixed step count or to disable merging, do **not** guess it; inspect its arguments first.

Optional: submit a separate MPI-only communication pilot using the *existing* verified datatype benchmark, for the same small N values:
```bash
bsub < jobs/communication.lsf
```
The communication run has its own job ID and writes `results/mpi_only_<jobid>.csv`. It measures communication in isolation, **not** the simulation's communication share.

After completion:
```bash
python3 analyze_pilot.py
ls -lt logs/ results/
```

## Interpretation
- Candidate N: 250, 500, 1000, 2000 (adjust after initial tests).
- Fixed 100 **simulation timesteps**, not 100 `Allgatherv` calls. Runs test the unchanged real simulator.
- 2, 4, 8 ranks, one host. Repeats 2 times. Increase repeats for final evidence.
- The Python driver measures **end-to-end walltime**, including MPI process startup, reading input and output! `steps / walltime` is only a *nominal* throughput estimate. For small N, process startup can distort results. Inspect original simulator's internal timing logs if available and use them in preference to this walltime.
- Run with star merging disabled (or verified constant N) so every configuration has a comparable workload.
- Choose a useful N based on measured MPI share in the **actual application**, not only the total runtime or isolated communications. Aim for a workload where communication/synchronization is visible but Python startup and I/O are not dominant. Consider a larger N as a control.
- `analyze_pilot.py --mpi results/mpi_only_<jobid>.csv` can include a clearly labeled **communication-only screening proxy** for matching N,P; it is NOT the real application's MPI percentage and shouldn't be used as a final answer.

## Safety
- All output stays within `n_pilot/`.
- Only stage `git add n_pilot/` when ready; inspect `git diff --cached --name-only` before committing.
- No changes to `ParallelImplementation.py`, `batch.sh`, or `amdahl/` are needed for this initial probe.
