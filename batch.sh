#!/bin/bash
# =============================================================================
#  LSF job script for the MPI N-body simulation (ParallelImplementation.py)
#
#  Submit:   bsub < batch.sh
#  Watch:    bstat   /   bjobs -w   /   bjobs -p <jobid>   (why is it pending?)
#  Cancel:   bkill <jobid>
#
#  To change the number of ranks or the placement, edit the #BSUB lines
#  below. LSF reads them before the script runs, so shell variables don't
#  work in them. Options given on the command line override the ones in
#  this file, e.g.   bsub -n 16 < batch.sh
#
#  Placement recipes (week 2):
#    one node, P ranks          -n P      span[hosts=1]
#    two nodes, P/2 per node    -n P      span[ptile=P/2]
#    spread over sockets        reserve the full node (-n 24, span[hosts=1]),
#                               then set NP and MAP_OPTS below, e.g.
#                               NP=8  MAP_OPTS="--map-by ppr:4:package --bind-to core"
# =============================================================================
#BSUB -J nbody_mpi
#BSUB -q hpcintro
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 00:15
#BSUB -o nbody_mpi_%J.out
#BSUB -e nbody_mpi_%J.err

# ----------------------------------------------------------------- settings
MODE=benchmark          # benchmark: one MPI run
                        # validate:  serial + MPI on the same input, then compare final states

DATA=gaia_subsets/gaia_nearest_5000.csv
STEPS=100               # pick from the serial baseline: steps ≈ 200 s · (interactions/s) / N²
DT=0.01
SOFTENING=0.01
MERGE_RADIUS=0          # 0 = merging off (use for benchmarks); the program's default is 0.01

WALLTIME_MIN=15         # keep equal to #BSUB -W above
# The program stops itself and saves its results before LSF kills the job.
# This is the stop that is guaranteed to work: in testing, a SIGTERM sent to
# mpirun (as bkill or the walltime limit may do) killed the ranks before
# they could save.
TIME_LIMIT=$(( WALLTIME_MIN * 60 - 120 ))

NP=""                   # empty: one rank per reserved slot (#BSUB -n). Set it to start fewer ranks
MAP_OPTS=""             # e.g. "--map-by ppr:4:package --bind-to core" (needs NP set)

# Every run (serial and MPI, every job) appends one row to this one file.
# The scripts lock it while writing, so jobs that finish at the same time
# don't lose each other's rows.
RESULTS=results.csv
# ---------------------------------------------------------------------------

module load pandas/2.3.0-python-3.12.11
module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8

# One core per rank: stop NumPy/BLAS from starting hidden extra threads
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg            # no display in a batch job
export PYTHONUNBUFFERED=1        # print output straight away, in order

# --- record what we ran on (goes in the report)
echo "=== job ${LSB_JOBID} (${LSB_JOBNAME}) started $(date)"
echo "slots: ${LSB_DJOB_NUMPROC}   hosts/slots: ${LSB_MCPU_HOSTS}"
echo "cpu:   $(lscpu | grep 'Model name' | sed 's/Model name:[[:space:]]*//')"
echo "data:  $DATA   steps: $STEPS   dt: $DT   softening: $SOFTENING   merge radius: $MERGE_RADIUS"
echo "time limit: ${TIME_LIMIT} s   NP: ${NP:-all slots}   MAP_OPTS: ${MAP_OPTS:-none}"
module list 2>&1 | tail -n +2

if [ ! -f "$DATA" ]; then
    echo "Data file $DATA not found. Create it with: python3 GaiaSubsetSelection.py --save --no-plot"
    exit 1
fi

ARGS="--data $DATA --steps $STEPS --dt $DT --softening $SOFTENING --merge-radius $MERGE_RADIUS \
      --time-limit $TIME_LIMIT --results $RESULTS"

NP_OPT=""
[ -n "$NP" ] && NP_OPT="-np $NP"

# --report-bindings prints (to stderr) which core every rank is pinned to: check it!
MPIRUN="mpirun --report-bindings $NP_OPT $MAP_OPTS"

case "$MODE" in
    benchmark)
        $MPIRUN python3 ParallelImplementation.py $ARGS
        ;;
    validate)
        # Same input and options for both. Without merging the two final states
        # should be identical to the last digit; with merging the merge count
        # and final N must match as well.
        python3 SerialImplementation.py $ARGS --save-final "final_serial_${LSB_JOBID}.csv"
        $MPIRUN python3 ParallelImplementation.py $ARGS --save-final "final_mpi_${LSB_JOBID}.csv"
        python3 - "final_serial_${LSB_JOBID}.csv" "final_mpi_${LSB_JOBID}.csv" <<'EOF'
import sys
import numpy as np
import pandas as pd
serial, mpi = (pd.read_csv(path) for path in sys.argv[1:3])
if serial.shape != mpi.shape:
    sys.exit(f"VALIDATION FAILED: {len(serial)} stars (serial) vs {len(mpi)} stars (MPI)")
numeric = serial.select_dtypes("number").columns
max_diff = float(np.abs(serial[numeric].to_numpy() - mpi[numeric].to_numpy()).max())
print(f"Validation: {len(serial)} stars, identical={serial.equals(mpi)}, max abs difference={max_diff:.3e}")
EOF
        ;;
    *)
        echo "Unknown MODE=$MODE (use benchmark or validate)"; exit 1 ;;
esac

echo "=== job ${LSB_JOBID} finished $(date)"
