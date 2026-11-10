#!/bin/bash
# =============================================================================
#  Amdahl sweep: serial baseline + MPI runs at P = 1, 2, 4, ... on ONE node
#
#  Submit from the repository root:   bsub < amdahl/amdahl_batch.sh
#  Other N without editing:           DATA=gaia_subsets/gaia_nearest_1000.csv STEPS=300 bsub < amdahl/amdahl_batch.sh
#  Laptop test (no LSF, tiny N):      LOCAL=1 DATA=gaia_subsets/gaia_nearest_100.csv STEPS=20 \
#                                     P_LIST="1 2 4" REPEATS=1 bash amdahl/amdahl_batch.sh
#  Analyse afterwards:                python3 amdahl/amdahl_analysis.py
#
#  Runs the unchanged SerialImplementation.py and ParallelImplementation.py.
#  Every run appends its normal row to $RESULTS (the shared results.csv) with
#  run_name = amdahl_<LABEL>_j<jobid>_r<repeat>, which is how the analysis finds it again.
#  In addition, every launch appends one row to $LAUNCH with the wall time of the
#  whole launch. That includes Python start-up, imports and MPI start-up, which
#  the programs cannot measure themselves but which are part of the serial fraction.
#
#  The whole node is reserved (-n 24) so that every P and the serial baseline run
#  on the same CPU: nodes of the same type differed by about 16 % in our first runs.
# =============================================================================
#BSUB -J amdahl
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 00:30
#BSUB -o amdahl/logs/amdahl_%J.out
#BSUB -e amdahl/logs/amdahl_%J.err

# ----------------------------------------------------------------- settings
# Every setting can also be set from outside (LSF passes your environment on), see above.
LABEL=${LABEL:-v1-numpy}              # name of the code version. Change it for every new parallelisation
DATA=${DATA:-gaia_subsets/gaia_nearest_5000.csv}
STEPS=${STEPS:-30}                    # fixed work: same N and steps for every P (strong scaling).
                                      # The step time hardly varies, so ~30 steps per point are enough
DT=${DT:-0.01}
SOFTENING=${SOFTENING:-0.01}
MERGE_RADIUS=${MERGE_RADIUS:-0}       # merging off: the work per step must not change during a run
P_LIST=${P_LIST:-"1 2 4 8 12 16 24"}
REPEATS=${REPEATS:-3}                 # the analysis takes the median per P
SERIAL=${SERIAL:-1}                   # 1: run SerialImplementation.py in every repeat (T(1) of Amdahl)
WALLTIME_MIN=${WALLTIME_MIN:-30}      # keep equal to #BSUB -W above
RESULTS=${RESULTS:-results.csv}       # the shared results file (file-locked by the programs)
LAUNCH=${LAUNCH:-amdahl/launch_times.csv}
LOCAL=${LOCAL:-0}                     # 1: laptop test, no LSF and no modules
# compact placement: fill socket 0 first. Record it, and keep it fixed when comparing versions
MPIRUN_OPTS_CLUSTER="--report-bindings --map-by core --bind-to core"
# ---------------------------------------------------------------------------

export LC_ALL=C                       # decimal points, not commas, in the timestamps
LABEL=${LABEL//_/-}                   # '_' separates the parts of run_name

cd "${LS_SUBCWD:-$PWD}" || exit 1
if [ ! -f ParallelImplementation.py ] || [ ! -f SerialImplementation.py ]; then
    echo "Start this from the repository root: bsub < amdahl/amdahl_batch.sh"; exit 1
fi

if [ "$LOCAL" = 1 ]; then
    JOBID="local$(date +%Y%m%d%H%M%S)"
    SLOTS=1000                        # laptops oversubscribe; MPICH does that by itself
    MPIRUN_OPTS=${MPIRUN_OPTS:-""}    # Open MPI needs --oversubscribe here
else
    module load pandas/2.3.0-python-3.12.11
    module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
    module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8
    JOBID=$LSB_JOBID
    SLOTS=${LSB_DJOB_NUMPROC:-24}
    MPIRUN_OPTS=${MPIRUN_OPTS:-$MPIRUN_OPTS_CLUSTER}
fi

# One core per rank: no hidden extra threads from NumPy/BLAS (or Numba, once it is used)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

now() {   # seconds since 1970 with microseconds; bash 5 has it built in, the macOS bash does not
    if [ -n "${EPOCHREALTIME:-}" ]; then echo "$EPOCHREALTIME"; else python3 -c 'import time; print(f"{time.time():.6f}")'; fi
}
JOB_START=$(now)
DEADLINE=$(awk -v s="$JOB_START" -v w="$WALLTIME_MIN" 'BEGIN { printf "%d", s + w * 60 - 60 }')

echo "=== Amdahl sweep, job ${JOBID}, started $(date)"
echo "label: $LABEL   data: $DATA   steps: $STEPS   dt: $DT   softening: $SOFTENING   merge radius: $MERGE_RADIUS"
echo "P: $P_LIST   repeats: $REPEATS   serial baseline: $SERIAL   slots: $SLOTS   mpirun options: ${MPIRUN_OPTS:-none}"
CPU=$(lscpu 2>/dev/null | grep -m1 'Model name' | sed 's/Model name:[[:space:]]*//')
[ -z "$CPU" ] && CPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)
echo "host: $(hostname)   cpu: $CPU"
echo "results: $RESULTS   launch times: $LAUNCH"
[ "$LOCAL" = 1 ] || module list 2>&1 | tail -n +2

if [ ! -f "$DATA" ]; then
    echo "Data file $DATA not found. Create it with: python3 GaiaSubsetSelection.py --save --no-plot"; exit 1
fi
if [ ! -f "$LAUNCH" ]; then
    echo "lsb_jobid,label,run_name,implementation,n_ranks,repeat,data_file,steps,host,mpirun_opts,exit_code,launch_wall_s,start_epoch" > "$LAUNCH"
fi

launch() {   # launch <serial|mpi> <ranks> <repeat>
    local impl=$1 np=$2 rep=$3
    local left
    left=$(awk -v d="$DEADLINE" -v n="$(now)" 'BEGIN { printf "%d", d - n }')
    if [ "$left" -lt 60 ]; then
        echo "--- skipped $impl P=$np repeat $rep: only ${left} s of walltime left"; return
    fi
    local name="amdahl_${LABEL}_j${JOBID}_r${rep}"
    # --time-limit: the program stops itself before LSF kills the job (a killed run saves nothing)
    local args="--data $DATA --steps $STEPS --dt $DT --softening $SOFTENING --merge-radius $MERGE_RADIUS
                --time-limit $((left - 30)) --results $RESULTS --run-name $name"
    local opts="" t0 t1 code wall
    echo "--- $impl P=$np repeat $rep ($(date +%H:%M:%S))"
    t0=$(now)
    if [ "$impl" = serial ]; then
        python3 SerialImplementation.py $args
    else
        opts=$MPIRUN_OPTS
        mpirun $opts -np "$np" python3 ParallelImplementation.py $args
    fi
    code=$?
    t1=$(now)
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.4f", b - a }')
    echo "$JOBID,$LABEL,$name,$impl,$np,$rep,$DATA,$STEPS,$(hostname),\"$opts\",$code,$wall,$t0" >> "$LAUNCH"
    echo "--- $impl P=$np repeat $rep: ${wall} s launch wall time, exit code $code"
}

for rep in $(seq 1 "$REPEATS"); do
    # every second repeat runs the P values in reverse, so a slow drift of the node hits every P alike
    order=$P_LIST
    [ $((rep % 2)) -eq 0 ] && order=$(echo $P_LIST | tr ' ' '\n' | sort -rn | tr '\n' ' ')
    [ "$SERIAL" = 1 ] && [ $((rep % 2)) -eq 1 ] && launch serial 1 "$rep"
    for np in $order; do
        if [ "$np" -gt "$SLOTS" ]; then echo "--- skipped P=$np: only $SLOTS slots reserved"; continue; fi
        launch mpi "$np" "$rep"
    done
    [ "$SERIAL" = 1 ] && [ $((rep % 2)) -eq 0 ] && launch serial 1 "$rep"
done

echo "=== job ${JOBID} finished $(date). Analyse with: python3 amdahl/amdahl_analysis.py"
