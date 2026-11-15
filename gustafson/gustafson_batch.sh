#!/bin/bash
# =============================================================================
#  Gustafson sweep (weak scaling): the problem grows with P, on ONE node
#
#  All sweeps at once (recommended):  bash gustafson/run_all.sh
#  One sweep with the defaults below: bsub < gustafson/gustafson_batch.sh
#  One sweep with other settings:     sed -e 's|N0:-1000|N0:-300|' -e 's|STEPS:-150|STEPS:-1200|' \
#                                         gustafson/gustafson_batch.sh | bsub
#    (VAR=value bsub < ... does NOT work on the DTU cluster: it does not pass your environment on)
#  Laptop test (no LSF, tiny N):      LOCAL=1 N0=20 STEPS=200 P_LIST="1 2 4" REPEATS=1 \
#                                     bash gustafson/gustafson_batch.sh
#  Analyse afterwards:                python3 gustafson/gustafson_analysis.py
#
#  For every P the problem has N(P) stars (gustafson/plan.py: with RULE=time every rank keeps
#  the serial run time of the base problem with N0 stars, with RULE=pairs its number of pair
#  interactions). For every P the job runs
#      ParallelImplementation.py on N(P) stars with P ranks, STEPS steps        -> T(P)
#      SerialImplementation.py   on N(P) stars, about STEPS/P steps (at least 10) -> tau(P),
#                                the serial time of the scaled problem
#  T(1) = tau(1) is the serial run on N0 stars with STEPS steps. Only times per step are
#  compared, so the serial runs of the big problems can be short.
#
#  Runs the unchanged SerialImplementation.py and ParallelImplementation.py. Every run appends
#  its normal row to $RESULTS (the shared results.csv) with
#  run_name = gustafson_<LABEL>_<RULE><N0>_j<jobid>_r<repeat>, which is how the analysis finds it.
#  Every launch also appends one row to $LAUNCH with the wall time of the whole launch
#  (Python start-up, imports and MPI start-up included).
#
#  The whole node is reserved (-n 24) so that every P and every serial run use the same CPU:
#  nodes of the same type differed by about 16 % in the Amdahl runs.
# =============================================================================
#BSUB -J gustafson
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 00:30
#BSUB -o gustafson/logs/gustafson_%J.out
#BSUB -e gustafson/logs/gustafson_%J.err

# ----------------------------------------------------------------- settings
# Every setting can be preset by an export line above this block; run_all.sh writes those.
# (Locally, VAR=value bash gustafson/gustafson_batch.sh works too.)
LABEL=${LABEL:-v1-numpy}              # name of the code version. Change it for every new parallelisation
RULE=${RULE:-time}                    # how N grows with P: time (same serial time per rank) or pairs (N0*sqrt(P))
N0=${N0:-1000}                        # stars of the base problem (P = 1)
N_LIST=${N_LIST:-}                    # N for every P, in P_LIST order. Empty: computed by gustafson/plan.py
COST_A_US=${COST_A_US:-}              # cost model for RULE=time (empty: the defaults in plan.py, fitted to v1-numpy)
COST_B_NS=${COST_B_NS:-}
STEPS=${STEPS:-150}                   # steps of every MPI run and of the serial base run (~20 s per run)
SERIAL_MIN_STEPS=${SERIAL_MIN_STEPS:-10}   # serial runs of the scaled problems: max(this, ceil(STEPS/P)) steps
DT=${DT:-0.01}
SOFTENING=${SOFTENING:-0.01}
MERGE_RADIUS=${MERGE_RADIUS:-0}       # merging off: the work per step must not change during a run
P_LIST=${P_LIST:-"1 2 4 8 12 16 24"}
REPEATS=${REPEATS:-3}                 # the analysis takes the median per P
SERIAL=${SERIAL:-1}                   # 1: serial run on the scaled problem of every P (tau(P) of Gustafson)
WALLTIME_MIN=${WALLTIME_MIN:-30}      # keep equal to #BSUB -W above
RESULTS=${RESULTS:-results.csv}       # the shared results file (file-locked by the programs)
LAUNCH=${LAUNCH:-gustafson/launch_times.csv}
DATA_DIR=${DATA_DIR:-gaia_subsets}
LOCAL=${LOCAL:-0}                     # 1: laptop test, no LSF and no modules
# compact placement: fill socket 0 first. The same as the Amdahl sweeps, so the two can be compared
MPIRUN_OPTS_CLUSTER="--report-bindings --map-by core --bind-to core"
# ---------------------------------------------------------------------------

export LC_ALL=C                       # decimal points, not commas, in the timestamps
LABEL=${LABEL//_/-}                   # '_' separates the parts of run_name

cd "${LS_SUBCWD:-$PWD}" || exit 1
if [ ! -f ParallelImplementation.py ] || [ ! -f SerialImplementation.py ]; then
    echo "Start this from the repository root: bsub < gustafson/gustafson_batch.sh"; exit 1
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

# N for every P
if [ -z "$N_LIST" ]; then
    plan_args="--n0 $N0 --rule $RULE --p $P_LIST --sizes"
    [ -n "$COST_A_US" ] && plan_args+=" --a-us $COST_A_US"
    [ -n "$COST_B_NS" ] && plan_args+=" --b-ns $COST_B_NS"
    N_LIST=$(python3 gustafson/plan.py $plan_args) || { echo "gustafson/plan.py failed"; exit 1; }
fi
read -r -a PS <<< "$P_LIST"
read -r -a NS <<< "$N_LIST"
if [ "${#PS[@]}" != "${#NS[@]}" ]; then
    echo "P_LIST ($P_LIST) and N_LIST ($N_LIST) differ in length"; exit 1
fi
if [ "${NS[0]}" != "$N0" ] || [ "${PS[0]}" != 1 ]; then
    echo "P_LIST must start with 1 and N_LIST with N0 = $N0 (got P = ${PS[0]}, N = ${NS[0]})"; exit 1
fi
n_of() {   # n_of <P>: N of that P
    local k
    for k in "${!PS[@]}"; do [ "${PS[$k]}" = "$1" ] && { echo "${NS[$k]}"; return; }; done
}
data_of() { echo "$DATA_DIR/gaia_nearest_$1.csv"; }

now() {   # seconds since 1970 with microseconds; bash 5 has it built in, the macOS bash does not
    if [ -n "${EPOCHREALTIME:-}" ]; then echo "$EPOCHREALTIME"; else python3 -c 'import time; print(f"{time.time():.6f}")'; fi
}
JOB_START=$(now)
DEADLINE=$(awk -v s="$JOB_START" -v w="$WALLTIME_MIN" 'BEGIN { printf "%d", s + w * 60 - 60 }')
NAME="gustafson_${LABEL}_${RULE}${N0}_j${JOBID}"

echo "=== Gustafson sweep, job ${JOBID}, started $(date)"
echo "label: $LABEL   rule: $RULE   N0: $N0   steps: $STEPS (serial runs at N(P): max($SERIAL_MIN_STEPS, ceil($STEPS/P)))"
echo "P: $P_LIST"
echo "N: $N_LIST"
echo "dt: $DT   softening: $SOFTENING   merge radius: $MERGE_RADIUS   repeats: $REPEATS   serial runs: $SERIAL"
echo "slots: $SLOTS   mpirun options: ${MPIRUN_OPTS:-none}"
CPU=$(lscpu 2>/dev/null | grep -m1 'Model name' | sed 's/Model name:[[:space:]]*//')
[ -z "$CPU" ] && CPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)
echo "host: $(hostname)   cpu: $CPU"
echo "results: $RESULTS   launch times: $LAUNCH"
[ "$LOCAL" = 1 ] || module list 2>&1 | tail -n +2

missing=""
for n in "${NS[@]}"; do [ -f "$(data_of "$n")" ] || missing+=" $n"; done
if [ -n "$missing" ]; then
    echo "Data files missing for N =$missing. Create them with:"
    echo "    python3 GaiaSubsetSelection.py --n$missing --save --no-plot"; exit 1
fi

LAUNCH_HEADER="lsb_jobid,label,rule,n0,run_name,implementation,n_ranks,scaled_p,repeat,data_file,n_stars,steps,host,mpirun_opts,exit_code,launch_wall_s,start_epoch"
append_launch() {   # several jobs may write to $LAUNCH at the same time: take a lock where flock exists
    if command -v flock >/dev/null 2>&1; then
        ( flock 9; [ -s "$LAUNCH" ] || echo "$LAUNCH_HEADER" >> "$LAUNCH"; echo "$1" >> "$LAUNCH" ) 9>>"$LAUNCH.lock"
    else
        [ -s "$LAUNCH" ] || echo "$LAUNCH_HEADER" >> "$LAUNCH"; echo "$1" >> "$LAUNCH"
    fi
}

launch() {   # launch <serial|mpi> <P of the scaled problem> <repeat>
    local impl=$1 p=$2 rep=$3
    local n np steps data left
    n=$(n_of "$p"); data=$(data_of "$n")
    if [ "$impl" = serial ]; then
        np=1; steps=$STEPS
        [ "$p" -gt 1 ] && steps=$(( (STEPS + p - 1) / p )) && [ "$steps" -lt "$SERIAL_MIN_STEPS" ] && steps=$SERIAL_MIN_STEPS
    else
        np=$p; steps=$STEPS
    fi
    left=$(awk -v d="$DEADLINE" -v t="$(now)" 'BEGIN { printf "%d", d - t }')
    if [ "$left" -lt 60 ]; then
        echo "--- skipped $impl P=$p (N=$n) repeat $rep: only ${left} s of walltime left"; return
    fi
    local name="${NAME}_r${rep}"
    # --time-limit: the program stops itself before LSF kills the job (a killed run saves nothing)
    local args="--data $data --steps $steps --dt $DT --softening $SOFTENING --merge-radius $MERGE_RADIUS
                --time-limit $((left - 30)) --results $RESULTS --run-name $name"
    local opts="" t0 t1 code wall
    echo "--- $impl on N=$n (problem of P=$p) with $np rank(s), $steps steps, repeat $rep ($(date +%H:%M:%S))"
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
    append_launch "$JOBID,$LABEL,$RULE,$N0,$name,$impl,$np,$p,$rep,$data,$n,$steps,$(hostname),\"$opts\",$code,$wall,$t0"
    echo "--- $impl N=$n ($np rank(s), problem of P=$p) repeat $rep: ${wall} s launch wall time, exit code $code"
}

for rep in $(seq 1 "$REPEATS"); do
    # every second repeat runs the P values in reverse (and MPI before serial), so a slow drift
    # of the node hits every P alike. Serial and MPI of the same P run back to back.
    order=$P_LIST
    [ $((rep % 2)) -eq 0 ] && order=$(echo $P_LIST | tr ' ' '\n' | sort -rn | tr '\n' ' ')
    for p in $order; do
        if [ "$p" -gt "$SLOTS" ]; then echo "--- skipped P=$p: only $SLOTS slots reserved"; continue; fi
        if [ $((rep % 2)) -eq 1 ]; then
            { [ "$SERIAL" = 1 ] || [ "$p" = 1 ]; } && launch serial "$p" "$rep"
            launch mpi "$p" "$rep"
        else
            launch mpi "$p" "$rep"
            { [ "$SERIAL" = 1 ] || [ "$p" = 1 ]; } && launch serial "$p" "$rep"
        fi
    done
done

echo "=== job ${JOBID} finished $(date). Analyse with: python3 gustafson/gustafson_analysis.py"
