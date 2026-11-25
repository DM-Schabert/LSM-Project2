#!/bin/bash
# =============================================================================
#  C1: every decomposition must give the serial result (allgather to the last
#  digit, ring and 2d to a small relative tolerance). Runs the serial program
#  once per N, then each decomposition at each P, and compares the final states.
#
#  Cluster (from the repository root):  bsub < communication/validate.sh
#      result: communication/out/validate_<jobid>.csv (copy it to your laptop)
#  Laptop:   LOCAL=1 P_LIST="1 2 3 4 6" bash communication/validate.sh
#
#  Settings (an export line above this block, or VAR=value locally):
#      N_LIST    subsets gaia_subsets/gaia_nearest_<N>.csv (default "5 100 1000")
#      P_LIST    rank counts (default "1 2 3 4 6 7 8 12 16 24")
#      STEPS     time steps (default 20), merging off
#      TOL       allowed relative difference (default 1e-9)
# =============================================================================
#BSUB -J comm_validate
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 00:20
#BSUB -o communication/logs/validate_%J.out
#BSUB -e communication/logs/validate_%J.err

N_LIST=${N_LIST:-"5 100 1000"}
P_LIST=${P_LIST:-"1 2 3 4 6 7 8 12 16 24"}
STEPS=${STEPS:-20}
TOL=${TOL:-1e-9}
LOCAL=${LOCAL:-0}

cd "${LS_SUBCWD:-$PWD}" || exit 1
if [ "$LOCAL" = 1 ]; then
    JOB="local$(date +%Y%m%d%H%M%S)"
    MPIRUN=${MPIRUN:-mpirun}
    PYTHON=${PYTHON:-python3}
else
    module load pandas/2.3.0-python-3.12.11
    module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
    module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8
    JOB=$LSB_JOBID
    MPIRUN="mpirun --map-by core --bind-to core"
    PYTHON=python3
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
TMP=$(mktemp -d "${TMPDIR:-/tmp}/c1_XXXXXX")
OUT=communication/out/validate_${JOB}.csv
COMMON="--steps $STEPS --merge-radius 0 --time-limit 600 --no-results"
VARIANTS=(
    "allgather|--decomposition allgather"
    "allgather-tile|--decomposition allgather --tile-rows 0"
    "ring-overlap|--decomposition ring --ring-comm overlap"
    "ring-blocking|--decomposition ring --ring-comm blocking"
    "ring-tile|--decomposition ring --tile-rows 0"
    "2d|--decomposition 2d"
    "2d-tile|--decomposition 2d --tile-rows 0"
)
echo "=== C1, job $JOB, $(date), host $(hostname); N: $N_LIST; P: $P_LIST; $STEPS steps; tol $TOL"
status=0
for n in $N_LIST; do
    data=gaia_subsets/gaia_nearest_$n.csv
    [ -f "$data" ] || { echo "skip N=$n: $data not found"; continue; }
    ref=$TMP/serial_N$n.csv
    $PYTHON SerialImplementation.py --data "$data" $COMMON --save-final "$ref" > "$TMP/serial_N$n.log" 2>&1 \
        || { echo "serial run failed for N=$n, see $TMP/serial_N$n.log"; status=1; continue; }
    runs=()
    for p in $P_LIST; do
        for v in "${VARIANTS[@]}"; do
            name=${v%%|*}; opts=${v#*|}
            f=$TMP/${name}_N${n}_P${p}.csv
            if $MPIRUN -np "$p" $PYTHON ParallelImplementation.py --data "$data" $COMMON $opts \
                   --save-final "$f" > "$TMP/${name}_N${n}_P${p}.log" 2>&1; then
                runs+=("$f")
            else
                echo "FAILED to run $name N=$n P=$p, see $TMP/${name}_N${n}_P${p}.log"; status=1
            fi
        done
        # 2d with the other grid shape: p_r = 2 (e.g. 2 x 3 instead of 3 x 2), where it divides P
        if [ "$p" -gt 2 ] && [ $((p % 2)) -eq 0 ]; then
            f=$TMP/2d-rows2_N${n}_P${p}.csv
            $MPIRUN -np "$p" $PYTHON ParallelImplementation.py --data "$data" $COMMON --decomposition 2d \
                --grid-rows 2 --save-final "$f" > "$TMP/2d-rows2_N${n}_P${p}.log" 2>&1 && runs+=("$f")
        fi
    done
    $PYTHON communication/compare_states.py "$ref" "${runs[@]}" --tol "$TOL" --out "$OUT" --job "$JOB" || status=1
done
grep -h "differ by up to" "$TMP"/*.log 2>/dev/null && status=1   # 2d row copies that drifted apart
echo "=== finished $(date): $([ $status = 0 ] && echo 'all PASSED' || echo 'something FAILED'); results in $OUT; logs in $TMP"
exit $status
