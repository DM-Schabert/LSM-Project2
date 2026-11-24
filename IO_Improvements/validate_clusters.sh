#!/bin/bash
# =============================================================================
#  Checks of the cluster version (Cluster_Plan_v3.md, steps 2 and 3).
#
#  Submit from the project root (where SerialImplementation.py is):
#      bsub < IO_Improvements/validate_clusters.sh
#  To use other settings, edit the settings below: LSF does not pass shell
#  variables on to the job.
#
#  Step 2, far field off: every cluster must be bit-identical to a serial run
#    on that cluster's stars alone.
#      1. split DATA into one CSV per cluster (DATA stem + _k0.csv, _k1.csv, ...)
#      2. SerialImplementation.py on each of them
#      3. ClusterImplementation.py --far-field off on DATA, then compare
#  Step 3, far field on:
#      4. cluster_reference.py simulate: the same approximation without MPI
#      5. ClusterImplementation.py --far-field monopole, then compare (to TOL_FAR:
#         the ranks add their parts of a cluster's sums in another order)
#      6. cluster_reference.py forces: the approximation's error against direct
#         summation at t = 0, for the distance factors in SCALES; also saved as CSV
#  The last lines of the .out file say PASSED or FAILED for steps 2 and 3.
#  The number of ranks (#BSUB -n) must be a multiple of the number of clusters.
# =============================================================================
#BSUB -J clusters_validate
#BSUB -q hpcintro
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 00:10
#BSUB -o clusters_validate_%J.out
#BSUB -e clusters_validate_%J.err

DATA=${DATA:-gaia_subsets/gaia_clusters_C2_n1000.csv}
STEPS=${STEPS:-300}            # each serial run must finish within TIME_LIMIT (1 000 stars: ~0.14 s/step)
DT=${DT:-0.01}
SOFTENING=${SOFTENING:-0.01}
MERGE_RADIUS=${MERGE_RADIUS:-0.001}
TIME_LIMIT=${TIME_LIMIT:-240}  # per run; the runs come one after another, so keep the total below -W
TOL_FAR=${TOL_FAR:-1e-12}      # step 3: largest allowed relative difference to the serial reference
SCALES=${SCALES:-"0.25 0.5 1 2 4"}  # step 3: distance factors between the clusters for the force error
MPIRUN=${MPIRUN:-mpirun --report-bindings}
JOB=${LSB_JOBID:-local}

module load pandas/2.3.0-python-3.12.11
module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

echo "=== job ${JOB} started $(date) on ${LSB_MCPU_HOSTS:-this machine}"
echo "data: $DATA   steps: $STEPS   dt: $DT   softening: $SOFTENING   merge radius: $MERGE_RADIUS"
if [ ! -f "$DATA" ]; then
    echo "Data file $DATA not found. Make it with: python3 IO_Improvements/MakeClusters.py --clusters 2 --n 1000"
    exit 1
fi

ARGS="--steps $STEPS --dt $DT --softening $SOFTENING --merge-radius $MERGE_RADIUS --time-limit $TIME_LIMIT"
STEM="${DATA%.csv}"

# ---- step 2: far field off, each cluster alone
python3 IO_Improvements/compare_clusters.py split "$DATA" || exit 1
C=$(python3 -c "import pandas as pd, sys; print(int(pd.read_csv(sys.argv[1], usecols=['cluster'])['cluster'].max()) + 1)" "$DATA")

REFS=()
for (( k = 0; k < C; k++ )); do
    echo "--- step 2: serial run on cluster $k alone"
    python3 SerialImplementation.py --data "${STEM}_k${k}.csv" $ARGS --no-results \
        --save-final "clusters_ref_k${k}_${JOB}.csv" || exit 1
    REFS+=("clusters_ref_k${k}_${JOB}.csv")
done

echo "--- step 2: ClusterImplementation.py --far-field off on all $C clusters"
$MPIRUN python3 IO_Improvements/ClusterImplementation.py --data "$DATA" $ARGS --far-field off \
    --save-final "clusters_final_${JOB}.csv" || exit 1

echo "--- step 2: comparison"
python3 IO_Improvements/compare_clusters.py compare "clusters_final_${JOB}.csv" "${REFS[@]}"
STEP2=$?

# ---- step 3: far field on
echo "--- step 3: serial reference of the same approximation (cluster_reference.py simulate)"
python3 IO_Improvements/cluster_reference.py simulate "$DATA" --steps $STEPS --dt $DT --softening $SOFTENING \
    --merge-radius $MERGE_RADIUS --save-final "clusters_farref_${JOB}.csv" || exit 1

echo "--- step 3: ClusterImplementation.py --far-field monopole on all $C clusters"
$MPIRUN python3 IO_Improvements/ClusterImplementation.py --data "$DATA" $ARGS --far-field monopole \
    --save-final "clusters_far_${JOB}.csv" || exit 1

echo "--- step 3: comparison"
python3 IO_Improvements/compare_clusters.py compare "clusters_far_${JOB}.csv" "clusters_farref_${JOB}.csv" --tol $TOL_FAR
STEP3=$?

echo "--- step 3: force error against direct summation at t = 0"
python3 IO_Improvements/cluster_reference.py forces "$DATA" --softening $SOFTENING --scale $SCALES \
    --out "force_error_$(basename "$STEM")_${JOB}.csv"

echo "=== step 2 (far field off, each cluster alone): $([ $STEP2 -eq 0 ] && echo PASSED || echo FAILED)"
echo "=== step 3 (far field on, against the serial reference): $([ $STEP3 -eq 0 ] && echo PASSED || echo FAILED)"
echo "=== job ${JOB} finished $(date)"
