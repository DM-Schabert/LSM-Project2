#!/bin/bash
# =============================================================================
#  Step 2 check (Cluster_Plan_v3.md): with the far field off, every cluster of
#  a ClusterImplementation.py run must be bit-identical to a serial run on that
#  cluster's stars alone.
#
#  Submit from the project root (where SerialImplementation.py is):
#      bsub < IO_Improvements/validate_clusters.sh
#  To use other settings, edit DATA / STEPS / MERGE_RADIUS below: LSF does not
#  pass shell variables on to the job.
#
#  The job:
#    1. splits DATA into one CSV per cluster (DATA stem + _k0.csv, _k1.csv, ...)
#    2. runs SerialImplementation.py on each of them
#    3. runs ClusterImplementation.py on DATA with all reserved ranks
#    4. compares cluster by cluster and prints PASSED / FAILED at the end of the .out file
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
STEPS=${STEPS:-300}            # each serial run must finish within TIME_LIMIT (1 000 stars: ~0.13 s/step)
DT=${DT:-0.01}
SOFTENING=${SOFTENING:-0.01}
MERGE_RADIUS=${MERGE_RADIUS:-0.001}
TIME_LIMIT=${TIME_LIMIT:-240}  # per run; the serial runs come one after another, so keep the sum below -W
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

python3 IO_Improvements/compare_clusters.py split "$DATA" || exit 1
C=$(python3 -c "import pandas as pd, sys; print(int(pd.read_csv(sys.argv[1], usecols=['cluster'])['cluster'].max()) + 1)" "$DATA")

REFS=()
for (( k = 0; k < C; k++ )); do
    echo "--- serial run on cluster $k alone"
    python3 SerialImplementation.py --data "${STEM}_k${k}.csv" $ARGS --no-results \
        --save-final "clusters_ref_k${k}_${JOB}.csv" || exit 1
    REFS+=("clusters_ref_k${k}_${JOB}.csv")
done

echo "--- ClusterImplementation.py on all $C clusters"
$MPIRUN python3 IO_Improvements/ClusterImplementation.py --data "$DATA" $ARGS \
    --save-final "clusters_final_${JOB}.csv" || exit 1

echo "--- comparison"
python3 IO_Improvements/compare_clusters.py compare "clusters_final_${JOB}.csv" "${REFS[@]}"
echo "=== job ${JOB} finished $(date)"
