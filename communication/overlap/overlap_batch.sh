#!/bin/bash
# =============================================================================
#  Overlap in depth: Isend/Irecv against Sendrecv across the eager/rendezvous limit, 2 ranks
#
#  On the cluster, from the repository root:  bsub < communication/overlap/overlap_batch.sh
#  Results page afterwards (on the laptop):   python3 communication/overlap/make_page.py
#
#  Two full nodes are reserved, so nobody else shares them. The pair runs twice: both ranks on
#  the first node (intra) and one rank per node (inter). The log also records which transport and
#  which eager/rendezvous limits Open MPI uses, so the measured switch can be compared with them.
#  Output: communication/overlap/out/overlap_<jobid>.csv
# =============================================================================
#BSUB -J overlap
#BSUB -q hpcintro
#BSUB -n 48
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 00:20
#BSUB -o communication/overlap/logs/overlap_%J.out
#BSUB -e communication/overlap/logs/overlap_%J.err

PLACEMENTS=${PLACEMENTS:-"intra inter"}
BENCH_ARGS=${BENCH_ARGS:-""}            # e.g. "--reps 50" (see overlap_bench.py --help)
INTRA_OPTS="--report-bindings --map-by core --bind-to core"         # ranks 0 and 1 on cores 0 and 1 of the first node
INTER_OPTS="--report-bindings --map-by ppr:1:node --bind-to core"   # one rank on each node

cd "${LS_SUBCWD:-$PWD}" || exit 1
[ -f communication/overlap/overlap_bench.py ] || { echo "Start this from the repository root"; exit 1; }
mkdir -p communication/overlap/logs communication/overlap/out
module load pandas/2.3.0-python-3.12.11
module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 LC_ALL=C
OUT=communication/overlap/out/overlap_${LSB_JOBID}.csv

echo "=== overlap, job $LSB_JOBID, $(date), hosts: $(echo $LSB_HOSTS | tr ' ' '\n' | sort -u | tr '\n' ' ')"
echo "--- transports and limits (Open MPI, UCX)"
ompi_info 2>/dev/null | grep -E 'MCA (pml|btl|osc|coll): (ucx|ob1|cm|sm|vader|tcp|self|uct|tuned)' | sed 's/^ */  /'
ompi_info --param pml all --level 9 2>/dev/null | grep -i -E 'eager|rndv|thresh|priority' | sed 's/^ */  /' | head -20
ompi_info --param btl all --level 9 2>/dev/null | grep -i -E 'eager_limit|rndv_eager|max_send_size' | sed 's/^ */  /' | head -20
command -v ucx_info >/dev/null && ucx_info -c 2>/dev/null | grep -E 'UCX_(RNDV_THRESH|RNDV_SCHEME|ZCOPY_THRESH|TLS|PROTO_ENABLE)' | sed 's/^/  /'
env | grep -E '^(UCX_|OMPI_MCA_)' | sed 's/^/  env /'

for placement in $PLACEMENTS; do
    if [ "$placement" = inter ]; then opts=$INTER_OPTS; else opts=$INTRA_OPTS; fi
    export MPIRUN_OPTS="$opts"
    echo "--- $placement ($(date +%H:%M:%S)): mpirun $opts -np 2"
    mpirun $opts -np 2 python3 communication/overlap/overlap_bench.py --out "$OUT" --placement "$placement" $BENCH_ARGS 2>&1
    echo "--- $placement: exit code $?"
done
echo "=== finished $(date). Copy $OUT and communication/overlap/logs/overlap_${LSB_JOBID}.* off the cluster, then: python3 communication/overlap/make_page.py"
