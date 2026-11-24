#!/bin/bash
# =============================================================================
#  C0 on one node: what communication costs, without physics (communication/bench/comm_bench.py)
#
#  From the repository root on the login node:   bsub < communication/jobs/bench_1node.sh
#  Result: communication/results/bench_<jobid>.csv  (copy it to your laptop, see communication/README.md)
#  Log:    communication/logs/bench_<jobid>.out / .err (.err has the --report-bindings output)
# =============================================================================
#BSUB -J comm_bench_1node
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 00:15
#BSUB -o communication/logs/bench_%J.out
#BSUB -e communication/logs/bench_%J.err

cd "${LS_SUBCWD:-$PWD}" || exit 1
module load pandas/2.3.0-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

echo "=== C0, job $LSB_JOBID, $(date), host $(hostname)"
# the same placement as the Amdahl runs: fill socket 0 first, one core per rank
mpirun --report-bindings --map-by core --bind-to core -np 24 \
    python3 communication/bench/comm_bench.py --time-limit 780
echo "=== finished $(date)"
