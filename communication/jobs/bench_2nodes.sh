#!/bin/bash
# =============================================================================
#  C0 on two nodes: what communication costs across the network (communication/bench/comm_bench.py)
#
#  From the repository root on the login node:   bsub < communication/jobs/bench_2nodes.sh
#  Two full nodes, 24 ranks each. Every P is split evenly over the two nodes, and pingpong
#  adds the pair "other node". Result: communication/results/bench_<jobid>.csv
#  Log: communication/logs/bench_<jobid>.out / .err (.err has the --report-bindings output)
# =============================================================================
#BSUB -J comm_bench_2nodes
#BSUB -q hpcintro
#BSUB -n 48
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 00:15
#BSUB -o communication/logs/bench_%J.out
#BSUB -e communication/logs/bench_%J.err

cd "${LS_SUBCWD:-$PWD}" || exit 1
module load pandas/2.3.0-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

echo "=== C0, job $LSB_JOBID, $(date), hosts $LSB_HOSTS" | cut -c1-400
mpirun --report-bindings --map-by core --bind-to core -np 48 \
    python3 communication/bench/comm_bench.py --time-limit 780
echo "=== finished $(date)"
