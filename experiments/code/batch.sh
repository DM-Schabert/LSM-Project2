#!/bin/bash
# Submit from the experiments directory: bsub < code/batch.sh
#BSUB -J merge_balance_v1
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 00:07
#BSUB -o merge_balance_%J.out
#BSUB -e merge_balance_%J.err

set -euo pipefail
module load pandas/2.3.0-python-3.12.11
module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
echo "job $LSB_JOBID; slots $LSB_DJOB_NUMPROC; hosts $LSB_MCPU_HOSTS"
module list
lscpu
python3 code/run_job.py --ranks 4
