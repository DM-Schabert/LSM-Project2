#!/usr/bin/env bash

# Same HPC environment as the existing simulation
module load pandas/2.3.0-python-3.12.11
module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8

# Prevent additional NumPy/BLAS threads
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# Existing star subsets
export PILOT_DATA_TEMPLATE='gaia_subsets/gaia_random_{N}.csv'

# Run the original parallel simulation unchanged
export PILOT_SIM_COMMAND='mpirun -np {P} python3 ParallelImplementation.py --data {DATA} --steps {STEPS} --dt 0.01 --softening 0.01 --merge-radius 0 --no-results'