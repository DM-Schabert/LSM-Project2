#!/usr/bin/env bash
# Only edit THIS NEW FILE if the job cannot find mpi4py/MPI.
# Copy the module-loading and virtual-environment lines from the group's
# existing MPI job script. Do not edit that existing script.
#
# Example (not enabled; versions must match the group's environment):
# module load mpi/gcc
# source "$HOME/your_mpi_environment/bin/activate"
#
# Leave untouched if the batch environment already provides mpi4py + NumPy.
set -euo pipefail

# Detect missing dependencies early and fail with an actionable error.
command -v mpirun >/dev/null || {
    echo 'ERROR: mpirun not found. Configure the MPI module in data_types/hpc_env.sh' >&2
    exit 1
}
command -v python3 >/dev/null || {
    echo 'ERROR: python3 not found. Configure Python in data_types/hpc_env.sh' >&2
    exit 1
}
python3 - <<'PY'
import numpy
import mpi4py
from mpi4py import MPI
print('Environment OK | NumPy:', numpy.__version__, '| mpi4py:', mpi4py.__version__)
print('MPI library:', MPI.Get_library_version().splitlines()[0])
PY
