#!/bin/bash
# =============================================================================
#  Start a scaling campaign for the current code (run on a login node, from
#  anywhere; it works in the code folder that contains isoeff/).
#
#      ./isoeff/start.sh v1 --merge-radius 0.001 --note "allgather"
#      ./isoeff/start.sh v2-ring --continue-from isoeff/out/v1 --mpi-args "--decomposition ring"
#      ./isoeff/start.sh v3 --continue-from isoeff/out/v2-ring --max-nodes 2
#
#  Default: capacity mode, 10,000 steps per run in a 7-minute job. Everything
#  after the label goes to 'controller.py init' (see its --help). The first job
#  is submitted at once; every job submits the next one itself.
#
#  Follow it:   python3 isoeff/controller.py status --campaign isoeff/out/<label> [--log]
#  Resume it:   python3 isoeff/controller.py next   --campaign isoeff/out/<label>
#  First time:  chmod +x isoeff/start.sh
# =============================================================================
set -euo pipefail
if [ $# -lt 1 ] || [[ "$1" == -* ]]; then
    echo "Usage: $0 <label> [init options]   e.g. $0 v2-ring --continue-from isoeff/out/v1 --mpi-args '--decomposition ring'" >&2
    exit 1
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# pandas/numpy are needed to make the first subset (the controller itself is stdlib only)
if command -v module >/dev/null 2>&1; then
    module load pandas/2.3.0-python-3.12.11 >/dev/null 2>&1 || true
fi

label="$1"; shift
python3 isoeff/controller.py init --label "$label" "$@" --submit
