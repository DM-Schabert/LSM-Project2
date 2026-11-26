#!/bin/bash
# =============================================================================
#  time.sh: how many steps fit in 300 s, for N stars, at several rank counts.
#
#  One LSF job per rank count, largest first. Each job runs
#  IO_Improvements/ClusterImplementation.py on one cluster of N stars with no
#  step limit (--steps 1e9) and --time-limit 300: the program stops itself at
#  the limit and results.csv records how many steps it finished (iterations).
#  With no merges every step costs the same, so one run per rank count gives
#  the answer; growing the step count run by run would only repeat it.
#
#  Usage (from the project root):
#      ./time.sh                    # N = 100, ranks 24 16 8 4 2 1
#      ./time.sh 500                # N = 500
#      ./time.sh 500 "24 8 1"       # N = 500, only these rank counts
#      ./time.sh 100 "48 24"        # 48 ranks = 2 whole nodes, then 24 ranks on 1 node
#      ./time.sh 500 "24" -- --merge-radius 0 --stop-check-every 100
#                                   # options after -- go to ClusterImplementation.py
#      ./time.sh --summary 500      # table of the finished runs for N = 500
#      ./time.sh 500 --dry-run      # print the job scripts instead of submitting
#
#  The input is gaia_subsets/gaia_clusters_C1_n<N>.csv; it is made with
#  IO_Improvements/MakeClusters.py if it isn't there yet.
#  Every job reserves whole nodes, so no other job shares its cores: rank
#  counts up to 24 get one node (-n 24), above 24 they must be whole nodes
#  (48, 72, ...) and get -n P with span[ptile=24], 24 ranks per node. The one
#  cluster is then spread over several nodes (Whiteboard C2): its per-step
#  Allgatherv crosses between nodes, and the start-up printout warns that the
#  cluster runs on more than one host. That is expected here.
#  Defaults: tiled kernel (--tile-rows 0) and merge radius 0.001 pc, as in the
#  C1 runs. TIME_LIMIT=60 ./time.sh ... gives shorter runs, e.g. for a test.
# =============================================================================

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "Run this as  ./time.sh ...  or  bash time.sh ...  (not with source)" >&2
    return 1
fi
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

N=100
RANKS="24 16 8 4 2 1"
EXTRA=()
DRY_RUN=false
SUMMARY=false
TIME_LIMIT=${TIME_LIMIT:-300}
MODULES="pandas/2.3.0-python-3.12.11 matplotlib/3.10.3-numpy-2.3.1-python-3.12.11 mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8"

positional=()
while [ $# -gt 0 ]; do
    case "$1" in
        --)         shift; EXTRA=("$@"); break ;;
        --dry-run)  DRY_RUN=true; shift ;;
        --summary)  SUMMARY=true; shift ;;
        -h|--help)  sed -n '2,29p' "$0"; exit 0 ;;
        *)          positional+=("$1"); shift ;;
    esac
done
[ ${#positional[@]} -ge 1 ] && N=${positional[0]}
[ ${#positional[@]} -ge 2 ] && RANKS=${positional[1]}
if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ]; then
    echo "N must be a whole number of stars, got '$N'" >&2; exit 1
fi
CORES_PER_NODE=24
for P in $RANKS; do
    [[ "$P" =~ ^[0-9]+$ ]] && [ "$P" -ge 1 ] || { echo "Rank counts must be whole numbers, got '$P'" >&2; exit 1; }
    if [ "$P" -gt $CORES_PER_NODE ] && [ $(( P % CORES_PER_NODE )) -ne 0 ]; then
        echo "Above $CORES_PER_NODE ranks, use whole nodes (48, 72, ...), got '$P'" >&2; exit 1
    fi
done

DATA="gaia_subsets/gaia_clusters_C1_n${N}.csv"
NAME_PREFIX="steps${TIME_LIMIT}_n${N}"

# Python with pandas: the modules, if they aren't loaded yet
if ! python3 -c "import pandas" 2>/dev/null; then
    if command -v module >/dev/null 2>&1; then
        module load $MODULES >/dev/null 2>&1 || true
    fi
    python3 -c "import pandas" 2>/dev/null || {
        echo "python3 can't import pandas. Load the modules first:  module load $MODULES" >&2; exit 1; }
fi

if $SUMMARY; then
    python3 - "$NAME_PREFIX" "$TIME_LIMIT" <<'EOF'
import sys
import pandas as pd
prefix, limit = sys.argv[1], sys.argv[2]
runs = pd.read_csv("results.csv", low_memory=False)
runs = runs[runs["run_name"].astype(str).str.startswith(prefix + "_p")]
if runs.empty:
    sys.exit(f"No runs named {prefix}_p* in results.csv yet")
runs = runs.sort_values(["n_ranks", "timestamp"], ascending=[False, True])
table = pd.DataFrame({
    "ranks": runs["n_ranks"].astype(int),
    "steps in the run": runs["iterations"].astype(int),
    "ms/step": runs["step_time_mean_ms"].round(3),
    "steps in 300 s (from ms/step)": (300_000 / runs["step_time_mean_ms"]).round(-3).astype(int),
    "stop": runs["stop_reason"],
    "nodes": runs["n_nodes"].fillna(1).astype(int) if "n_nodes" in runs else 1,
    "host (rank 0)": runs["hostname"],
    "when": runs["timestamp"],
})
print(f"{prefix}: one row per run, newest last within each rank count")
print(table.to_string(index=False))
EOF
    exit 0
fi

if [ ! -f "$DATA" ]; then
    echo "Making $DATA (1 cluster of $N stars)"
    if $DRY_RUN; then
        echo "  (dry run: would run python3 IO_Improvements/MakeClusters.py --clusters 1 --n $N)"
    else
        python3 IO_Improvements/MakeClusters.py --clusters 1 --n "$N"
    fi
fi

WALLTIME_MIN=$(( (TIME_LIMIT + 120 + 59) / 60 ))   # the run, plus 2 minutes for start-up and saving
EXTRA_TEXT="${EXTRA[*]:-}"

for P in $RANKS; do
    NAME="${NAME_PREFIX}_p${P}"
    if [ "$P" -le $CORES_PER_NODE ]; then
        SLOTS=$CORES_PER_NODE; SPAN="span[hosts=1]"           # one whole node
    else
        SLOTS=$P; SPAN="span[ptile=${CORES_PER_NODE}]"         # P / 24 whole nodes
    fi
    JOB=$(cat <<EOF
#!/bin/bash
#BSUB -J ${NAME}
#BSUB -q hpcintro
#BSUB -n ${SLOTS}
#BSUB -R "${SPAN}"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 00:$(printf '%02d' "$WALLTIME_MIN")
#BSUB -o ${NAME}_%J.out
#BSUB -e ${NAME}_%J.err
module load ${MODULES}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
echo "=== ${NAME}: ${N} stars, ${P} ranks on \${LSB_MCPU_HOSTS}, as many steps as fit in ${TIME_LIMIT} s"
mpirun --report-bindings -np ${P} python3 IO_Improvements/ClusterImplementation.py \\
    --data ${DATA} --steps 1000000000 --time-limit ${TIME_LIMIT} \\
    --tile-rows 0 --merge-radius 0.001 --run-name ${NAME} ${EXTRA_TEXT}
EOF
)
    if $DRY_RUN; then
        echo "----- job for ${P} ranks"
        echo "$JOB"
    else
        echo "$JOB" | bsub
    fi
done

if ! $DRY_RUN; then
    echo "Submitted $(echo $RANKS | wc -w) job(s). When they are done:  ./time.sh --summary ${N}"
fi