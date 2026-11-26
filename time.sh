#!/bin/bash
# =============================================================================
#  time.sh: how many steps fit in 300 s, for N stars, at several rank counts.
#
#  One LSF job per rank count, largest first. Each job runs one cluster of N
#  stars with no step limit (--steps 1e9) and --time-limit 300: the program
#  stops itself at the limit and results.csv records how many steps it
#  finished (iterations). With no merges every step costs the same, so one run
#  per rank count gives the answer; growing the step count run by run would
#  only repeat it.
#
#  Usage (from the project root):
#      ./time.sh                    # N = 100, ranks 24 16 8 4 2 1, no IO
#      ./time.sh 500                # N = 500
#      ./time.sh 500 "24 8 1"       # N = 500, only these rank counts
#      ./time.sh 100 "48 24"        # 48 ranks = 2 whole nodes, then 24 ranks on 1 node
#      ./time.sh 100 --io           # with IO: a snapshot of every star every IO_EVERY steps
#      IO_EVERY=1000 IO_METHOD=gather ./time.sh 100 --io
#      ./time.sh 100 --program parallel             # ParallelImplementation.py (v1) instead
#      ./time.sh 100 --program parallel -- --decomposition ring
#      ./time.sh 500 "24" -- --merge-radius 0 --stop-check-every 100
#                                   # options after -- go to the program
#      ./time.sh --summary 500      # table of the finished runs for N = 500, all variants
#      ./time.sh 500 --dry-run      # print the job scripts instead of submitting
#
#  --no-io (default) / --io
#      IO doesn't buy steps, it only costs them (IO1: under 1 % of a run), so
#      the default runs have none during the run. --io adds snapshots
#      (IO3): ClusterImplementation.py --snapshot-every $IO_EVERY (default
#      10000) --snapshot-method $IO_METHOD (mpiio, the default, or gather),
#      written to snapshots/<run name>/. Only with --program cluster.
#  --program cluster (default) / parallel
#      cluster:  IO_Improvements/ClusterImplementation.py (lean per-step path)
#      parallel: ParallelImplementation.py, the team's code (v1 allgather by
#                default; ring or 2-D with -- --decomposition ...)
#      Both read the same file and use the same kernel options, so the
#      summary compares them directly.
#
#  The input is gaia_subsets/gaia_clusters_C1_n<N>.csv; it is made with
#  IO_Improvements/MakeClusters.py if it isn't there yet.
#  Every job reserves whole nodes, so no other job shares its cores: rank
#  counts up to 24 get one node (-n 24), above 24 they must be whole nodes
#  (48, 72, ...) and get -n P with span[ptile=24], 24 ranks per node. The one
#  cluster is then spread over several nodes (Whiteboard C2): its per-step
#  Allgatherv crosses between nodes, and the cluster code's start-up printout
#  warns that the cluster runs on more than one host. That is expected here.
#  Defaults: tiled kernel (--tile-rows 0) and merge radius 0.001 pc, as in the
#  C1 runs. TIME_LIMIT=60 ./time.sh ... gives shorter runs, e.g. for a test.
#  Run names: steps<limit>_<program>[_io-<method><every>]_n<N>_p<ranks>.
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
WITH_IO=false
PROGRAM=cluster
TIME_LIMIT=${TIME_LIMIT:-300}
IO_EVERY=${IO_EVERY:-10000}
IO_METHOD=${IO_METHOD:-mpiio}
MODULES="pandas/2.3.0-python-3.12.11 matplotlib/3.10.3-numpy-2.3.1-python-3.12.11 mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8"

positional=()
while [ $# -gt 0 ]; do
    case "$1" in
        --)         shift; EXTRA=("$@"); break ;;
        --dry-run)  DRY_RUN=true; shift ;;
        --summary)  SUMMARY=true; shift ;;
        --io)       WITH_IO=true; shift ;;
        --no-io)    WITH_IO=false; shift ;;
        --program)  PROGRAM=${2:-}; shift 2 || { echo "--program needs cluster or parallel" >&2; exit 1; } ;;
        -h|--help)  sed -n '2,49p' "$0"; exit 0 ;;
        -*)         echo "Unknown option '$1' (program options go after --)" >&2; exit 1 ;;
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
case "$PROGRAM" in
    cluster)  SCRIPT="IO_Improvements/ClusterImplementation.py" ;;
    parallel) SCRIPT="ParallelImplementation.py" ;;
    *)        echo "--program must be cluster or parallel, got '$PROGRAM'" >&2; exit 1 ;;
esac
if $WITH_IO; then
    [ "$PROGRAM" = cluster ] || { echo "--io needs --program cluster: ParallelImplementation.py has no snapshots" >&2; exit 1; }
    [[ "$IO_EVERY" =~ ^[0-9]+$ ]] && [ "$IO_EVERY" -ge 1 ] || { echo "IO_EVERY must be 1 or more, got '$IO_EVERY'" >&2; exit 1; }
    [[ "$IO_METHOD" == mpiio || "$IO_METHOD" == gather ]] || { echo "IO_METHOD must be mpiio or gather, got '$IO_METHOD'" >&2; exit 1; }
fi

DATA="gaia_subsets/gaia_clusters_C1_n${N}.csv"
VARIANT="${PROGRAM}"
IO_OPTIONS=""
if $WITH_IO; then
    VARIANT="${PROGRAM}_io-${IO_METHOD}${IO_EVERY}"
    IO_OPTIONS="--snapshot-every ${IO_EVERY} --snapshot-method ${IO_METHOD}"
fi

# Python with pandas: the modules, if they aren't loaded yet
if ! python3 -c "import pandas" 2>/dev/null; then
    if command -v module >/dev/null 2>&1; then
        module load $MODULES >/dev/null 2>&1 || true
    fi
    python3 -c "import pandas" 2>/dev/null || {
        echo "python3 can't import pandas. Load the modules first:  module load $MODULES" >&2; exit 1; }
fi

if $SUMMARY; then
    python3 - "$TIME_LIMIT" "$N" <<'EOF'
import re
import sys
import pandas as pd
limit, n = sys.argv[1], sys.argv[2]
pattern = re.compile(rf"^steps{limit}_(?P<variant>.+)_n{n}_p(?P<p>\d+)$")
runs = pd.read_csv("results.csv", low_memory=False)
names = runs["run_name"].astype(str).str.extract(pattern)
runs = runs[names["variant"].notna()].assign(variant=names["variant"])
if runs.empty:
    sys.exit(f"No runs named steps{limit}_<variant>_n{n}_p<ranks> in results.csv yet")
runs = runs.sort_values(["variant", "n_ranks", "timestamp"], ascending=[True, False, True])
ms = runs["step_time_mean_ms"].astype(float)
table = pd.DataFrame({
    "variant": runs["variant"],
    "decomposition": runs["decomposition"].fillna("") if "decomposition" in runs else "",
    "ranks": runs["n_ranks"].astype(int),
    "nodes": runs["n_nodes"].fillna(1).astype(int) if "n_nodes" in runs else 1,
    "steps in the run": runs["iterations"].astype(int),
    "ms/step": ms.round(3),
    f"steps in {limit} s (from ms/step)": (float(limit) * 1000 / ms).round(-3).astype(int),
    "IO time (s)": runs["io_time_s"].fillna(0).round(3) if "io_time_s" in runs else 0.0,
    "snapshots": runs["snapshots_written"].fillna(0).astype(int) if "snapshots_written" in runs else 0,
    "stop": runs["stop_reason"],
    "host (rank 0)": runs["hostname"],
    "when": runs["timestamp"],
})
print(f"N = {n}, {limit} s runs: one row per run, by variant, newest last within each rank count")
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
    NAME="steps${TIME_LIMIT}_${VARIANT}_n${N}_p${P}"
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
echo "=== ${NAME}: ${N} stars, ${P} ranks on \${LSB_MCPU_HOSTS}, ${SCRIPT}, as many steps as fit in ${TIME_LIMIT} s"
mpirun --report-bindings -np ${P} python3 ${SCRIPT} \\
    --data ${DATA} --steps 1000000000 --time-limit ${TIME_LIMIT} \\
    --tile-rows 0 --merge-radius 0.001 --run-name ${NAME} ${IO_OPTIONS} ${EXTRA_TEXT}
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
    echo "Submitted $(echo $RANKS | wc -w) job(s) (${VARIANT}). When they are done:  ./time.sh --summary ${N}"
fi