#!/bin/bash
# =============================================================================
#  time.sh: how many steps fit in 300 s, for N stars, at several rank counts.
#
#  One LSF job per rank count, largest first. Each job runs one cluster of N
#  stars with no step limit (--steps 1e9) and --time-limit 300: the program
#  stops itself at the limit and results.csv records how many steps it
#  finished (iterations). The job's walltime is the limit plus
#  STARTUP_MARGIN (120 s), rounded up to whole minutes: -W 0:07. The program's
#  clock starts after mpirun, module load and the imports, and it stops
#  stepping at 95 % of the limit to leave time for saving, so a job ends
#  around 5 to 5.5 minutes in, well inside the 7.
#      TIME_LIMIT=60 ./time.sh ...   a short test (a 3-minute job)
#  If a job is ever killed at the walltime (TERM_RUNLIMIT in its .out),
#  raise STARTUP_MARGIN. --summary projects "steps in 300 s" from ms/step, so
#  runs with other limits compare directly.
#
#  Usage (from the project root):
#      ./time.sh                    # N = 100, ranks 24 16 8 4 2 1, no IO
#      ./time.sh 500                # N = 500
#      ./time.sh 500 "24 8 1"       # N = 500, only these rank counts
#      ./time.sh 100 "48 24"        # 48 ranks = 2 whole nodes, then 24 ranks on 1 node
#      ./time.sh 100 --io           # with IO: a snapshot of every star every IO_EVERY steps
#      IO_EVERY=1000 IO_METHOD=gather ./time.sh 100 --io
#      ./time.sh 100 --program parallel             # ParallelImplementation.py (v1) instead
#      TAG=ring ./time.sh 100 --program parallel -- --decomposition ring
#      ./time.sh 500 "24" -- --merge-radius 0 --stop-check-every 100
#                                   # options after -- go to the program; without
#                                   # TAG they become the tag (see "Names and tags")
#      MERGE_RADIUS=0.01 ./time.sh 100 "24 8"       # more merges (name gets _mr0.01)
#      ./time.sh 1250 "96 72 48 24" --clusters per-node
#                                   # one cluster of 1250 stars per node (4, 3, 2, 1 clusters)
#      ./time.sh 5000 "96 48 24" --clusters per-node --total
#                                   # 5000 stars in all, split over one cluster per node
#                                   # (N must divide by every C: 72 ranks = 3 clusters)
#      ./time.sh 1000 "96" --clusters 8              # 8 clusters of 1000 on 4 nodes
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
#  --clusters 1 (default) / K / per-node
#      How many clusters the stars are split into. 1: one cluster, so every
#      rank needs every star each step (direct summation, as v1). per-node:
#      one cluster per node, C = P / CORES_PER_NODE (1 for P <= CORES_PER_NODE),
#      so each cluster's Allgatherv stays inside its node and only the
#      far-field Allreduce crosses between nodes (Whiteboard C2/C3). K: always
#      K clusters; P must be a multiple of K. Needs --program cluster.
#  --total
#      With --clusters: N is the total number of stars, split evenly over the
#      C clusters (N must divide by C). Without it N is the stars PER CLUSTER,
#      so the total grows with C. Use --total to compare with the
#      one-cluster runs at the same N.
#
#  Names and tags
#      TAG=<name> adds _<name> to the run name (letters, digits, '.' and '-').
#      Without TAG, the options after -- become the tag: "-- --decomposition
#      ring" gives _decomposition-ring. So runs with different program options
#      never share a name, and --summary shows them as separate variants.
#      MERGE_RADIUS (default 0.001 pc, as in the C1 runs) sets --merge-radius;
#      any other value adds _mr<value> to the name. Merging is part of what we
#      measure, so the summary shows the stars at start -> end and the merge
#      time per step. Prefer MERGE_RADIUS over "-- --merge-radius".
#
#  --summary N
#      One row per run, for every time limit (old steps300_ runs included;
#      the "limit s" column tells them apart). The "us" columns are microseconds per step: the
#      slowest rank's total for that section over the run, divided by the
#      steps. forces = the force kernel; comm = the Allgatherv of positions
#      (and the re-gathers after a merge); sync = the per-step Allreduce (stop
#      check, merge check and, with clusters, the far-field summaries); merge =
#      the merge path. v1 keeps its stop check out of ms/step, but it is in
#      sync. The summary warns when a run merged and when the rows come from
#      more than one CPU model.
#
#  The input is gaia_subsets/gaia_clusters_C<C>_n<n per cluster>.csv; it is
#  made with IO_Improvements/MakeClusters.py if it isn't there yet (all files
#  before any job is submitted). Options for MakeClusters go in MAKE_ARGS, e.g.
#  MAKE_ARGS="--min-gap 1.5" when C large clusters don't fit at theta <= 0.5.
#  Every job reserves whole nodes, so no other job shares its cores: rank
#  counts up to CORES_PER_NODE get one node, above it they must be whole
#  nodes (48, 72, ...) and get -n P with span[ptile=CORES_PER_NODE]. The one
#  cluster is then spread over several nodes (Whiteboard C2): its per-step
#  Allgatherv crosses between nodes, and the cluster code's start-up printout
#  warns that the cluster runs on more than one host. That is expected here.
#
#  hpcintro (bqueues -l hpcintro, 26 Nov): run limit 240 min (the walltime
#  can't go above it). LSF puts all
#  nodes of one job on the same CPU model and in one enclosure (RES_REQ
#  same[type:model] cu[maxcus=1]), but different jobs can land on different
#  models. CORES_PER_NODE (default 24) must match the queue's nodes, or
#  "whole node" isn't true: check with  bmgroup rhpcintro2  and bhosts.
#
#  Defaults: tiled kernel (--tile-rows 0) and merge radius 0.001 pc, as in the
#  C1 runs.
#  Run names (<limit> = TIME_LIMIT, 300 by default): steps<limit>_<program>[-C<C>x<n>][_io-<method><every>][_mr<r>][_<tag>]_n<total>_p<ranks>,
#  so --summary <total> shows the one-cluster, the split and the tagged runs side by side.
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
CLUSTERS=1
TOTAL=false
MAKE_ARGS=${MAKE_ARGS:-}
TIME_LIMIT=${TIME_LIMIT:-300}          # the program's --time-limit, s
STARTUP_MARGIN=${STARTUP_MARGIN:-120}  # s on top of it for start-up and saving -> the walltime
PROJECT_TO_S=300                       # --summary: steps in this many seconds, from ms/step
IO_EVERY=${IO_EVERY:-10000}
IO_METHOD=${IO_METHOD:-mpiio}
TAG=${TAG:-}
DEFAULT_MERGE_RADIUS=0.001
MERGE_RADIUS=${MERGE_RADIUS:-$DEFAULT_MERGE_RADIUS}
CORES_PER_NODE=${CORES_PER_NODE:-24}
QUEUE_RUNLIMIT_MIN=240   # hpcintro's RUNLIMIT
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
        --clusters) CLUSTERS=${2:-}; shift 2 || { echo "--clusters needs a number or per-node" >&2; exit 1; } ;;
        --total)    TOTAL=true; shift ;;
        -h|--help)  sed -n '2,/^# ====/p' "$0"; exit 0 ;;
        -*)         echo "Unknown option '$1' (program options go after --)" >&2; exit 1 ;;
        *)          positional+=("$1"); shift ;;
    esac
done
[ ${#positional[@]} -ge 1 ] && N=${positional[0]}
[ ${#positional[@]} -ge 2 ] && RANKS=${positional[1]}
if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ]; then
    echo "N must be a whole number of stars, got '$N'" >&2; exit 1
fi
if ! [[ "$CORES_PER_NODE" =~ ^[0-9]+$ ]] || [ "$CORES_PER_NODE" -lt 1 ]; then
    echo "CORES_PER_NODE must be a whole number, got '$CORES_PER_NODE'" >&2; exit 1
fi
for P in $RANKS; do
    [[ "$P" =~ ^[0-9]+$ ]] && [ "$P" -ge 1 ] || { echo "Rank counts must be whole numbers, got '$P'" >&2; exit 1; }
    if [ "$P" -gt $CORES_PER_NODE ] && [ $(( P % CORES_PER_NODE )) -ne 0 ]; then
        echo "Above $CORES_PER_NODE ranks, use whole nodes (multiples of $CORES_PER_NODE), got '$P'" >&2; exit 1
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

if ! [[ "$CLUSTERS" == per-node || ( "$CLUSTERS" =~ ^[0-9]+$ && "$CLUSTERS" -ge 1 ) ]]; then
    echo "--clusters must be a whole number or per-node, got '$CLUSTERS'" >&2; exit 1
fi
if [ "$CLUSTERS" != 1 ] && [ "$PROGRAM" != cluster ]; then
    echo "--clusters needs --program cluster: ParallelImplementation.py has no clusters" >&2; exit 1
fi

# Merge radius and tag: both go into the run name, so different settings never share one
if ! [[ "$MERGE_RADIUS" =~ ^([0-9]+\.?[0-9]*|\.[0-9]+)$ ]]; then
    echo "MERGE_RADIUS must be a number of pc (0 = no merging), got '$MERGE_RADIUS'" >&2; exit 1
fi
MR_TAG=""
[ "$MERGE_RADIUS" != "$DEFAULT_MERGE_RADIUS" ] && MR_TAG="_mr${MERGE_RADIUS}"
NAME_TAG=""
if [ -n "$TAG" ]; then
    [[ "$TAG" =~ ^[A-Za-z0-9.-]+$ ]] || { echo "TAG may only have letters, digits, '.' and '-', got '$TAG'" >&2; exit 1; }
    NAME_TAG="_${TAG}"
elif [ ${#EXTRA[@]} -gt 0 ]; then
    # "--decomposition ring --tile-rows 0" -> "decomposition-ring-tile-rows-0"
    slug=$(printf '%s ' "${EXTRA[@]}" | sed -E 's/(^| )-+/\1/g; s/[^A-Za-z0-9.]+/-/g; s/^-+//; s/-+$//')
    [ -n "$slug" ] && NAME_TAG="_${slug}"
fi

# Walltime: the program's time limit plus the start-up margin, in whole minutes
for var in TIME_LIMIT STARTUP_MARGIN; do
    [[ "${!var}" =~ ^[0-9]+$ ]] || { echo "$var must be a whole number of seconds, got '${!var}'" >&2; exit 1; }
done
[ "$TIME_LIMIT" -ge 1 ] || { echo "TIME_LIMIT must be 1 s or more" >&2; exit 1; }
WALLTIME_MIN=$(( (TIME_LIMIT + STARTUP_MARGIN + 59) / 60 ))
if [ "$WALLTIME_MIN" -gt $QUEUE_RUNLIMIT_MIN ]; then
    echo "TIME_LIMIT=$TIME_LIMIT needs $WALLTIME_MIN min of walltime; hpcintro allows $QUEUE_RUNLIMIT_MIN" >&2; exit 1
fi
WALLTIME=$(printf '%d:%02d' $(( WALLTIME_MIN / 60 )) $(( WALLTIME_MIN % 60 )))

clusters_for() {   # C for P ranks
    if [ "$CLUSTERS" = per-node ]; then
        if [ "$1" -le $CORES_PER_NODE ]; then echo 1; else echo $(( $1 / CORES_PER_NODE )); fi
    else
        echo "$CLUSTERS"
    fi
}
per_cluster_for() {   # stars per cluster for C clusters
    if $TOTAL; then echo $(( N / $1 )); else echo "$N"; fi
}
for P in $RANKS; do
    C=$(clusters_for "$P")
    if [ $(( P % C )) -ne 0 ]; then
        echo "$P ranks can't be split over $C clusters (must be a multiple)" >&2; exit 1
    fi
    if $TOTAL && [ $(( N % C )) -ne 0 ]; then
        echo "--total: $N stars don't split evenly over $C clusters" >&2; exit 1
    fi
done

IO_OPTIONS=""
IO_TAG=""
if $WITH_IO; then
    IO_TAG="_io-${IO_METHOD}${IO_EVERY}"
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
    python3 - "$N" "$PROJECT_TO_S" <<'EOF'
import re
import sys

import numpy as np
import pandas as pd

n, project_s = sys.argv[1], float(sys.argv[2])
pattern = re.compile(rf"^steps(?P<limit>\d+)_(?P<variant>.+)_n{n}_p(?P<p>\d+)$")
runs = pd.read_csv("results.csv", low_memory=False)
names = runs["run_name"].astype(str).str.extract(pattern)
runs = runs[names["variant"].notna()].assign(variant=names["variant"], limit=names["limit"].astype(int))
if runs.empty:
    sys.exit(f"No runs named steps<limit>_<variant>_n{n}_p<ranks> in results.csv yet")
runs = runs.sort_values(["variant", "n_ranks", "timestamp"], ascending=[True, False, True])


def col(name, default=0.0):
    """A results column, with `default` for rows (or files) from before the column existed."""
    if name in runs:
        return runs[name].fillna(default)
    return pd.Series(default, index=runs.index)


steps = runs["iterations"].astype(float)
per_step = steps.clip(lower=1)
ms = runs["step_time_mean_ms"].astype(float)


def us_per_step(name):
    """Microseconds per step of one timed section (slowest rank's total / steps)."""
    return (col(name).astype(float) / per_step * 1e6).round(1)


stars_start = col("n_stars_start", 0).astype(int)
stars_end = col("n_stars_end", 0).astype(int)
table = pd.DataFrame({
    "variant": runs["variant"],
    "decomposition": col("decomposition", ""),
    "limit s": runs["limit"],
    "ranks": runs["n_ranks"].astype(int),
    "nodes": col("n_nodes", 1).astype(int),
    "merge r": col("merge_radius_pc", np.nan),
    "stars": stars_start.astype(str) + "->" + stars_end.astype(str),
    "steps": steps.astype(int),
    "ms/step": ms.round(3),
    f"steps in {project_s:g} s": (project_s * 1000 / ms).replace([np.inf, -np.inf], np.nan)
                                 .round(-3).fillna(0).astype(int),
    "forces us": us_per_step("forces_time_s"),
    "comm us": us_per_step("communication_time_s"),
    "sync us": us_per_step("sync_time_s"),
    "merge us": us_per_step("merge_time_s"),
})
snapshots = col("snapshots_written", 0).astype(int)
if (snapshots > 0).any():  # IO columns only when some run wrote snapshots
    table["IO s"] = col("io_time_s").astype(float).round(3)
    table["snapshots"] = snapshots
table["stop"] = runs["stop_reason"]
table["host (rank 0)"] = runs["hostname"]
table["when"] = runs["timestamp"]

print(f"N = {n}: one row per run, by variant, newest last within each rank count. "
      f"'steps in {project_s:g} s' is projected from ms/step, so runs with different limits compare")
print("'us' columns: microseconds per step, slowest rank (comm = Allgatherv, sync = per-step Allreduce)")
print(table.to_string(index=False))

merged = table[stars_end < stars_start]
if not merged.empty:
    print(f"\nNote: {len(merged)} run(s) merged stars (stars start->end above). Their steps include merge "
          f"steps, which cost more than the others, and later steps have fewer stars: compare them with "
          f"runs that merged about as much.")
models = col("cpu_model", "").astype(str)
used = models[models != ""].value_counts()
if len(used) > 1:
    print("\nNote: these rows come from more than one CPU model; compare runs on the same model:")
    for model, count in used.items():
        print(f"  {count:3d} run(s): {model}")
EOF
    exit 0
fi

# Make every data file first, so a cluster layout that doesn't fit stops the
# script before any job is submitted
for P in $RANKS; do
    C=$(clusters_for "$P"); n=$(per_cluster_for "$C")
    DATA="gaia_subsets/gaia_clusters_C${C}_n${n}.csv"
    [ -f "$DATA" ] && continue
    echo "Making $DATA ($C cluster(s) of $n stars)"
    if $DRY_RUN; then
        echo "  (dry run: would run python3 IO_Improvements/MakeClusters.py --clusters $C --n $n $MAKE_ARGS)"
    else
        python3 IO_Improvements/MakeClusters.py --clusters "$C" --n "$n" $MAKE_ARGS || {
            echo "MakeClusters couldn't place $C clusters of $n stars; nothing was submitted." >&2
            echo "Try fewer stars per cluster, or e.g. MAKE_ARGS=\"--min-gap 1.5\" (larger theta)." >&2
            exit 1; }
    fi
done

EXTRA_TEXT="${EXTRA[*]:-}"

for P in $RANKS; do
    C=$(clusters_for "$P"); n=$(per_cluster_for "$C")
    DATA="gaia_subsets/gaia_clusters_C${C}_n${n}.csv"
    if [ "$C" = 1 ]; then   # the same run as without --clusters, so the same name
        VARIANT="${PROGRAM}${IO_TAG}"; CLUSTER_OPTION=""
    else
        VARIANT="${PROGRAM}-C${C}x${n}${IO_TAG}"; CLUSTER_OPTION="--clusters ${C}"
    fi
    NAME="steps${TIME_LIMIT}_${VARIANT}${MR_TAG}${NAME_TAG}_n$(( C * n ))_p${P}"
    if [ "$P" -le $CORES_PER_NODE ]; then
        SLOTS=$CORES_PER_NODE; SPAN="span[hosts=1]"           # one whole node
    else
        SLOTS=$P; SPAN="span[ptile=${CORES_PER_NODE}]"         # P / CORES_PER_NODE whole nodes
    fi
    JOB=$(cat <<EOF
#!/bin/bash
#BSUB -J ${NAME}
#BSUB -q hpcintro
#BSUB -n ${SLOTS}
#BSUB -R "${SPAN}"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W ${WALLTIME}
#BSUB -o ${NAME}_%J.out
#BSUB -e ${NAME}_%J.err
module load ${MODULES}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
echo "=== ${NAME}: ${C} cluster(s) of ${n} stars, ${P} ranks on \${LSB_MCPU_HOSTS}, ${SCRIPT}, as many steps as fit in ${TIME_LIMIT} s"
mpirun --report-bindings -np ${P} python3 ${SCRIPT} \\
    --data ${DATA} --steps 1000000000 --time-limit ${TIME_LIMIT} \\
    --tile-rows 0 --merge-radius ${MERGE_RADIUS} --run-name ${NAME} ${CLUSTER_OPTION} ${IO_OPTIONS} ${EXTRA_TEXT}
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
    if $TOTAL || [ "$CLUSTERS" = 1 ]; then
        echo "Submitted $(echo $RANKS | wc -w) job(s). When they are done:  ./time.sh --summary ${N}"
    else
        echo "Submitted $(echo $RANKS | wc -w) job(s). The total grows with C, so the summary is per total:"
        for P in $RANKS; do C=$(clusters_for "$P"); echo "  ./time.sh --summary $(( C * N ))   (P = $P, $C cluster(s))"; done
    fi
    if [ -n "${MR_TAG}${NAME_TAG}" ]; then
        echo "The run names end in '${MR_TAG}${NAME_TAG}' before _n...: these runs are a variant of their own in the summary."
    fi
fi