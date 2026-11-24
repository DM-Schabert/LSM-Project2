# Shared by the bench5 jobs (sourced, not run): environment, placement, the tile rule of every configuration,
# one run of a configuration, and copying the job's rows out of results.csv. Set LOCAL before sourcing it.

cd "${LS_SUBCWD:-$PWD}" || exit 1
[ -f bench5/configs.txt ] || { echo "Start this from the repository root"; exit 1; }
mkdir -p bench5/logs bench5/out
if [ "${LOCAL:-0}" = 1 ]; then                  # laptop: only checks that the scripts run, never a measurement
    export LSB_JOBID="local$(date +%Y%m%d%H%M%S)"
    MPIRUN=${MPIRUN:-mpiexec}; RESULTS=${RESULTS:-bench5/out/local_results.csv}
    place() { echo ""; }
    ranks() { case "$1" in 24) echo 2 ;; 48) echo 4 ;; *) echo "$1" ;; esac; }   # 24 -> 2, 48 -> 4 ranks
else
    module load pandas/2.3.0-python-3.12.11
    module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
    module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8
    MPIRUN=mpirun; RESULTS=${RESULTS:-results.csv}
    place() {   # Open MPI 5.0.8 fails silently if ppr leaves a socket partly empty, so one node maps by core
        case "$1" in
            close|full) echo "--report-bindings --map-by core --bind-to core" ;;   # cores 0-11 socket 0, 12-23 socket 1
            nodes2)     echo "--report-bindings --map-by ppr:24:node --bind-to core" ;;
        esac
    }
    ranks() { echo "$1"; }
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 LC_ALL=C
JOB=$LSB_JOBID
TARGET_PAIRS=13000      # ~1 MB of temporaries at 79 B per pair (cache/: tiles of this size stay fast on a full node)

configs() {     # the configuration lines: label|ranks|placement|scheme|tiles|since|what (comments and blanks left out)
    grep -v -E '^\s*(#|$)' bench5/configs.txt | tr -d ' ' | cut -d'|' -f1-5
}
grid_cols() {   # p_c of MPI.Compute_dims(P, 2) for the rank counts we use
    case "$1" in 48) echo 6 ;; 24) echo 4 ;; 12) echo 3 ;; 4) echo 2 ;; *) echo 1 ;; esac
}
tile_rows() {   # tile_rows <rule> <N> <P>: the numeric --tile-rows for a configuration's rule (at least 2 rows,
                # because 1 row switches the program to the serial row loop)
    local own=$(( ($2 + $3 - 1) / $3 )) k
    case "$1" in
        small) echo 2 ;;
        piece) k=$(( TARGET_PAIRS / own )); [ "$k" -lt 2 ] && k=2; echo "$k" ;;
        grid)  local cols=$(( ($2 + $(grid_cols "$3") - 1) / $(grid_cols "$3") )); k=$(( TARGET_PAIRS / cols )); [ "$k" -lt 2 ] && k=2; echo "$k" ;;
        *)     echo "$1" ;;
    esac
}
steps_for() {   # steps_for <N> <P> <seconds>: steps for about that long at ~110 ns per pair, at least MIN_STEPS
    awk -v n="$1" -v p="$2" -v s="$3" -v m="${MIN_STEPS:-3}" 'BEGIN { t = n * n / p * 110e-9 + 0.002; k = int(s / t); if (k < m) k = m; print k }'
}
run_config() {  # run_config <label|ranks|placement|scheme|tiles> <N> <name> <steps> <time limit>
    local label p pl scheme tiles np
    IFS='|' read -r label p pl scheme tiles <<< "$1"
    np=$(ranks "$p")
    local data="gaia_subsets/gaia_nearest_$2.csv" common="--data gaia_subsets/gaia_nearest_$2.csv --steps $4 --merge-radius 0 --time-limit $5 --results $RESULTS --run-name $3"
    if [ "$scheme" = serial ]; then
        python3 SerialImplementation.py $common < /dev/null > /dev/null      # < /dev/null: keep the caller's loop input
    else
        $MPIRUN $(place "$pl") -np "$np" python3 ParallelImplementation.py $common \
            --decomposition "$scheme" --tile-rows "$(tile_rows "$tiles" "$2" "$np")" < /dev/null > /dev/null
    fi
    # mpirun can return 0 although the program never ran: a run counts only if its row is in the results file
    # (run names end in _j<job>, so they are unique)
    if grep -q ",$3," "$RESULTS" 2>/dev/null; then
        echo "  $3: $np ranks, $4 steps max ($(date +%H:%M:%S))"
    else
        echo "  $3 FAILED (no row)"
    fi
}
subset() {      # make gaia_subsets/gaia_nearest_<N>.csv if it is missing
    [ -f "gaia_subsets/gaia_nearest_$1.csv" ] || python3 isoeff/make_subset.py --n "$1" || echo "could not make N=$1"
}
copy_rows() {   # this job's bench5 rows of results.csv -> bench5/out/runs_<job>.csv, found by the _j<job> at the end
                # of their run names (serial rows carry no job id of their own)
    python3 - "$RESULTS" "$JOB" "bench5/out/runs_${JOB}.csv" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1], low_memory=False)
name = d["run_name"].astype(str)
d = d[name.str.startswith("bench5") & name.str.endswith("_j" + sys.argv[2])]
d.to_csv(sys.argv[3], index=False)
print(f"{len(d)} runs -> {sys.argv[3]}")
EOF
}
