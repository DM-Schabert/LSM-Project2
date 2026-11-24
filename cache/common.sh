# Shared by the cache job scripts (sourced, not run): environment, rank placement, one run of the program,
# and copying the job's rows out of results.csv. Set LOCAL (and any settings) before sourcing it.

cd "${LS_SUBCWD:-$PWD}" || exit 1
[ -f cache/bandwidth.py ] || { echo "Start this from the repository root"; exit 1; }
mkdir -p cache/logs cache/out
if [ "${LOCAL:-0}" = 1 ]; then                  # laptop: only checks that the scripts run, never a measurement
    export LSB_JOBID="local$(date +%Y%m%d%H%M%S)"
    MPIRUN=${MPIRUN:-mpiexec}; RESULTS=${RESULTS:-cache/out/local_results.csv}
    place() { echo ""; }
else
    module load pandas/2.3.0-python-3.12.11
    module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
    module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8
    MPIRUN=mpirun; RESULTS=${RESULTS:-results.csv}
    place() {   # place <placement> <P>: mpirun options; "package" is a socket, as in the week 2 Jacobi lab
        # Open MPI 5.0.8 aborts with a PMIX error, and still returns 0, when ppr leaves a socket partly empty
        # (e.g. ppr:12:package with fewer than 12 ranks); so "close" maps by core: cores 0-11 are socket 0 (lscpu)
        case "$1" in
            close|full) echo "--report-bindings --map-by core --bind-to core" ;;                # socket 0 first, then 1
            spread)     echo "--report-bindings --map-by ppr:$(( ${2:-12} / 2 )):package --bind-to core" ;;   # P/2 on each
            nodes2)     echo "--report-bindings --map-by ppr:24:node --bind-to core" ;;          # 24 + 24
        esac
    }
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 LC_ALL=C
JOB=$LSB_JOBID
RUN_S=${RUN_S:-8}
TIME_LIMIT=${TIME_LIMIT:-120}

subset() {      # make gaia_subsets/gaia_nearest_<N>.csv if it is missing
    [ -f "gaia_subsets/gaia_nearest_$1.csv" ] || python3 isoeff/make_subset.py --n "$1" || echo "could not make N=$1"
}
steps_for() {   # steps so that a run takes about RUN_S seconds at ~110 ns per pair (3 to 400)
    awk -v n="$1" -v p="$2" -v s="$RUN_S" 'BEGIN { t = n * n / p * 110e-9; k = int(s / t); if (k < 3) k = 3; if (k > 400) k = 400; print k }'
}
rows_for() {    # the numeric --tile-rows for a label; blk = all own rows (one tile per step)
    if [ "$1" = blk ]; then echo $(( ($2 + $3 - 1) / $3 )); else echo "$1"; fi
}
run() {         # run <K> <scheme> <N> <P> <tile label> <placement> <repeat> [steps]
    local k=$1 s=$2 n=$3 p=$4 t=$5 pl=$6 r=$7 steps=${8:-}
    [ -n "$steps" ] || steps=$(steps_for "$n" "$p")
    local name="cache_${k}_${s}_N${n}_P${p}_t${t}_${pl}_r${r}"
    $MPIRUN $(place "$pl" "$p") -np "$p" python3 ParallelImplementation.py --data "gaia_subsets/gaia_nearest_${n}.csv" \
        --decomposition "$s" --steps "$steps" --merge-radius 0 --time-limit "$TIME_LIMIT" \
        --tile-rows "$(rows_for "$t" "$n" "$p")" --results "$RESULTS" --run-name "$name" > /dev/null
    # mpirun can return 0 although the program never ran: count the run only if its row is in the results file
    if grep -q ",${name},.*,${JOB}," "$RESULTS" 2>/dev/null; then echo "  $name ($steps steps, $(date +%H:%M:%S))"; else echo "  $name FAILED (no row)"; fi
}
copy_rows() {   # this job's rows of results.csv -> cache/out/runs_<job>.csv
    python3 - "$RESULTS" "$JOB" "cache/out/runs_${JOB}.csv" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1], low_memory=False)
d = d[(d["lsb_jobid"].astype(str).str.replace(r"\.0$", "", regex=True) == sys.argv[2]) & d["run_name"].astype(str).str.startswith("cache_")]
d.to_csv(sys.argv[3], index=False)
print(f"{len(d)} runs -> {sys.argv[3]}")
EOF
}
