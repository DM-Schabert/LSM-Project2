#!/bin/bash
# =============================================================================
#  run_all.sh: the whole Amdahl campaign in one go.
#
#  Submits one strong-scaling sweep per N (amdahl_batch.sh: serial baseline plus MPI
#  at every P, each sweep on its own fully reserved node) and one small analysis job
#  that starts by itself once all sweeps have ended. Afterwards read
#  amdahl/out/<campaign>/report.txt and look at the plots in the same folder.
#
#  From the repository root on the HPC login node:
#      bash amdahl/run_all.sh                    # submit everything
#      DRY_RUN=1 bash amdahl/run_all.sh          # only write and show the job scripts
#      LABEL=v2-numba bash amdahl/run_all.sh     # the same campaign for a new code version
#
#  The DTU cluster does not pass your shell's environment on to jobs. So, like submit.sh,
#  this script writes each sweep's settings as export lines into a copy of amdahl_batch.sh
#  and pipes that copy to bsub. The copies are kept in amdahl/logs/<campaign>/, so you can
#  always see exactly what ran.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LABEL=${LABEL:-v1-numpy}                # name of the code version
LABEL=${LABEL//_/-}                     # '_' separates the parts of run_name
P_LIST=${P_LIST:-"1 2 4 8 12 16 24"}
DRY_RUN=${DRY_RUN:-0}
CAMPAIGN=${CAMPAIGN:-${LABEL}_$(date +%Y%m%d_%H%M)}

# One line per sweep. Fixed work per N: as many steps as make the serial run last about
# 30-90 s (the efficiency of the step loop does not depend on the number of steps), and our
# 10 000 steps wherever that fits. Times measured on Xeon E5-2650 v4 nodes in November.
#            N    steps  repeats  walltime [min]    serial run    whole job
SWEEPS=${SWEEPS:-"
            50    20000     3        15             #   ~31 s       ~7 min
           100    10000     3        15             #   ~31 s       ~6 min
           300    10000     2        30             #  ~225 s      ~25 min (estimated)
          1000      300     3        15             #   ~42 s       ~8 min
          5000       30     3        30             #   ~86 s      ~15 min
"}

BATCH=amdahl/amdahl_batch.sh
MARKER="# ----------------------------------------------------------------- settings"
LOGDIR=amdahl/logs/$CAMPAIGN
grep -qxF "$MARKER" "$BATCH" || { echo "Settings marker not found in $BATCH" >&2; exit 1; }
if [ "$DRY_RUN" != 1 ] && ! command -v bsub >/dev/null 2>&1; then
    echo "bsub not found: run this on the HPC login node (or use DRY_RUN=1)." >&2; exit 1
fi
mkdir -p "$LOGDIR"

submit() {   # submit <job script> [bsub options]: prints the LSF job ID
    local script=$1 out id; shift
    if [ "$DRY_RUN" = 1 ]; then
        echo "dry-$(basename "$script" .sh)"; return
    fi
    out=$(bsub "$@" < "$script") || { echo "$out" >&2; return 1; }
    id=$(echo "$out" | sed -n 's/^Job <\([0-9][0-9]*\)>.*/\1/p')
    [ -n "$id" ] || { echo "Could not read the job ID from: $out" >&2; return 1; }
    echo "$id"
}

echo "Campaign $CAMPAIGN   label $LABEL   P: $P_LIST"
ids=()
while IFS= read -r line; do
    line=${line%%#*}
    read -r n steps reps wall _ <<< "$line" || true
    [ -n "${n:-}" ] || continue
    data=gaia_subsets/gaia_nearest_$n.csv
    if [ ! -f "$data" ]; then
        echo "  skip N=$n: $data not found (make it with GaiaSubsetSelection.py)"; continue
    fi
    # copy of amdahl_batch.sh with this sweep's settings as export lines before the settings block
    # (after the #BSUB lines: LSF ignores #BSUB lines after the first command)
    exports=$(printf 'export LABEL=%q DATA=%q STEPS=%q REPEATS=%q WALLTIME_MIN=%q P_LIST=%q' \
              "$LABEL" "$data" "$steps" "$reps" "$wall" "$P_LIST")
    job=$LOGDIR/sweep_N$n.sh
    inserted=0
    while IFS= read -r l; do
        if [ "$inserted" = 0 ] && [ "$l" = "$MARKER" ]; then
            printf '# settings from run_all.sh (campaign %s)\n%s\n\n' "$CAMPAIGN" "$exports"; inserted=1
        fi
        printf '%s\n' "$l"
    done < "$BATCH" > "$job"
    # -W on the command line overrides the #BSUB -W line; WALLTIME_MIN above keeps the runs' own limit below it
    id=$(submit "$job" -J "amdahl_N$n" -W "$wall")
    ids+=("$id")
    printf '  N = %-5s %6s steps, %s repeats, %2s min walltime   job %s\n' "$n" "$steps" "$reps" "$wall" "$id"
    echo "N=$n steps=$steps repeats=$reps walltime=$wall job=$id" >> "$LOGDIR/jobs.txt"
done <<< "$SWEEPS"

[ ${#ids[@]} -gt 0 ] || { echo "Nothing submitted." >&2; exit 1; }

# The analysis waits until every sweep has ended (also if one failed) and analyses exactly these jobs
dep=""
for id in "${ids[@]}"; do dep+="${dep:+ && }ended($id)"; done
OUT=amdahl/out/$CAMPAIGN
cat > "$LOGDIR/analysis.sh" <<EOF
#!/bin/bash
#BSUB -J amdahl_analysis
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 00:15
#BSUB -o amdahl/logs/analysis_%J.out
#BSUB -e amdahl/logs/analysis_%J.err
# Campaign $CAMPAIGN: starts when all its sweeps have ended (bsub -w), then analyses exactly those jobs.
cd "\${LS_SUBCWD:-\$PWD}" || exit 1
module load pandas/2.3.0-python-3.12.11
module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
export MPLBACKEND=Agg
mkdir -p $OUT
python3 amdahl/amdahl_analysis.py --label $LABEL --job ${ids[*]} --out $OUT 2>&1 | tee $OUT/report.txt
EOF
aid=$(submit "$LOGDIR/analysis.sh" -w "$dep")
echo "analysis job=$aid waits for: $dep" >> "$LOGDIR/jobs.txt"
echo "  analysis                                              job $aid  (starts when all sweeps have ended)"

echo
if [ "$DRY_RUN" = 1 ]; then
    echo "Dry run: nothing submitted. Job scripts are in $LOGDIR/"
else
    echo "Watch:    bjobs            (PEND = waiting for a node, RUN = running)"
    echo "Results:  $OUT/report.txt and the plots in $OUT/"
    echo "Logs:     amdahl/logs/amdahl_<jobid>.out, job list in $LOGDIR/jobs.txt"
fi
