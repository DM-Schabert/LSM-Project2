#!/bin/bash
# =============================================================================
#  run_all.sh: the whole Gustafson (weak-scaling) campaign in one go.
#
#  Submits one weak-scaling sweep per line of SWEEPS (gustafson_batch.sh: for every P the
#  MPI run and the serial run on the scaled problem N(P), each sweep on its own fully
#  reserved node) and one small analysis job that starts by itself once all sweeps have
#  ended. Afterwards read gustafson/out/<campaign>/report.txt and look at the plots there.
#
#  From the repository root on the HPC login node:
#      bash gustafson/run_all.sh                    # submit everything
#      DRY_RUN=1 bash gustafson/run_all.sh          # only write and show the job scripts and sizes
#      LABEL=v2-numba COST_A_US=.. COST_B_NS=.. bash gustafson/run_all.sh   # a new code version
#
#  The DTU cluster does not pass your shell's environment on to jobs. So, like amdahl/run_all.sh,
#  this script writes each sweep's settings as export lines into a copy of gustafson_batch.sh
#  and pipes that copy to bsub. The copies are kept in gustafson/logs/<campaign>/.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LABEL=${LABEL:-v1-numpy}                # name of the code version
LABEL=${LABEL//_/-}                     # '_' separates the parts of run_name
P_LIST=${P_LIST:-"1 2 4 8 12 16 24"}
DRY_RUN=${DRY_RUN:-0}
CAMPAIGN=${CAMPAIGN:-${LABEL}_$(date +%Y%m%d_%H%M)}
# Serial cost per step a*N + b*N^2 of the code version, for RULE=time. Empty: the defaults in
# gustafson/plan.py (fitted to the v1-numpy Amdahl campaign). For a new version, fit them with
#     python3 gustafson/plan.py --fit amdahl/out/<its campaign>/by_N_<label>.csv
COST_A_US=${COST_A_US:-}
COST_B_NS=${COST_B_NS:-}

# The experiment list: one line per sweep, each on its own node.
#  rule  time:  N(P) such that every rank keeps the serial step time of the base problem (Gustafson's
#               assumption: the parallel time per rank stays the same). The main experiment.
#        pairs: N(P) = N0*sqrt(P), every rank keeps the pair interactions of the base problem.
#               For comparison: NumPy's per-row overhead makes such a share cheaper than the base
#               problem, so the weak efficiency can exceed 100 % without any parallel effect.
#  steps: of every MPI run and of the serial base run, chosen for ~20 s per run (Xeon E5-2650 v4).
#         The serial runs on the scaled problems take max(10, ceil(steps/P)) steps, also ~20 s.
#  Estimated with gustafson/plan.py: every sweep ~15-16 min, so 30 min walltime leaves margin.
#              rule    N0   steps  repeats  walltime [min]    N at P = 24
SWEEPS=${SWEEPS:-"
               time    50   14000     3        30            #    461
               time   100    6000     3        30            #    761
               time   300    1200     3        30            #  1 810
               time  1000     150     3        30            #  5 279
               pairs  100    6000     3        30            #    490
               pairs 1000     150     3        30            #  4 899
"}

BATCH=gustafson/gustafson_batch.sh
MARKER="# ----------------------------------------------------------------- settings"
LOGDIR=gustafson/logs/$CAMPAIGN
grep -qxF "$MARKER" "$BATCH" || { echo "Settings marker not found in $BATCH" >&2; exit 1; }
if [ "$DRY_RUN" != 1 ] && ! command -v bsub >/dev/null 2>&1; then
    echo "bsub not found: run this on the HPC login node (or use DRY_RUN=1)." >&2; exit 1
fi
cost_args=()
[ -n "$COST_A_US" ] && cost_args+=(--a-us "$COST_A_US")
[ -n "$COST_B_NS" ] && cost_args+=(--b-ns "$COST_B_NS")

# ---- 1. sizes of every sweep, and the data files they need
declare -a lines=()
all_n=""
while IFS= read -r line; do
    line=${line%%#*}
    read -r rule n0 steps reps wall _ <<< "$line" || true
    [ -n "${rule:-}" ] || continue
    sizes=$(python3 gustafson/plan.py --n0 "$n0" --rule "$rule" --p $P_LIST --sizes ${cost_args[@]+"${cost_args[@]}"})
    lines+=("$rule $n0 $steps $reps $wall|$sizes")
    all_n+=" $sizes"
done <<< "$SWEEPS"
[ ${#lines[@]} -gt 0 ] || { echo "No sweeps in SWEEPS." >&2; exit 1; }

missing=$(for n in $all_n; do [ -f "gaia_subsets/gaia_nearest_$n.csv" ] || echo "$n"; done | sort -n | uniq | tr '\n' ' ')
if [ -n "$missing" ]; then
    echo "These subsets are missing in gaia_subsets/ (nearest-N, nested like the existing ones):"
    echo "    $missing"
    echo "Make them where gaia_cleaned.csv is (and copy them to gaia_subsets/ here if that is elsewhere):"
    echo "    python3 GaiaSubsetSelection.py --n $missing--save --no-plot"
    if [ "$DRY_RUN" != 1 ]; then exit 1; fi
    echo "(dry run: carrying on)"
    echo
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

# ---- 2. one job per sweep
echo "Campaign $CAMPAIGN   label $LABEL   P: $P_LIST"
ids=()
for entry in "${lines[@]}"; do
    read -r rule n0 steps reps wall <<< "${entry%%|*}"
    sizes=${entry#*|}
    # copy of gustafson_batch.sh with this sweep's settings as export lines before the settings block
    # (after the #BSUB lines: LSF ignores #BSUB lines after the first command)
    exports=$(printf 'export LABEL=%q RULE=%q N0=%q N_LIST=%q STEPS=%q REPEATS=%q WALLTIME_MIN=%q P_LIST=%q' \
              "$LABEL" "$rule" "$n0" "$sizes" "$steps" "$reps" "$wall" "$P_LIST")
    [ -n "$COST_A_US" ] && exports+=$(printf ' COST_A_US=%q' "$COST_A_US")
    [ -n "$COST_B_NS" ] && exports+=$(printf ' COST_B_NS=%q' "$COST_B_NS")
    job=$LOGDIR/sweep_${rule}_n0_$n0.sh
    inserted=0
    while IFS= read -r l; do
        if [ "$inserted" = 0 ] && [ "$l" = "$MARKER" ]; then
            printf '# settings from run_all.sh (campaign %s)\n%s\n\n' "$CAMPAIGN" "$exports"; inserted=1
        fi
        printf '%s\n' "$l"
    done < "$BATCH" > "$job"
    # -W on the command line overrides the #BSUB -W line; WALLTIME_MIN keeps the runs' own limit below it
    id=$(submit "$job" -J "gustafson_${rule}_n0_$n0" -W "$wall")
    ids+=("$id")
    printf '  %-5s N0 = %-5s %6s steps, %s repeats, %2s min   N = %-34s job %s\n' \
           "$rule" "$n0" "$steps" "$reps" "$wall" "$sizes" "$id"
    echo "rule=$rule N0=$n0 N=\"$sizes\" steps=$steps repeats=$reps walltime=$wall job=$id" >> "$LOGDIR/jobs.txt"
done

# ---- 3. the analysis waits until every sweep has ended (also if one failed) and analyses exactly these jobs
dep=""
for id in "${ids[@]}"; do dep+="${dep:+ && }ended($id)"; done
OUT=gustafson/out/$CAMPAIGN
cat > "$LOGDIR/analysis.sh" <<EOF
#!/bin/bash
#BSUB -J gustafson_analysis
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 00:15
#BSUB -o gustafson/logs/analysis_%J.out
#BSUB -e gustafson/logs/analysis_%J.err
# Campaign $CAMPAIGN: starts when all its sweeps have ended (bsub -w), then analyses exactly those jobs.
cd "\${LS_SUBCWD:-\$PWD}" || exit 1
module load pandas/2.3.0-python-3.12.11
module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
export MPLBACKEND=Agg
mkdir -p $OUT
python3 gustafson/gustafson_analysis.py --label $LABEL --job ${ids[*]} --out $OUT 2>&1 | tee $OUT/report.txt
EOF
aid=$(submit "$LOGDIR/analysis.sh" -w "$dep")
echo "analysis job=$aid waits for: $dep" >> "$LOGDIR/jobs.txt"
echo "  analysis                                                                  job $aid  (starts when all sweeps have ended)"

echo
if [ "$DRY_RUN" = 1 ]; then
    echo "Dry run: nothing submitted. Job scripts are in $LOGDIR/"
    echo "Time per sweep: python3 gustafson/plan.py --n0 <N0> --rule <rule> --steps <steps> --walltime 30"
else
    echo "Watch:    bjobs            (PEND = waiting for a node, RUN = running)"
    echo "Results:  $OUT/report.txt and the plots in $OUT/"
    echo "Logs:     gustafson/logs/gustafson_<jobid>.out, job list in $LOGDIR/jobs.txt"
fi
