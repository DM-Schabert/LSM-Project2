#!/bin/bash
# =============================================================================
#  submit.sh: submit batch.sh with settings, without relying on LSF to pass
#  your shell's environment on to the job (the DTU cluster doesn't).
#
#  The settings are written into a copy of batch.sh as "export" lines, right
#  after the #BSUB lines, and that copy is piped to bsub. batch.sh itself is
#  not changed.
#
#  Usage:   ./submit.sh [N] [NAME=value ...] [-- bsub options]
#
#      ./submit.sh 2828                                # N=2828 -> gaia_subsets/gaia_nearest_2828.csv
#      ./submit.sh 2828 STEPS=100 MERGE_RADIUS=0
#      ./submit.sh N=1000 MODE=validate STEPS=500
#      ./submit.sh 2828 NP=2 STEPS=100 -- -n 24 -J weak_2828      # 2 ranks on a fully reserved node
#      ./submit.sh 5000 NP=8 MAP_OPTS="--map-by ppr:4:package --bind-to core" -- -n 24
#      ./submit.sh 300 --dry-run                       # print the job script instead of submitting
#
#  A bare number is N. NAME=value sets any setting from batch.sh (MODE, N, SUBSET,
#  DATA, STEPS, DT, SOFTENING, MERGE_RADIUS, WALLTIME_MIN, NP, MAP_OPTS, RESULTS).
#  Everything after "--" goes to bsub (e.g. -n, -J, -W). If you pass -W, also
#  pass WALLTIME_MIN so the program's own time limit stays below the walltime.
#  First time:  chmod +x submit.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

JOB_SCRIPT=batch.sh
MARKER="# ----------------------------------------------------------------- settings"
KNOWN="MODE N SUBSET DATA STEPS DT SOFTENING MERGE_RADIUS WALLTIME_MIN NP MAP_OPTS RESULTS"

exports=()
bsub_opts=()
dry_run=false

while [ $# -gt 0 ]; do
    case "$1" in
        --)         shift; bsub_opts=("$@"); break ;;
        --dry-run)  dry_run=true; shift ;;
        *=*)
            name="${1%%=*}"
            if [[ " $KNOWN " != *" $name "* ]]; then
                echo "Unknown setting '$name'. Known: $KNOWN" >&2; exit 1
            fi
            exports+=("export ${name}=$(printf '%q' "${1#*=}")"); shift ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                exports+=("export N=$1"); shift
            else
                echo "Don't understand '$1'. Use a number (N), NAME=value, or put bsub options after --" >&2
                exit 1
            fi ;;
    esac
done

if ! grep -qF "$MARKER" "$JOB_SCRIPT"; then
    echo "Could not find the settings marker in $JOB_SCRIPT" >&2; exit 1
fi

# Copy of batch.sh with the export lines inserted just before the settings block.
# (They must come after the #BSUB lines: LSF ignores #BSUB lines after the first command.)
job=$(awk -v block="$(printf '%s\n' "${exports[@]+"${exports[@]}"}")" -v marker="$MARKER" '
    $0 == marker && !done { print "# settings from submit.sh"; if (block != "") print block; print ""; done = 1 }
    { print }' "$JOB_SCRIPT")

if $dry_run; then
    echo "$job"
    echo "--- would run: bsub ${bsub_opts[*]+"${bsub_opts[*]}"}" >&2
    exit 0
fi

echo "Settings: ${exports[*]+"${exports[*]}"}" | sed 's/export //g'
echo "$job" | bsub ${bsub_opts[@]+"${bsub_opts[@]}"}
