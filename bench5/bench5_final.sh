#!/bin/bash
# =============================================================================
#  bench5, phase 2 (two nodes, ~25 min): the winner of every group of stars runs for real, 5 minutes each.
#  The program's own time limit (--time-limit 300) counts the start-up too and stops one step early rather than
#  late (it keeps 5 % in reserve), so the iterations of this run are the time steps a 5-minute job really gets.
#
#  Needs bench5/out/winners.txt (written by python3 bench5/analyze.py after phase 1): one "N label" per line.
#  On the cluster, from the repository root:   bsub < bench5/bench5_final.sh
#  Laptop (only checks that everything runs):  LOCAL=1 bash bench5/bench5_final.sh
#  Settings: ONLY="5000 20000" runs only these groups (4 groups x 5 min fit into the 30 min walltime).
# =============================================================================
#BSUB -J bench5_final
#BSUB -q hpcintro
#BSUB -n 48
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 00:30
#BSUB -o bench5/logs/bench5_final_%J.out
#BSUB -e bench5/logs/bench5_final_%J.err

LOCAL=${LOCAL:-0}
LIMIT=300
ONLY=${ONLY:-}
WINNERS=${WINNERS:-bench5/out/winners.txt}
if [ "$LOCAL" = 1 ]; then LIMIT=8; WINNERS=${WINNERS_LOCAL:-bench5/out/winners_local.txt}; fi
cd "${LS_SUBCWD:-$PWD}" || exit 1
source bench5/common.sh
[ -f "$WINNERS" ] || { echo "No $WINNERS: run python3 bench5/analyze.py after phase 1 and copy the file here"; exit 1; }

echo "=== bench5 phase 2, job $JOB, $(date), hosts: $(echo $LSB_HOSTS | tr ' ' '\n' | sort -u | tr '\n' ' ')"
while read -r n label; do
    [ -n "$n" ] && [ "${n:0:1}" != "#" ] || continue
    [ -z "$ONLY" ] || case " $ONLY " in *" $n "*) ;; *) continue ;; esac
    c=$(configs | grep "^${label}|") || { echo "  unknown configuration $label"; continue; }
    subset "$n"
    echo "--- N = $n: $label for ${LIMIT} s ($(date +%H:%M:%S))"
    run_config "$c" "$n" "bench5final_${label}_N${n}_j${JOB}" 100000000 "$LIMIT"
done < "$WINNERS"

copy_rows
echo "=== finished $(date). Copy bench5/out and bench5/logs off the cluster, then: python3 bench5/analyze.py"
