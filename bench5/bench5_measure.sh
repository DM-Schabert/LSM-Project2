#!/bin/bash
# =============================================================================
#  bench5, phase 1 (two nodes, 10-25 min): every configuration of the project for ONE group of stars N.
#  Each run is short (about RUN_S seconds, at least MIN_STEPS steps); analyze.py turns its time per step into
#  the number of time steps that fit into 5 minutes. Phase 2 (bench5_final.sh) then runs the winners for real.
#
#  On the cluster, from the repository root, one job per group of stars:
#      for n in 1000 5000 20000 100000; do sed "s/^N=.*/N=$n/" bench5/bench5_measure.sh | bsub; done
#  Laptop (only checks that everything runs):  LOCAL=1 bash bench5/bench5_measure.sh
#
#  The configurations are bench5/configs.txt (serial, v1, tiles, ring, 2D, two nodes, the cache/ tile sizes).
#  Every run writes its row to results.csv, named bench5_<config>_N<N>_r<repeat>_j<job>; at the end the job
#  copies its rows to bench5/out/runs_<job>.csv.
# =============================================================================
#BSUB -J bench5
#BSUB -q hpcintro
#BSUB -n 48
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 00:30
#BSUB -o bench5/logs/bench5_%J.out
#BSUB -e bench5/logs/bench5_%J.err

LOCAL=${LOCAL:-0}
N=5000
REPS=${REPS:-}                  # repeats; empty: 2, and 1 for N >= 50 000 (a step takes tens of seconds there)
RUN_S=${RUN_S:-}                # seconds per run; empty: 15 for N <= 5 000, 20 above
SKIP=${SKIP:-}                  # labels to leave out, e.g. "serial v1-24"
if [ "$LOCAL" = 1 ]; then N=1000; REPS=1; RUN_S=2; fi
[ -n "$REPS" ] || { REPS=2; [ "$N" -ge 50000 ] && REPS=1; }
[ -n "$RUN_S" ] || { RUN_S=15; [ "$N" -gt 5000 ] && RUN_S=20; }
MIN_STEPS=${MIN_STEPS:-$([ "$N" -ge 50000 ] && echo 2 || echo 3)}
cd "${LS_SUBCWD:-$PWD}" || exit 1
source bench5/common.sh

echo "=== bench5 phase 1, job $JOB, N = $N, $(date), hosts: $(echo $LSB_HOSTS | tr ' ' '\n' | sort -u | tr '\n' ' ')"
subset "$N"
[ -f "gaia_subsets/gaia_nearest_${N}.csv" ] || exit 1

for r in $(seq 1 "$REPS"); do                    # repeats outermost: a drift of the nodes hits every configuration alike
    echo "--- repeat $r ($(date +%H:%M:%S))"
    while IFS= read -r c; do
        label=${c%%|*}
        case " $SKIP " in *" $label "*) continue ;; esac
        p=$(echo "$c" | cut -d'|' -f2)
        if [ "$label" = serial ] && [ "$N" -ge 50000 ]; then echo "  serial left out at N = $N (one step would take ~$(( N / 1000 * N / 1000 / 10 )) s)"; continue; fi
        steps=$(steps_for "$N" "$(ranks "$p")" "$RUN_S")
        run_config "$c" "$N" "bench5_${label}_N${N}_r${r}_j${JOB}" "$steps" 280
    done < <(configs)
done

copy_rows
echo "=== finished $(date). Copy bench5/out and bench5/logs off the cluster, then: python3 bench5/analyze.py"
