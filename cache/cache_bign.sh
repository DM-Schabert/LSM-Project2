#!/bin/bash
# =============================================================================
#  Cache and memory, two nodes (~20 min): K3 the three schemes at N = 100 000
#
#  On the cluster, from the repository root:   bsub < cache/cache_bign.sh
#  Laptop (only checks that everything runs):  LOCAL=1 bash cache/cache_bign.sh
#
#  At N = 100 000 every v1 rank holds 2.4 MB of positions, 29 MB for the 12 ranks of a socket: about its whole
#  L3. The ring and 2D hold a fraction of that, and the ring's pieces can be made small enough for the cache,
#  which v1's rows of 100 000 stars cannot. The project description expected the schemes to part here.
#  A few steps per run (one step is ~25 s at P = 48, ~50 s at P = 24), one repeat; merging off as in C2/C3.
#
#  Runs (scheme:tile rows):
#      P = 48 on two nodes:  allgather:0 allgather:1 ring:0 ring:8 2d:0 2d:1      (3 steps each)
#      P = 24 on one node:   allgather:0 ring:0 ring:4 2d:0                       (2 steps each)
#  Ring pieces with 8 (P = 48) or 4 (P = 24) rows are ~1.3 MB; 2D with 1 row is ~1.3 MB at P = 48.
#  Rows go to results.csv (cache_K3_...) and, at the end, to cache/out/runs_<job>.csv.
# =============================================================================
#BSUB -J cache_bign
#BSUB -q hpcintro
#BSUB -n 48
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 00:30
#BSUB -o cache/logs/cache_bign_%J.out
#BSUB -e cache/logs/cache_bign_%J.err

LOCAL=${LOCAL:-0}
N=${N:-100000}
RUNS_48=${RUNS_48:-"allgather:0 allgather:1 ring:0 ring:8 2d:0 2d:1"}
RUNS_24=${RUNS_24:-"allgather:0 ring:0 ring:4 2d:0"}
P_TWO=48; P_ONE=24; STEPS_TWO=${STEPS_TWO:-3}; STEPS_ONE=${STEPS_ONE:-2}
TIME_LIMIT=300
if [ "$LOCAL" = 1 ]; then N=1000; P_TWO=4; P_ONE=2; STEPS_TWO=2; STEPS_ONE=2; RUNS_48="allgather:0 ring:0 ring:8 2d:0"; RUNS_24="allgather:0 ring:4"; fi
cd "${LS_SUBCWD:-$PWD}" || exit 1
source cache/common.sh

echo "=== cache_bign, job $JOB, $(date), hosts: $(echo $LSB_HOSTS | tr ' ' '\n' | sort -u | tr '\n' ' ')"
subset "$N"
[ -f "gaia_subsets/gaia_nearest_${N}.csv" ] || exit 1

echo "--- K3: P = $P_TWO on two nodes ($(date +%H:%M:%S))"
for x in $RUNS_48; do run K3 "${x%%:*}" "$N" "$P_TWO" "${x##*:}" nodes2 1 "$STEPS_TWO"; done
echo "--- K3: P = $P_ONE on one node ($(date +%H:%M:%S))"
for x in $RUNS_24; do run K3 "${x%%:*}" "$N" "$P_ONE" "${x##*:}" full 1 "$STEPS_ONE"; done

copy_rows
echo "=== finished $(date). Copy cache/out and cache/logs off the cluster, then: python3 cache/analyze.py"
