#!/bin/bash
# =============================================================================
#  Cache and memory, one node, part 2 (~15 min): K1 and K2 at N = 20 000
#
#  On the cluster, from the repository root:   bsub < cache/cache_node20k.sh
#  Laptop (only checks that everything runs):  LOCAL=1 bash cache/cache_node20k.sh
#
#  K1  v1 with --tile-rows 1 (the row loop, 1.6 MB), 2, 4, 0 (automatic, 13 rows), 64 and the whole block
#      (834 rows at P = 24, 1.3 GB per rank; 1 667 rows at P = 12, 2.6 GB per rank: 32 GB of the node's 257 GB)
#      at P = 12 (one socket) and 24 (the node)
#  K2  tiles 1, 0, 64 at P = 12 spread over both sockets (6 + 6)
#  Part 1 (K0 and N = 5 000): cache/cache_node.sh. Settings: TILES, K2_TILES, P_LIST, REPS (3), RUN_S (8)
# =============================================================================
#BSUB -J cache_node20k
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:30
#BSUB -o cache/logs/cache_node20k_%J.out
#BSUB -e cache/logs/cache_node20k_%J.err

LOCAL=${LOCAL:-0}
N=20000
TILES=${TILES:-"1 2 4 0 64 blk"}
K2_TILES=${K2_TILES:-"1 0 64"}
P_LIST=${P_LIST:-"12 24"}
K2_P=12
REPS=${REPS:-3}
if [ "$LOCAL" = 1 ]; then N=1000; TILES="1 0 blk"; K2_TILES="1 0"; P_LIST="2"; K2_P=2; REPS=1; RUN_S=1; fi
cd "${LS_SUBCWD:-$PWD}" || exit 1
source cache/common.sh

echo "=== cache_node20k, job $JOB, $(date), host $(hostname)"
free -g 2>/dev/null | head -2
subset "$N"

echo "--- K1 and K2 at N = $N ($(date +%H:%M:%S))"
for r in $(seq 1 "$REPS"); do
    for p in $P_LIST; do
        pl=close; [ "$p" -gt 12 ] && pl=full
        for t in $TILES; do run K1 allgather "$N" "$p" "$t" "$pl" "$r"; done
    done
    for t in $K2_TILES; do run K2 allgather "$N" "$K2_P" "$t" spread "$r"; done
done

copy_rows
echo "=== finished $(date). Copy cache/out and cache/logs off the cluster, then: python3 cache/analyze.py"
