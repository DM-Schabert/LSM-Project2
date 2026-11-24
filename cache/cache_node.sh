#!/bin/bash
# =============================================================================
#  Cache and memory, one node, part 1 (~20 min): K0 bandwidth; K1 and K2 at N = 5 000
#
#  On the cluster, from the repository root:   bsub < cache/cache_node.sh
#  Laptop (only checks that everything runs):  LOCAL=1 bash cache/cache_node.sh
#
#  K0  memory bandwidth of NumPy loops on 1 ... 24 cores, ranks on one socket (close) or on both (spread)
#  K1  v1 with --tile-rows 1, 2, 4, 8, 16, 0 (automatic, ~2^18 pairs) and the whole block: time per pair against
#      tile size at P = 1, 12 (one socket) and 24 (the node)
#  K2  the same at P = 12 with the ranks spread over both sockets (6 + 6); K1's P = 12 runs are the "close" half
#  Part 2, N = 20 000: cache/cache_node20k.sh.
#
#  Every run writes its row to results.csv as always, named cache_<K>_<scheme>_N<N>_P<P>_t<tiles>_<placement>_r<repeat>;
#  at the end the job copies its rows to cache/out/runs_<job>.csv. Analysis on the laptop: python3 cache/analyze.py
#  Settings: TILES, P_LIST, REPS (3), RUN_S (seconds per run, 8), K0_CLOSE, K0_SPREAD
# =============================================================================
#BSUB -J cache_node
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:30
#BSUB -o cache/logs/cache_node_%J.out
#BSUB -e cache/logs/cache_node_%J.err

LOCAL=${LOCAL:-0}
N=5000
TILES=${TILES:-"1 2 4 8 16 0 blk"}
P_LIST=${P_LIST:-"1 12 24"}
K2_P=12
K0_CLOSE=${K0_CLOSE:-"1 2 4 6 8 12"}
K0_SPREAD=${K0_SPREAD:-"2 4 8 12"}
K0_MB=64
REPS=${REPS:-3}
if [ "$LOCAL" = 1 ]; then N=1000; TILES="1 4 0 blk"; P_LIST="1 2"; K2_P=2; K0_CLOSE="1 2"; K0_SPREAD="2"; K0_MB=8; REPS=1; RUN_S=1; fi
cd "${LS_SUBCWD:-$PWD}" || exit 1
source cache/common.sh

echo "=== cache_node, job $JOB, $(date), host $(hostname)"
lscpu 2>/dev/null | grep -i -E 'model name|socket|core|cache|numa'
free -g 2>/dev/null | head -2
subset "$N"

echo "--- K0: memory bandwidth ($(date +%H:%M:%S))"
OUT=cache/out/bandwidth_${JOB}.csv
for p in $K0_CLOSE;  do $MPIRUN $(place close)  -np "$p" python3 cache/bandwidth.py --placement close  --mb "$K0_MB" --out "$OUT"; done
for p in $K0_SPREAD; do $MPIRUN $(place spread) -np "$p" python3 cache/bandwidth.py --placement spread --mb "$K0_MB" --out "$OUT"; done
[ "$LOCAL" = 1 ] || $MPIRUN $(place full) -np 24 python3 cache/bandwidth.py --placement full --mb "$K0_MB" --out "$OUT"

echo "--- K1 and K2 at N = $N ($(date +%H:%M:%S))"
for r in $(seq 1 "$REPS"); do                    # repeats outermost: a drift of the node hits every setting alike
    for p in $P_LIST; do
        pl=close; [ "$p" -gt 12 ] && pl=full
        for t in $TILES; do run K1 allgather "$N" "$p" "$t" "$pl" "$r"; done
    done
    for t in $TILES; do run K2 allgather "$N" "$K2_P" "$t" spread "$r"; done
done

copy_rows
echo "=== finished $(date). Copy cache/out and cache/logs off the cluster, then: python3 cache/analyze.py"
