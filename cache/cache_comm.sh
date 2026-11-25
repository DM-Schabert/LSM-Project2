#!/bin/bash
# =============================================================================
#  Cache and memory, two nodes (~5 min): K4 communication while the other cores load the memory
#
#  On the cluster, from the repository root:   bsub < cache/cache_comm.sh
#  Laptop (only checks that everything runs):  LOCAL=1 bash cache/cache_comm.sh
#
#  intra  24 ranks on the first node: rank 0 and rank 1 (two cores of socket 0) ping-pong through shared
#         memory, the other 22 ranks run the load
#  inter  48 ranks on two nodes: rank 0 and rank 24 (core 0 of each node) ping-pong over InfiniBand,
#         the other 46 ranks run the load
#  Loads: idle, small (force kernel, ~1.6 MB tiles), big (force kernel, ~20 MB tiles), stream (NumPy triad).
#  Output: cache/out/comm_load_<job>.csv (see cache/comm_load.py).
# =============================================================================
#BSUB -J cache_comm
#BSUB -q hpcintro
#BSUB -n 48
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 00:20
#BSUB -o cache/logs/cache_comm_%J.out
#BSUB -e cache/logs/cache_comm_%J.err

LOCAL=${LOCAL:-0}
NP_INTRA=24; NP_INTER=48; PARTNER_INTER=24; EXTRA=""
if [ "$LOCAL" = 1 ]; then NP_INTRA=3; NP_INTER=4; PARTNER_INTER=2; EXTRA="--sizes 8 65536 --seconds 0.05 --warmup 0.05 --load-n 1000 --stream-mb 4"; fi
cd "${LS_SUBCWD:-$PWD}" || exit 1
source cache/common.sh
OUT=cache/out/comm_load_${JOB}.csv

echo "=== cache_comm, job $JOB, $(date), hosts: $(echo $LSB_HOSTS | tr ' ' '\n' | sort -u | tr '\n' ' ')"
echo "--- K4 intra: one node, shared memory ($(date +%H:%M:%S))"
$MPIRUN $(place full "$NP_INTRA") -np "$NP_INTRA" python3 cache/comm_load.py --out "$OUT" --placement intra --partner 1 $EXTRA
echo "--- K4 inter: two nodes, InfiniBand ($(date +%H:%M:%S))"
$MPIRUN $(place nodes2 "$NP_INTER") -np "$NP_INTER" python3 cache/comm_load.py --out "$OUT" --placement inter --partner "$PARTNER_INTER" $EXTRA
echo "=== finished $(date). Copy cache/out and cache/logs off the cluster, then: python3 cache/analyze.py"
