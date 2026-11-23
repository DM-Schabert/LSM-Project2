#!/bin/bash
# =============================================================================
#  Memory in depth: what fills the memory of a rank, and which limit comes first?  (two nodes, ~15 min)
#
#  On the cluster, from the repository root:  bsub < memory/memory_batch.sh
#  Results page afterwards (on the laptop):   python3 memory/make_page.py
#
#  M3  memory against P: v1, N = 1 000, P = 1 ... 24 on one node, 32 and 48 on two
#  M1  memory against N: v1, ring, 2D at P = 24, N = 1 000 ... all 589 993 stars, kernel on one tile
#  M2  memory against tile size: v1, N = 20 000, P = 24, the whole block per step, --tile-rows 0 ... 417
#      (417 = half the block; the largest stays below 1 GB per rank, the reservation of this job)
#  Output: memory/out/memory_<jobid>.csv; the log has the node's memory, the LSF limits and, at the
#  end, LSF's "Max Memory" of the whole job (a cross-check of the sum over the ranks).
# =============================================================================
#BSUB -J memory
#BSUB -q hpcintro
#BSUB -n 48
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 00:30
#BSUB -o memory/logs/memory_%J.out
#BSUB -e memory/logs/memory_%J.err

M1_N=${M1_N:-"1000 5000 20000 50000 100000 200000 500000 589993"}
M1_SCHEMES=${M1_SCHEMES:-"allgather ring 2d"}
M2_TILES=${M2_TILES:-"0 32 104 208 417"}
M3_P=${M3_P:-"1 2 4 8 12 16 24 32 48"}
OPTS="--report-bindings --map-by core --bind-to core"   # P <= 24 on the first node, as in all our runs

cd "${LS_SUBCWD:-$PWD}" || exit 1
[ -f memory/memory_probe.py ] || { echo "Start this from the repository root"; exit 1; }
mkdir -p memory/logs memory/out
module load pandas/2.3.0-python-3.12.11
module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 LC_ALL=C
OUT=memory/out/memory_${LSB_JOBID}.csv

echo "=== memory, job $LSB_JOBID, $(date), hosts: $(echo $LSB_HOSTS | tr ' ' '\n' | sort -u | tr '\n' ' ')"
echo "--- node memory (free -m) and limits"
free -m
ulimit -a | grep -i -E 'memory|virtual|data'
lshosts -w "$(hostname)" 2>/dev/null
bjobs -l "$LSB_JOBID" 2>/dev/null | grep -i -E 'rusage|mem|limit' | head -10

echo "--- subsets (made from gaia_cleaned.csv where missing)"
for n in $M1_N; do
    [ -f "gaia_subsets/gaia_nearest_${n}.csv" ] || python3 isoeff/make_subset.py --n "$n" || echo "could not make N=$n"
done

probe() {   # probe <P> <experiment> <scheme> <N> <tile rows> <kernel>
    local f="gaia_subsets/gaia_nearest_${4}.csv"
    if [ ! -f "$f" ]; then echo "--- skipped $2 $3 N=$4: no $f"; return; fi
    mpirun $OPTS -np "$1" python3 memory/memory_probe.py --out "$OUT" --job "$LSB_JOBID" \
        --experiment "$2" --scheme "$3" --data "$f" --tile-rows "$5" --kernel "$6" 2>&1 | grep -v -E 'MCW rank|bound to'
}

echo "--- M3: against P ($(date +%H:%M:%S))"
for p in $M3_P; do probe "$p" M3 allgather 1000 0 tile; done
echo "--- M1: against N ($(date +%H:%M:%S))"
for n in $M1_N; do for s in $M1_SCHEMES; do probe 24 M1 "$s" "$n" 0 tile; done; done
echo "--- M2: against tile size ($(date +%H:%M:%S))"
for k in $M2_TILES; do probe 24 M2 allgather 20000 "$k" block; done

echo "=== finished $(date). Copy $OUT and memory/logs/memory_${LSB_JOBID}.* (LSF adds 'Max Memory' at the end of the .out) off the cluster, then: python3 memory/make_page.py"
