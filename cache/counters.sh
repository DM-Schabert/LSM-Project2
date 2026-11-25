#!/bin/bash
# =============================================================================
#  K5: where the bytes of the force kernel go: hardware counters (one node, ~15 min)
#
#  On the cluster, from the repository root:   bsub < cache/counters.sh
#  Laptop (only checks that everything runs):  LOCAL=1 bash cache/counters.sh
#  Analysis on the laptop:                     python3 cache/counters_analyze.py
#
#  The kernel alone (cache/kernel_bench.py), on 1, 12 (socket 0) and 24 ranks at once, with
#    tile sweep:  5 000 columns, 209 own rows, tiles of 2, 8, 52 (automatic, 20 MB) and 209 rows (82 MB)
#    shape sweep: the same tile size with different shapes, 20 MB (1 000 to 50 000 columns) and 1 MB
#  and counts the bytes that come from RAM (and from L3, if likwid is there). The job finds out what this
#  node allows and logs it:
#    likwid       likwid-perfctr, groups MEM (RAM data volume) and L3 (L2 <-> L3 data volume)
#    perf-uncore  perf stat -a on the memory controllers (uncore_imc cas_count): RAM bytes read and written
#    perf-llc     perf stat per rank: last-level cache misses x 64 B, a lower bound of the RAM traffic
#    none         times only (the shape sweep still answers part of the question)
#  A run with --seconds 0 (start-up only) per rank count and group is subtracted, so the counts are the kernel's.
#  Settings: METHOD (auto), SECS (12 s per run), COUNTER_GROUPS for likwid ("MEM L3").
# =============================================================================
#BSUB -J cache_counters
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 00:30
#BSUB -o cache/logs/counters_%J.out
#BSUB -e cache/logs/counters_%J.err

LOCAL=${LOCAL:-0}
METHOD=${METHOD:-auto}
SECS=${SECS:-12}
COUNTER_GROUPS=${COUNTER_GROUPS:-"MEM L3"}      # likwid groups (not GROUPS: bash keeps that name for itself)
P_LIST="1 12 24"
TILES="2 8 52 209"                              # rows per tile at 5 000 columns: 0.8, 3.2, 20.5, 82 MB
SHAPES_20MB="1000:262 5000:52 20000:13 50000:5" # columns:rows, all ~260 000 pairs per tile
SHAPES_1MB="1000:13 2000:6 5000:2"              # all ~13 000 pairs per tile
if [ "$LOCAL" = 1 ]; then METHOD=none; SECS=1; P_LIST="1 2"; TILES="2 52"; SHAPES_20MB="1000:262 5000:52"; SHAPES_1MB="1000:13"; fi
cd "${LS_SUBCWD:-$PWD}" || exit 1
source cache/common.sh
OUTDIR=cache/out/counters_${JOB}
BENCH=cache/out/kbench_${JOB}.csv
mkdir -p "$OUTDIR"

echo "=== counters, job $JOB, $(date), host $(hostname)"
echo "--- what this node allows"
lscpu 2>/dev/null | grep -i -E 'model name|socket|L2 cache|L3 cache'
echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo '?')"
command -v perf >/dev/null && perf --version || echo "no perf"
echo "event sources: $(ls /sys/bus/event_source/devices 2>/dev/null | tr '\n' ' ')"
module avail likwid 2>&1 | grep -i likwid || echo "no likwid module"

UNCORE="uncore_imc/cas_count_read/,uncore_imc/cas_count_write/"
CORE="cycles,instructions,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses"
if [ "$METHOD" = auto ]; then
    METHOD=none
    if module load likwid >/dev/null 2>&1 && command -v likwid-perfctr >/dev/null && likwid-perfctr -c 0 -g MEM sleep 0.2 >/dev/null 2>&1; then
        METHOD=likwid
    elif command -v perf >/dev/null && perf stat -a -x, -e "$UNCORE" sleep 0.2 2>&1 | grep -q cas_count && ! perf stat -a -x, -e "$UNCORE" sleep 0.2 2>&1 | grep -q 'not supported\|not counted'; then
        METHOD=perf-uncore
    elif command -v perf >/dev/null && perf stat -x, -e LLC-load-misses true 2>&1 | grep -q LLC-load-misses && ! perf stat -x, -e LLC-load-misses true 2>&1 | grep -q 'not supported\|not counted'; then
        METHOD=perf-llc
    fi
fi
[ "$METHOD" = likwid ] && module load likwid >/dev/null 2>&1
echo "--- method: $METHOD"
echo "$METHOD" > "$OUTDIR/method.txt"
[ "$METHOD" = likwid ] || COUNTER_GROUPS="$METHOD"

bench() {   # bench <label> <ranks> <columns> <rows> <tile rows> <seconds>
    local label=$1 p=$2 g np cores
    np=$p; [ "$LOCAL" = 1 ] && [ "$p" -gt 2 ] && np=2
    cores="0-$(( np - 1 ))"
    for g in $COUNTER_GROUPS; do
        local cmd=(python3 cache/kernel_bench.py --label "$label" --tag "$g" --cols "$3" --rows "$4" --tile-rows "$5" --seconds "$6" --out "$BENCH" --job "$JOB")
        case "$METHOD" in
            likwid)      likwid-perfctr -c "$cores" -g "$g" -O -o "$OUTDIR/${label}__${g}.csv" $MPIRUN $(place close "$np") -np "$np" "${cmd[@]}" < /dev/null > "$OUTDIR/${label}__${g}.log" 2>>"$OUTDIR/errors.txt" ;;
            perf-uncore) perf stat -a -x, -o "$OUTDIR/${label}__${g}.txt" -e "$UNCORE,$CORE" -- $MPIRUN $(place close "$np") -np "$np" "${cmd[@]}" < /dev/null 2>>"$OUTDIR/errors.txt" ;;
            perf-llc)    PERF_EVENTS="$CORE" $MPIRUN $(place close "$np") -x PERF_EVENTS -np "$np" cache/perf_rank.sh "$OUTDIR/${label}__${g}" "${cmd[@]}" < /dev/null 2>>"$OUTDIR/errors.txt" ;;
            *)           $MPIRUN $(place close "$np") -np "$np" "${cmd[@]}" < /dev/null ;;
        esac
    done
}

echo "--- baselines: start-up only ($(date +%H:%M:%S))"
for p in $P_LIST; do bench "base_P$p" "$p" 5000 209 2 0; done
echo "--- tile sweep: 5 000 columns ($(date +%H:%M:%S))"
for p in $P_LIST; do for k in $TILES; do bench "tile_P${p}_c5000_k$k" "$p" 5000 209 "$k" "$SECS"; done; done
echo "--- shape sweep: the same tile size, other shapes ($(date +%H:%M:%S))"
for p in 1 24; do for s in $SHAPES_20MB; do bench "shape_P${p}_c${s%%:*}_k${s##*:}" "$p" "${s%%:*}" $(( 4 * ${s##*:} )) "${s##*:}" "$SECS"; done; done
for s in $SHAPES_1MB; do bench "shape_P24_c${s%%:*}_k${s##*:}" 24 "${s%%:*}" $(( 4 * ${s##*:} )) "${s##*:}" "$SECS"; done

echo "=== finished $(date). Copy cache/out and cache/logs off the cluster, then: python3 cache/counters_analyze.py"
