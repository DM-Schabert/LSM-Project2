#!/bin/bash
# =============================================================================
#  baseline.sh: every run of the baseline before the communication work
#  (26 Nov 2026), as time.sh calls. 300 s runs, 7-minute jobs, whole nodes.
#
#  Usage (from the project root, next to time.sh):
#      ./baseline.sh probe          # first: a 3-minute test that 4-node jobs start
#      ./baseline.sh 1 2            # submit blocks 1 and 2
#      ./baseline.sh all            # blocks 1-5 (not io, repeat or probe)
#      ./baseline.sh --dry-run 3    # print block 3's job scripts, submit nothing
#      ./baseline.sh --list         # print the runs of every block
#
#  Blocks (jobs; the 24-rank points of blocks 3-5 come from blocks 1-2):
#    1  one node, N = 100: the headline curve, v1, the merge-off floor and
#       a merge-heavy run                                                  (13)
#    2  one node, N = 480 and 4 800: where communication stops mattering    (6)
#    3  one cluster spread over 2-4 nodes: the Allgatherv between nodes,
#       plus v1 (allgather and ring) across nodes                         (11)
#    4  one cluster per node, same total N: clusters vs spread              (6)
#    5  weak scaling, 480 stars per node, with and without the far field
#       and merging (the second is the no-communication ceiling)           (7)
#    io      snapshots, MPI-IO vs gather, 1 and 2 nodes                     (4)
#    repeat  the key points twice more, to see the job-to-job spread        (4)
#    probe   48 and 96 ranks for 30 s, no results row: do multi-node jobs run?
#
#  Afterwards:  ./time.sh --summary 100 / 480 / 960 / 1440 / 1920 / 4800
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DRY=""
LIST=false
blocks=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY="--dry-run" ;;
        --list)    LIST=true ;;
        -h|--help) sed -n '2,/^# ====/p' "$0"; exit 0 ;;
        all)       blocks+=(1 2 3 4 5) ;;
        1|2|3|4|5|io|repeat|probe) blocks+=("$arg") ;;
        *) echo "Unknown block '$arg' (1-5, io, repeat, probe, all)" >&2; exit 1 ;;
    esac
done
$LIST && blocks=(probe 1 2 3 4 5 io repeat)
[ ${#blocks[@]} -gt 0 ] || { sed -n '2,/^# ====/p' "$0"; exit 1; }

# run [VAR=value ...] N "ranks" [time.sh options] [-- program options]
run() {
    local envs=()
    while [[ "$1" == *=* ]]; do envs+=("$1"); shift; done
    if $LIST; then
        local line="    ${envs[*]:+${envs[*]} }./time.sh" a
        for a in "$@"; do [[ "$a" == *" "* ]] && line+=" \"$a\"" || line+=" $a"; done
        echo "$line"
    else
        echo ">>> ${envs[*]:+${envs[*]} }./time.sh $*"
        env ${envs[@]+"${envs[@]}"} ./time.sh $DRY "$@"
    fi
}

block_probe() {
    run TIME_LIMIT=30 100 "96 48" -- --no-results
}

block_1() {   # one node, N = 100
    run 100 "24 16 8 4 2 1"                                         # lean path, merge radius 0.001
    run 100 "24 8 1" --program parallel                             # v1 (allgather), same data
    run MERGE_RADIUS=0 100 "24 8" -- --stop-check-every 100         # floor: Allgatherv + kernel only
    run MERGE_RADIUS=0.01 100 "24 8"                                # more merges
}

block_2() {   # one node, larger N
    run 480 "24 8 1"
    run 4800 "24 8 1"
}

block_3() {   # one cluster spread over nodes (24-rank points: blocks 1-2)
    run 100 "96 48"
    run 480 "96 72 48"
    run 4800 "96 72 48"
    run MERGE_RADIUS=0 100 "48" -- --stop-check-every 100           # floor across 2 nodes
    run 100 "48" --program parallel                                 # v1 allgather across 2 nodes
    run 100 "48" --program parallel -- --decomposition ring         # v1 ring across 2 nodes
}

block_4() {   # one cluster per node, same total (24-rank points: block 2)
    run 480 "96 72 48" --clusters per-node --total
    run 4800 "96 72 48" --clusters per-node --total
}

block_5() {   # weak scaling: 480 stars per node (24-rank point: block 2)
    run 480 "96 72 48" --clusters per-node
    run MERGE_RADIUS=0 480 "96 72 48 24" --clusters per-node -- --far-field off --stop-check-every 100
}

block_io() {
    run 100 "48 24" --io
    run IO_METHOD=gather 100 "48 24" --io
}

block_repeat() {
    run 100 "48 24"
    run 100 "48 24"
}

for b in "${blocks[@]}"; do
    if $LIST; then echo "block $b:"; else echo "===== block $b"; fi
    "block_$b"
done