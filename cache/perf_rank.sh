#!/bin/bash
# perf stat around one MPI rank, for cache/counters.sh when only per-process counters are allowed.
#   cache/perf_rank.sh <output prefix> <command ...>   ->   <output prefix>_r<rank>.txt
# The events come from $PERF_EVENTS (passed with mpirun -x).
out=$1
shift
exec perf stat -x, -o "${out}_r${OMPI_COMM_WORLD_RANK:-0}.txt" -e "$PERF_EVENTS" "$@"
