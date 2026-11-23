#!/usr/bin/env python3
"""Summarize full-simulator pilot and optional separate Allgatherv microbenchmark.

The optional MPI fraction is a screening proxy, NOT a measured application MPI fraction.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent

def read_mpi(path):
    d = defaultdict(list)
    if not path or not path.exists():
        return {}
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('method') == 'double':
                d[(int(row['n']), int(row['ranks']))].append(float(row['us_per_call']))
    return {k:statistics.median(v) for k,v in d.items()}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', default=str(HERE/'results/simulation_runs.csv'))
    ap.add_argument('--mpi', default='', help='Optional raw data_types bench_*.csv (single file)')
    args = ap.parse_args()
    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f'No results yet: {path}')
    grouped = defaultdict(list)
    with path.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if int(r['exit_code']) == 0:
                grouped[(int(r['n']), int(r['p']), int(r['steps']))].append(float(r['elapsed_s']))
    if not grouped:
        raise SystemExit('No successful full-simulation runs in CSV.')
    mpi = read_mpi(Path(args.mpi) if args.mpi else None)
    print('End-to-end timings include MPI startup, data loading and output. Do not interpret as step-only timing.')
    print(f"{'N':>7} {'P':>4} {'steps':>7} {'runs':>5} {'median wall (s)':>18} {'nominal steps/s':>18} {'MPI-only proxy %':>18}")
    print('-'*91)
    for (n,p,steps), values in sorted(grouped.items()):
        wall = statistics.median(values)
        mpi_us = mpi.get((n,p))
        frac = f'{mpi_us*steps/wall/1e4:.2f}' if mpi_us is not None else '---'
        print(f'{n:7d} {p:4d} {steps:7d} {len(values):5d} {wall:18.3f} {steps/wall:18.2f} {frac:>18}')
    print('\nSelection rule: pick a small N where the MPI fraction is appreciable AND total runs are long enough to overwhelm startup. Confirm with in-simulation phase timers.')
    if mpi:
        print('The proxy is from an isolated Allgatherv test; it is NOT a measured share of simulation runtime.')

if __name__ == '__main__':
    main()
