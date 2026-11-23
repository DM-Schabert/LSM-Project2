#!/usr/bin/env python3
"""Run the existing N-body program unchanged for candidate N and P values.

Example (from n_pilot/, with config.sh sourced):
    python3 run_pilot.py --dry-run
    python3 run_pilot.py --n 250 500 1000 2000 --ranks 2 4 8 --steps 100 --repeats 2

Requires PILOT_SIM_COMMAND to be defined in config.sh. Logs are retained.
"""
import argparse
import csv
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / 'results'
LOGS = HERE / 'logs'
FIELDS = ['n','p','steps','repeat','elapsed_s','steps_per_s','exit_code','hostname','command','log']


def command_for(template, n, p, steps, rep):
    data_pattern = os.environ.get('PILOT_DATA_TEMPLATE', '')
    data = data_pattern.format(N=n, P=p, STEPS=steps, REP=rep) if data_pattern else ''
    cmd = template.format(N=n, P=p, STEPS=steps, REP=rep, DATA=data)
    return shlex.split(cmd)


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n', type=int, nargs='+', default=[250, 500, 1000, 2000])
    p.add_argument('--ranks', type=int, nargs='+', default=[2, 4, 8])
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--timeout', type=int, default=900, help='Seconds allowed per run')
    p.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    p.add_argument('--output', default='results/simulation_runs.csv')
    args = p.parse_args()
    if min(args.n + args.ranks + [args.steps, args.repeats]) < 1:
        p.error('N, P, steps and repeats must all be positive.')
    template = os.environ.get('PILOT_SIM_COMMAND','').strip()
    if not template:
        p.error('PILOT_SIM_COMMAND is not configured. See n_pilot/config.sh and ../batch.sh.')
    if '{P}' not in template or '{N}' not in template and '{DATA}' not in template or '{STEPS}' not in template:
        p.error('Command template must reference {P}, {STEPS}, and either {N} or {DATA}.')
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    output = HERE / args.output
    for rep in range(1, args.repeats + 1):
        for n in args.n:
            for ranks in args.ranks:
                try:
                    cmd = command_for(template, n, ranks, args.steps, rep)
                except (ValueError, KeyError) as e:
                    p.error(f'Invalid template: {e}')
                # Refuse command substitutions and unexpected programs; no shell execution.
                if not cmd or Path(cmd[0]).name not in ('mpirun', 'mpiexec'):
                    p.error('PILOT_SIM_COMMAND must begin with mpirun or mpiexec.')
                label = f'N{n}_P{ranks}_R{rep}'
                logfile = LOGS / f'{label}.log'
                print(f'{label}: {shlex.join(cmd)}', flush=True)
                if args.dry_run:
                    continue
                t0 = time.perf_counter()
                try:
                    with logfile.open('w', encoding='utf-8') as file:
                        file.write(f'COMMAND: {shlex.join(cmd)}\n\n')
                        file.flush()
                        proc = subprocess.run(cmd, cwd=ROOT, stdout=file, stderr=subprocess.STDOUT,
                                              check=False, timeout=args.timeout)
                    rc = proc.returncode
                except subprocess.TimeoutExpired:
                    rc = 124
                    with logfile.open('a', encoding='utf-8') as file:
                        file.write(f'\nTIMEOUT after {args.timeout} seconds\n')
                dt = time.perf_counter() - t0
                row = dict(n=n, p=ranks, steps=args.steps, repeat=rep,
                           elapsed_s=round(dt, 6), steps_per_s=round(args.steps/dt, 6),
                           exit_code=rc, hostname=platform.node(), command=shlex.join(cmd),
                           log=str(logfile.relative_to(HERE)))
                append_row(output, row)
                print(f'  exit={rc}, wall={dt:.3f} s, {args.steps/dt:.2f} nominal steps/s', flush=True)
                if rc != 0:
                    print(f'ERROR: {label} failed. See {logfile}; stopping.', file=sys.stderr)
                    sys.exit(rc)
    print(f'Complete. Raw results: {output}', flush=True)
    print('CAUTION: elapsed_s includes process launch, input, setup, and output; review logs for application-only timing.')


if __name__ == '__main__':
    main()
