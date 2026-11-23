#!/usr/bin/env python3
"""Thin wrapper around the existing run_pilot.py. It is not a new simulator.

Places logs in a job-specific directory so repeated jobs never overwrite them.
Use --spread for a multi-node job (Open MPI --map-by node).
Every other argument is forwarded to run_pilot.py unchanged.
"""
import os
from pathlib import Path
import shlex
import subprocess
import sys

import run_pilot


def main():
    spread = '--spread' in sys.argv
    sys.argv = [v for v in sys.argv if v != '--spread']
    job = os.environ.get('LSB_JOBID', 'manual')
    topology = 'two_nodes' if spread else 'one_node'
    run_pilot.LOGS = run_pilot.HERE / 'logs' / f'high_ranks_{topology}_{job}'
    run_pilot.LOGS.mkdir(parents=True, exist_ok=True)

    template = os.environ.get('PILOT_SIM_COMMAND', '')
    if spread:
        words = shlex.split(template)
        if not words or Path(words[0]).name != 'mpirun':
            raise SystemExit('PILOT_SIM_COMMAND must begin with mpirun for --spread')
        words[1:1] = ['--map-by', 'node']
        os.environ['PILOT_SIM_COMMAND'] = shlex.join(words)

    # Preserve actual allocation details and CPU information beside each run.
    cpu = subprocess.run(['lscpu'], capture_output=True, text=True, check=False)
    cpu_line = next((line.strip() for line in cpu.stdout.splitlines()
                     if line.startswith('Model name:')), '(CPU unknown)')
    meta = run_pilot.LOGS / 'allocation.txt'
    meta.write_text('\n'.join([
        f'job_id={job}', f'topology_requested={topology}',
        f'lsb_slots={os.environ.get("LSB_DJOB_NUMPROC", "unknown")}',
        f'lsb_hosts={os.environ.get("LSB_MCPU_HOSTS", "unknown")}',
        f'cpu={cpu_line}',
        f'command_template={os.environ.get("PILOT_SIM_COMMAND", "")}',
        f'arguments={shlex.join(sys.argv[1:])}',
        '',
    ]), encoding='utf-8')
    print(f'Run logs: {run_pilot.LOGS.relative_to(run_pilot.HERE)}', flush=True)
    print(f'Allocation: {os.environ.get("LSB_MCPU_HOSTS", "unknown")}', flush=True)
    print(cpu_line, flush=True)
    run_pilot.main()


if __name__ == '__main__':
    main()
