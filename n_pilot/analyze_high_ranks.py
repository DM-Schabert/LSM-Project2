#!/usr/bin/env python3
"""Summarise real simulator phase timings (not process-startup timings).

Reads job-isolated logs under logs/high_ranks_one_node_*/ and
logs/high_ranks_two_nodes_*/. Produces a summary CSV and two PNG figures.
Requires matplotlib to make plots; summary CSV works without it.
"""
from collections import defaultdict
import csv
from pathlib import Path
import re
from statistics import median

HERE = Path(__file__).resolve().parent
LOGS = HERE / 'logs'
RESULTS = HERE / 'results'
NAME = re.compile(r'N(?P<n>\d+)_P(?P<p>\d+)_R(?P<rep>\d+)\.log$')
STEPS = re.compile(r'Simulated\s+([\d.,]+)\s+steps.*?\bin\s+([\d.]+)\s+s')
FORCES = re.compile(r'^\s*forces\s+([\d.]+)\s+s\s*$', re.MULTILINE)
COMM = re.compile(r'^\s*communication\s+([\d.]+)\s+s.*?sync\s+([\d.]+)\s+s', re.MULTILINE)


def parse_file(path):
    m = NAME.search(path.name)
    if not m:
        return None
    text = path.read_text(encoding='utf-8', errors='replace')
    s, f, c = STEPS.search(text), FORCES.search(text), COMM.search(text)
    if not (s and f and c):
        print(f'SKIP incomplete log: {path}')
        return None
    # Existing simulator writes "10.000" for ten thousand steps.
    nsteps = int(s.group(1).replace('.', '').replace(',', ''))
    sim_s, forces_s, comm_s, sync_s = (float(s.group(2)), float(f.group(1)),
                                      float(c.group(1)), float(c.group(2)))
    if sim_s <= 0:
        return None
    parent = path.parent.name
    topology = 'one node' if parent.startswith('high_ranks_one_node_') else 'two nodes'
    job_id = parent.rsplit('_', 1)[-1]
    return dict(n=int(m.group('n')), p=int(m.group('p')), repeat=int(m.group('rep')),
                topology=topology, job_id=job_id, steps=nsteps, sim_s=sim_s,
                steps_per_s=nsteps/sim_s, forces_s=forces_s, comm_s=comm_s,
                comm_pct=100*comm_s/sim_s, sync_s=sync_s, log=str(path.relative_to(HERE)))


def main():
    files = sorted(LOGS.glob('high_ranks_one_node_*/N*_P*_R*.log')) + \
            sorted(LOGS.glob('high_ranks_two_nodes_*/N*_P*_R*.log'))
    rows = [item for file in files if (item := parse_file(file)) is not None]
    if not rows:
        print('No completed high-rank simulator logs found yet. Submit both jobs first.')
        return
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS/'high_ranks_raw_phases.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[(row['n'], row['topology'], row['p'])].append(row)
    summary = []
    for (n, topology, p), group in sorted(groups.items()):
        rate = [r['steps_per_s'] for r in group]
        frac = [r['comm_pct'] for r in group]
        summary.append(dict(n=n, topology=topology, p=p, runs=len(group),
                            steps_each=group[0]['steps'],
                            median_steps_per_s=median(rate), min_steps_per_s=min(rate), max_steps_per_s=max(rate),
                            median_comm_pct=median(frac), min_comm_pct=min(frac), max_comm_pct=max(frac),
                            median_sim_s=median([r['sim_s'] for r in group]),
                            median_forces_s=median([r['forces_s'] for r in group]),
                            median_sync_s=median([r['sync_s'] for r in group])))
    with (RESULTS/'high_ranks_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print('Simulator-only timing; excludes Python/MPI launch and CSV loading.\n')
    print(f'{"N":>4} {"topology":>11} {"P":>3} {"runs":>4} {"steps/s (median)":>18} {"MPI share %":>13}')
    print('-'*64)
    for s in summary:
        print(f'{s["n"]:>4} {s["topology"]:>11} {s["p"]:>3} {s["runs"]:>4} '
              f'{s["median_steps_per_s"]:>18.1f} {s["median_comm_pct"]:>13.2f}')
    print('\nNote: phase times are slowest-rank diagnostics and need not sum to total.')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Install/load matplotlib to generate PNG figures. CSV saved.')
        return

    # Graphs compare both N and physical rank placement; min/max shows run spread.
    colors = {50: '#279a89', 100: '#5981dc'}
    for measure, lo, hi, title, ylabel, out in [
        ('median_steps_per_s', 'min_steps_per_s', 'max_steps_per_s',
         'N-body timestep throughput vs MPI ranks', 'Completed timesteps / second', 'high_ranks_throughput.png'),
        ('median_comm_pct', 'min_comm_pct', 'max_comm_pct',
         'Communication share vs MPI ranks', 'Communication / simulation time (%)', 'high_ranks_communication.png'),
    ]:
        fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=170)
        fig.patch.set_facecolor('#fafbfe')
        ax.set_facecolor('white')
        for n in (50, 100):
            for topology, style, mark in [('one node', '-', 'o'), ('two nodes', '--', 's')]:
                points = sorted((v for v in summary if v['n']==n and v['topology']==topology),
                                key=lambda x: x['p'])
                if not points:
                    continue
                x = [p['p'] for p in points]
                y = [p[measure] for p in points]
                error = [[max(0, p[measure]-p[lo]) for p in points],
                         [max(0, p[hi]-p[measure]) for p in points]]
                ax.errorbar(x, y, yerr=error, label=f'N={n}, {topology}',
                            color=colors[n], linestyle=style, marker=mark,
                            markersize=7, linewidth=2.2, capsize=3, alpha=.95)
        ax.set_xscale('log', base=2)
        ax.set_xticks([2, 4, 8, 16, 32], labels=['2', '4', '8', '16', '32'])
        ax.grid(axis='y', color='#dbe2ef', linewidth=.8)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(title, loc='left', fontsize=17, fontweight='bold', pad=20)
        ax.set_xlabel('MPI ranks (P)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='best', frameon=False, fontsize=10)
        fig.text(.12, .015, 'Original ParallelImplementation.py | 50,000 steps | merges off | error bars: run range',
                 color='#627084', fontsize=9)
        fig.tight_layout(rect=(0, .035, 1, 1))
        fig.savefig(RESULTS/out, bbox_inches='tight')
        plt.close(fig)
        print('Saved', RESULTS/out)
    print('Saved', RESULTS/'high_ranks_summary.csv')


if __name__=='__main__':
    main()
