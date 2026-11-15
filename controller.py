"""Adaptive Amdahl -> Gustafson campaign (isoefficiency sweep) for the MPI N-body code.

One campaign measures one code version. It is a chain of short LSF jobs; each
job measures one point (N stars, P ranks) and, at its end, asks this script
what to measure next and submits that job itself.

The rule (one fixed N at a time, P going up):

    1. Amdahl (strong scaling): keep N fixed, step P up through p_list
       (1, 2, 4, 8, 12, 16, 24, then whole nodes: 48, 72, ...).
    2. A step P_prev -> P is "cost-effective" when BOTH
         marginal efficiency  (S(P) / S(P_prev)) / (P / P_prev) >= marginal   (default 0.8)
         parallel efficiency  E(P) = S(P) / P                   >= efficiency (default 0.5)
       S is the speed-up over the serial program, run in the same job on the same node.
    3. When a step is not cost-effective, P_prev is the last cost-effective P for
       this N (the "threshold"). Then Gustafson: grow N (N *= growth) and test the
       same step again at the bigger N. If it is cost-effective now, keep stepping P.
    4. The campaign ends when P reaches the end of p_list, or N can no longer be
       measured within the walltime, or a point fails twice.

The thresholds trace the isoefficiency curve: how fast N must grow with P
to keep the extra ranks worth it. plot.py draws it.

Everything is decided by replaying this rule over the measurements in
results.csv, so the campaign can be stopped, inspected, resumed or extended
(e.g. more nodes later) at any time without losing anything.

Commands (run from the code folder, the one with ParallelImplementation.py):
    python3 isoeff/controller.py init  --label v2-ring --mpi-args "--decomposition ring" [options] [--submit]
    python3 isoeff/controller.py next   --campaign isoeff/out/v2-ring      # decide + submit (also resumes)
    python3 isoeff/controller.py status --campaign isoeff/out/v2-ring
    python3 isoeff/controller.py set    --campaign isoeff/out/v2-ring max_nodes=2   # then 'next'
    python3 isoeff/controller.py finish --campaign ... --job <id>         # called by the job itself

Standard library only (the subset maker and plots import numpy/pandas/matplotlib when needed).
"""

import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

HERE: Path = Path(__file__).resolve().parent

DEFAULTS: Dict[str, object] = {
    # --- the sweep
    "n0": 500,                  # first N
    "growth": 2.0,              # N *= growth when a threshold is hit
    "n_max": 600_000,           # never above this N (Gaia has ~626k valid stars)
    "cores_per_node": 24,
    "max_nodes": 1,             # raise later with:  controller.py set ... max_nodes=2
    "p_list": None,             # None: 1,2,4,8,12,16,24 (one node) + 48,72,... (whole nodes)
    "marginal": 0.8,            # minimum marginal efficiency of a step P_prev -> P
    "efficiency": 0.5,          # minimum overall parallel efficiency E = S/P
    "confirm": True,            # re-measure a failing point once before accepting a threshold
    "at_max_p": "stop",         # stop | grow: at the last P, stop, or keep growing N (pure weak scaling)
    "max_jobs": 80,             # safety cap on the number of jobs in one campaign
    "max_attempts": 2,          # a point whose job produced no usable result this often ends the campaign
    # --- one job
    "walltime_min": 7,
    "reserve_s": 150,           # walltime not given to the runs: modules, subsets, controller, slack
    "run_overhead_s": 8,        # start-up per run outside the program's own time limit (mpirun, imports)
    "serial_share": 0.25,       # share of the run budget for the serial baseline (more if N needs it)
    "mpi_reps": 3,              # MPI repeats per job (fewer if they don't fit)
    "min_steps": 3,             # a run needs at least this many steps to count
    "queue": "hpcintro",
    "mem_per_slot": "2GB",
    "full_node": True,          # reserve whole nodes, so other jobs never share our cores
    "modules": ["pandas/2.3.0-python-3.12.11",
                "matplotlib/3.10.3-numpy-2.3.1-python-3.12.11",
                "mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8"],
    "mpirun": "mpirun --report-bindings",
    "map_opts": "",             # e.g. "--map-by core --bind-to core"
    "mpi_args": "",             # extra options for ParallelImplementation.py only, e.g. "--decomposition ring"
    "executor": "lsf",          # lsf | local (runs each job right here, for testing)
    # --- the simulation
    "dt": 0.01,
    "softening": 0.01,
    "merge_radius": 0.0,        # 0: N stays fixed, so every step does the same work
    "seed": 42,
    "subset_method": "nearest",
    # --- serial cost per step T(N) = a*N + b*N^2 [s], refitted from this campaign's own serial runs
    "cost_a": 23.47e-6,         # v1-numpy fit (Gustafson framework, 12 Nov)
    "cost_b": 113.5e-9,
}

RUN_RE = re.compile(r"_N(\d+)_P(\d+)_(serial|mpi)(?:_r\d+)?_j([A-Za-z0-9.]+)$")
LABEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")


# ---------------------------------------------------------------- campaign files

class Campaign:
    def __init__(self, path: Path) -> None:
        self.dir: Path = path.resolve()
        self.config_path = self.dir / "config.json"
        self.jobs_path = self.dir / "jobs.json"
        self.results = self.dir / "results.csv"
        if not self.config_path.exists():
            sys.exit(f"{self.dir} is not a campaign (no config.json). Start one with 'init'.")
        self.cfg: Dict = json.loads(self.config_path.read_text())

    # --- config
    def save_config(self) -> None:
        write_json(self.config_path, self.cfg)

    def p_list(self) -> List[int]:
        if self.cfg.get("p_list"):
            return [int(p) for p in self.cfg["p_list"]]
        cpn, nodes = int(self.cfg["cores_per_node"]), int(self.cfg["max_nodes"])
        one_node = [p for p in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128) if p < cpn] + [cpn]
        return one_node + [cpn * k for k in range(2, nodes + 1)]

    def n_grid(self) -> Iterator[int]:
        n: float = float(self.cfg["n0"])
        while round(n) <= int(self.cfg["n_max"]):
            yield int(round(n))
            n *= float(self.cfg["growth"])

    # --- jobs
    def jobs(self) -> List[Dict]:
        return json.loads(self.jobs_path.read_text()) if self.jobs_path.exists() else []

    def save_jobs(self, jobs: List[Dict]) -> None:
        write_json(self.jobs_path, jobs)

    @contextmanager
    def lock(self) -> Iterator[None]:
        import fcntl
        with open(self.dir / ".controller.lock", "a") as handle:
            fcntl.lockf(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.lockf(handle, fcntl.LOCK_UN)

    @property
    def code(self) -> Path:
        return self.dir / "code"

    @property
    def tools(self) -> Path:
        return self.dir / "isoeff"

    def subset(self, n: int) -> Path:
        return Path(self.cfg["subsets_dir"]) / f"gaia_{self.cfg['subset_method']}_{n}.csv"


def write_json(path: Path, data: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------- measurements

@dataclass
class JobResult:
    job: str
    serial_ms: float
    mpi_ms: List[float]
    host: str

    @property
    def speedup(self) -> float:
        return self.serial_ms / statistics.median(self.mpi_ms)


@dataclass
class Point:
    n: int
    p: int
    runs: List[JobResult] = field(default_factory=list)

    @property
    def speedup(self) -> float:
        return statistics.median(r.speedup for r in self.runs)

    @property
    def efficiency(self) -> float:
        return self.speedup / self.p

    @property
    def serial_ms(self) -> float:
        return statistics.median(r.serial_ms for r in self.runs)

    @property
    def mpi_ms(self) -> float:
        return statistics.median(statistics.median(r.mpi_ms) for r in self.runs)


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def measurements(camp: Campaign) -> Tuple[Dict[Tuple[int, int], Point], List[Tuple[int, float]]]:
    """Valid points from results.csv, plus every valid serial run as (N, seconds per step)."""
    min_steps: int = int(camp.cfg["min_steps"])
    per_job: Dict[str, Dict] = {}
    serial_samples: List[Tuple[int, float]] = []
    for row in read_rows(camp.results):
        match = RUN_RE.search(row.get("run_name", ""))
        if not match:
            continue
        n, p, kind, job = int(match[1]), int(match[2]), match[3], match[4]
        try:
            iterations, step_ms = int(row["iterations"]), float(row["step_time_mean_ms"])
        except (KeyError, ValueError):
            continue
        if iterations < min_steps or not math.isfinite(step_ms) or step_ms <= 0:
            continue
        entry = per_job.setdefault(job, {"n": n, "p": p, "serial": None, "mpi": [], "host": ""})
        if kind == "serial":
            entry["serial"] = step_ms
            entry["host"] = row.get("hostname", "")
            serial_samples.append((n, step_ms / 1000))
        else:
            entry["mpi"].append(step_ms)
    points: Dict[Tuple[int, int], Point] = {}
    for job, e in per_job.items():
        if e["serial"] is None or not e["mpi"]:
            continue
        points.setdefault((e["n"], e["p"]), Point(e["n"], e["p"])).runs.append(
            JobResult(job, e["serial"], e["mpi"], e["host"]))
    return points, serial_samples


# ---------------------------------------------------------------- time budget

def fit_cost(camp: Campaign, samples: List[Tuple[int, float]]) -> Tuple[float, float]:
    """Serial seconds per step T(N) = a*N + b*N^2, fitted to this campaign's serial runs."""
    a0, b0 = float(camp.cfg["cost_a"]), float(camp.cfg["cost_b"])
    by_n: Dict[int, List[float]] = {}
    for n, t in samples:
        by_n.setdefault(n, []).append(t)
    if not by_n:
        return a0, b0
    ns = sorted(by_n)
    ts = [statistics.median(by_n[n]) for n in ns]
    if len(ns) == 1:  # keep a, scale b to the one measurement
        return a0, max((ts[0] - a0 * ns[0]) / ns[0] ** 2, 1e-12)
    # least squares on T/N = a + b*N
    xs, ys = ns, [t / n for t, n in zip(ts, ns)]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else b0
    a = my - b * mx
    if a < 0 or b <= 0:  # bad fit (noise at small N): fall back to a pure N^2 fit
        b = statistics.fmean(t / n ** 2 for t, n in zip(ts, ns))
        a = 0.0
    return a, b


@dataclass
class Plan:
    serial_limit: int
    mpi_limit: int
    reps: int
    serial_step_s: float
    mpi_step_s: float


def plan_job(camp: Campaign, n: int, p: int, cost: Tuple[float, float],
             points: Dict[Tuple[int, int], Point]) -> Optional[Plan]:
    """Time limits for one job, or None if N is too big to measure within the walltime."""
    c = camp.cfg
    a, b = cost
    ts: float = a * n + b * n * n
    # efficiency guess for P: the last measured one at this P (any N), else 0.9
    e_guess: float = 0.9
    same_p = [pt for (pn, pp), pt in points.items() if pp == p]
    if same_p:
        e_guess = min(1.0, max(0.2, max(same_p, key=lambda pt: pt.n).efficiency))
    tm: float = ts / (p * e_guess)
    min_steps, overhead = int(c["min_steps"]), float(c["run_overhead_s"])

    def needed(step_s: float) -> float:  # time limit that surely gives min_steps (+1 warm-up) steps
        return (min_steps + 1) * step_s / 0.95 * 1.15 + 5

    budget: float = float(c["walltime_min"]) * 60 - float(c["reserve_s"])
    for reps in range(int(c["mpi_reps"]), 0, -1):
        usable: float = budget - (1 + reps) * overhead
        serial_limit = max(float(c["serial_share"]) * usable, needed(ts))
        mpi_limit = (usable - serial_limit) / reps
        if mpi_limit >= needed(tm) and serial_limit <= usable:
            return Plan(int(serial_limit), int(mpi_limit), reps, ts, tm)
    return None


# ---------------------------------------------------------------- the rule

@dataclass
class Decision:
    next_point: Optional[Tuple[int, int]]   # (N, P) to measure next, or None when done
    reason: str
    log: List[str]
    verdicts: Dict[Tuple[int, int], str]
    thresholds: List[Dict]


def replay(camp: Campaign, points: Dict[Tuple[int, int], Point],
           cost: Tuple[float, float]) -> Decision:
    c = camp.cfg
    ps: List[int] = camp.p_list()
    grid: List[int] = list(camp.n_grid())
    log: List[str] = []
    verdicts: Dict[Tuple[int, int], str] = {}
    thresholds: List[Dict] = []
    th_m, th_e = float(c["marginal"]), float(c["efficiency"])

    def done(reason: str) -> Decision:
        log.append(f"DONE: {reason}")
        return Decision(None, reason, log, verdicts, thresholds)

    def need(n: int, p: int, why: str) -> Decision:
        log.append(f"NEXT: N={n} P={p} ({why})")
        return Decision((n, p), why, log, verdicts, thresholds)

    def fits(n: int, p: int) -> bool:
        return (n, p) in points or plan_job(camp, n, p, cost, points) is not None

    if not grid:
        return done("n0 is larger than n_max")
    gi, k = 0, 0
    growing_at_max = False
    while True:
        n = grid[gi]
        p = ps[k]
        point = points.get((n, p))
        if point is None:
            if not fits(n, p):
                return done(f"N={n} does not fit in a {c['walltime_min']}-min job at P={p}")
            return need(n, p, "first measurement")

        if growing_at_max:  # pure weak scaling at the largest P: just record E
            verdicts[(n, p)] = "at max P"
            log.append(f"N={n} P={p}: S={point.speedup:.2f} E={point.efficiency:.0%} (growing N at max P)")
        elif k == 0:
            verdicts[(n, p)] = "baseline"
            log.append(f"N={n} P={p}: S={point.speedup:.2f} (MPI with one rank vs. serial)")
            if len(ps) == 1:
                return done("p_list has only one entry")
            k = 1
            continue
        else:
            p_prev = ps[k - 1]
            prev = points.get((n, p_prev))
            if prev is None:
                if not fits(n, p_prev):
                    return done(f"N={n} does not fit in a {c['walltime_min']}-min job at P={p_prev}")
                return need(n, p_prev, f"reference for the step {p_prev}->{p}")
            marginal = (point.speedup / prev.speedup) / (p / p_prev)
            ok = marginal >= th_m and point.efficiency >= th_e
            text = (f"N={n} P={p_prev}->{p}: S {prev.speedup:.2f}->{point.speedup:.2f}, "
                    f"marginal {marginal:.0%}, E={point.efficiency:.0%}")
            if not ok and c["confirm"] and len(point.runs) < 2:
                log.append(text + " -> below threshold, measuring again to confirm")
                return need(n, p, "confirm a threshold")
            verdicts.setdefault((n, p_prev), "reference")
            if ok:
                verdicts[(n, p)] = "ok"
                log.append(text + " -> cost-effective")
                if k < len(ps) - 1:
                    k += 1
                    continue
                if c["at_max_p"] != "grow":
                    return done(f"reached the largest P ({p}) and it is still cost-effective at N={n}: "
                                f"Amdahl's limit lies beyond {c['max_nodes']} node(s)")
                log.append(f"Reached the largest P ({p}); growing N at P={p} (at_max_p=grow)")
                growing_at_max = True
            else:
                verdicts[(n, p)] = "threshold"
                why = []
                if marginal < th_m:
                    why.append(f"marginal {marginal:.0%} < {th_m:.0%}")
                if point.efficiency < th_e:
                    why.append(f"E {point.efficiency:.0%} < {th_e:.0%}")
                log.append(text + f" -> NOT cost-effective ({', '.join(why)}). "
                                  f"Threshold: P*={p_prev} at N={n}; growing N")
                thresholds.append({"n": n, "p_star": p_prev, "p_fail": p, "marginal": marginal,
                                   "efficiency_at_fail": point.efficiency,
                                   "efficiency_at_p_star": prev.efficiency})

        # grow N (after a threshold, or at max P in grow mode)
        if gi + 1 >= len(grid):
            return done(f"N would pass n_max={c['n_max']}")
        n_next = grid[gi + 1]
        check = [p] if growing_at_max else [ps[k - 1], p]
        for pc in check:
            if not fits(n_next, pc):
                return done(f"next N={n_next} does not fit in a {c['walltime_min']}-min job at P={pc} "
                            f"(the serial baseline alone needs ~{(cost[0] * n_next + cost[1] * n_next ** 2):.0f} s/step)")
        gi += 1


# ---------------------------------------------------------------- jobs

def job_script(camp: Campaign, n: int, p: int, plan: Plan, name: str) -> str:
    c = camp.cfg
    cpn = int(c["cores_per_node"])
    if p <= cpn:
        slots, span = (cpn if c["full_node"] else p), "span[hosts=1]"
    else:
        nodes = math.ceil(p / cpn)
        slots, span = nodes * cpn, f"span[ptile={cpn}]"
    wall = int(c["walltime_min"])
    logs = camp.dir / "logs"
    q = shlex.quote
    sim = (f"--steps 100000000 --dt {c['dt']} --softening {c['softening']} "
           f"--merge-radius {c['merge_radius']} --seed {c['seed']}")
    base = f"iso_{c['label']}_N{n}_P{p}"
    modules = " ".join(c["modules"])
    return f"""#!/bin/bash
#BSUB -J {name}
#BSUB -q {c['queue']}
#BSUB -n {slots}
#BSUB -R "{span}"
#BSUB -R "rusage[mem={c['mem_per_slot']}]"
#BSUB -W {wall // 60:02d}:{wall % 60:02d}
#BSUB -o {logs}/{name}_%J.out
#BSUB -e {logs}/{name}_%J.err
# Generated by isoeff/controller.py for campaign '{c['label']}': N={n}, P={p}.
# Time plan: serial {plan.serial_limit} s (expect ~{plan.serial_step_s:.3g} s/step),
#            {plan.reps} x MPI {plan.mpi_limit} s (expect ~{plan.mpi_step_s:.3g} s/step)
set -u
CAMPAIGN={q(str(camp.dir))}
JOB=${{LSB_JOBID:-${{ISO_LOCAL_JOBID:-local}}}}
cd "$CAMPAIGN/code"
if command -v module >/dev/null 2>&1; then module load {modules}; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

echo "=== {name} job $JOB on $(hostname) started $(date)"
echo "slots: ${{LSB_DJOB_NUMPROC:-?}}   hosts/slots: ${{LSB_MCPU_HOSTS:-?}}"
echo "cpu:   $(lscpu 2>/dev/null | grep 'Model name' | sed 's/Model name:[[:space:]]*//')"

DATA={q(str(camp.subset(n)))}
python3 "$CAMPAIGN/isoeff/make_subset.py" --n {n} --method {c['subset_method']} \\
    --csv {q(str(c['gaia_csv']))} --out {q(str(c['subsets_dir']))}
ARGS="--data $DATA {sim} --results $CAMPAIGN/results.csv"

echo "--- serial baseline, N={n}"
timeout -k 10 {plan.serial_limit + 60} python3 SerialImplementation.py $ARGS \\
    --time-limit {plan.serial_limit} --run-name {base}_serial_j$JOB

for r in $(seq 1 {plan.reps}); do
    echo "--- MPI repeat $r/{plan.reps}, P={p}"
    timeout -k 10 {plan.mpi_limit + 60} {c['mpirun']} -np {p} {c['map_opts']} python3 ParallelImplementation.py $ARGS {c.get('mpi_args', '')} \\
        --time-limit {plan.mpi_limit} --run-name {base}_mpi_r${{r}}_j$JOB
done

python3 "$CAMPAIGN/isoeff/controller.py" finish --campaign "$CAMPAIGN" --job "$JOB"
if [ "${{ISO_CHAIN:-1}}" = 1 ]; then
    echo "--- deciding and submitting the next point"
    python3 "$CAMPAIGN/isoeff/controller.py" next --campaign "$CAMPAIGN" --from-job "$JOB"
fi
echo "=== job $JOB finished $(date)"
"""


def lsf_state(jobid: str) -> str:
    """PEND/RUN/DONE/EXIT/..., or GONE if LSF no longer knows the job."""
    try:
        out = subprocess.run(["bjobs", "-noheader", "-o", "stat", jobid],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return "UNKNOWN"
    text = (out.stdout or "").strip().split()
    return text[0] if text else "GONE"


def submit(camp: Campaign, script: str, path: Path) -> str:
    path.write_text(script)
    if camp.cfg["executor"] == "local":
        return f"local{int(time.time() * 1000)}"
    out = subprocess.run(["bsub"], input=script, capture_output=True, text=True)
    match = re.search(r"Job <(\d+)>", out.stdout)
    if out.returncode != 0 or not match:
        raise RuntimeError(f"bsub failed ({out.returncode}): {out.stdout.strip()} {out.stderr.strip()}")
    return match[1]


# ---------------------------------------------------------------- reporting

def write_reports(camp: Campaign, points: Dict[Tuple[int, int], Point], decision: Decision,
                  cost: Tuple[float, float]) -> None:
    ps = camp.p_list()
    with open(camp.dir / "points.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["N", "P", "jobs", "serial_ms", "mpi_ms", "speedup", "efficiency",
                    "marginal_eff", "karp_flatt", "verdict", "hosts"])
        for (n, p) in sorted(points):
            pt = points[(n, p)]
            marg = ""
            if p in ps and ps.index(p) > 0:
                prev = points.get((n, ps[ps.index(p) - 1]))
                if prev:
                    marg = f"{(pt.speedup / prev.speedup) / (p / prev.p):.4f}"
            kf = f"{(1 / pt.speedup - 1 / p) / (1 - 1 / p):.5f}" if p > 1 else ""
            w.writerow([n, p, len(pt.runs), f"{pt.serial_ms:.3f}", f"{pt.mpi_ms:.3f}",
                        f"{pt.speedup:.4f}", f"{pt.efficiency:.4f}", marg, kf,
                        decision.verdicts.get((n, p), ""), ";".join(sorted({r.host for r in pt.runs}))])
    with open(camp.dir / "thresholds.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["N", "P_star", "P_fail", "marginal_eff", "E_at_fail", "E_at_P_star"])
        for t in decision.thresholds:
            w.writerow([t["n"], t["p_star"], t["p_fail"], f"{t['marginal']:.4f}",
                        f"{t['efficiency_at_fail']:.4f}", f"{t['efficiency_at_p_star']:.4f}"])
    lines = [f"# campaign {camp.cfg['label']} - {datetime.now().isoformat(timespec='seconds')}",
             f"# rule: marginal >= {camp.cfg['marginal']}, E >= {camp.cfg['efficiency']}, "
             f"P list {ps}, N0={camp.cfg['n0']} x{camp.cfg['growth']}",
             f"# serial cost fit: T(N) = {cost[0]:.4g}*N + {cost[1]:.4g}*N^2 s"] + decision.log
    (camp.dir / "decisions.log").write_text("\n".join(lines) + "\n")


def print_status(camp: Campaign, points: Dict[Tuple[int, int], Point], decision: Decision) -> None:
    print(f"Campaign {camp.cfg['label']}  ({camp.dir})")
    print(f"{'N':>7} {'P':>4} {'jobs':>4} {'serial ms':>10} {'MPI ms':>9} {'S':>7} {'E':>6}  verdict")
    for (n, p) in sorted(points):
        pt = points[(n, p)]
        print(f"{n:>7} {p:>4} {len(pt.runs):>4} {pt.serial_ms:>10.1f} {pt.mpi_ms:>9.2f} "
              f"{pt.speedup:>7.2f} {pt.efficiency:>6.0%}  {decision.verdicts.get((n, p), '')}")
    if decision.thresholds:
        print("Thresholds (last cost-effective P per N): " +
              ", ".join(f"N={t['n']}: P*={t['p_star']}" for t in decision.thresholds))
    jobs = camp.jobs()
    open_jobs = [j for j in jobs if not j.get("finished")]
    print(f"Jobs: {len(jobs)} submitted, {len(open_jobs)} not finished"
          + (f" ({', '.join(j['jobid'] for j in open_jobs)})" if open_jobs else ""))
    done = camp.dir / "DONE"
    if done.exists():
        print(f"Finished - {done.read_text().strip()}")
    elif decision.next_point:
        print("Next: N={} P={} - {}".format(*decision.next_point, decision.reason))
    else:
        print(f"Finished - {decision.reason}")


# ---------------------------------------------------------------- commands

def cmd_init(args: argparse.Namespace) -> None:
    code_dir: Path = HERE.parent
    if not LABEL_RE.match(args.label):
        sys.exit("Label may only contain letters, digits, '.', '_' and '-'")
    for name in ("SerialImplementation.py", "ParallelImplementation.py"):
        if not (code_dir / name).exists():
            sys.exit(f"{name} not found in {code_dir}: isoeff/ must sit in the code folder")
    out: Path = (code_dir / "isoeff" / "out" / args.label)
    if out.exists():
        sys.exit(f"{out} already exists. Use 'next --campaign {out}' to resume it, or pick a new label.")
    cfg: Dict = dict(DEFAULTS)
    cfg.update({k: v for k, v in vars(args).items() if k in DEFAULTS and v is not None})
    if args.no_confirm:
        cfg["confirm"] = False
    cfg["label"] = args.label
    cfg["created"] = datetime.now().isoformat(timespec="seconds")
    cfg["gaia_csv"] = str((args.gaia_csv or code_dir / "gaia_cleaned.csv").resolve())
    cfg["subsets_dir"] = str((args.subsets_dir or code_dir / "gaia_subsets").resolve())
    if args.note:
        cfg["note"] = args.note

    # Snapshot the code, so editing the next version while this campaign runs changes nothing here
    (out / "code").mkdir(parents=True)
    for src in code_dir.glob("*.py"):
        shutil.copy2(src, out / "code" / src.name)
    (out / "isoeff").mkdir()
    for name in ("controller.py", "make_subset.py", "plot.py"):
        if (HERE / name).exists():
            shutil.copy2(HERE / name, out / "isoeff" / name)
    (out / "logs").mkdir()
    (out / "jobs").mkdir()
    write_json(out / "config.json", cfg)
    camp = Campaign(out)
    print(f"Campaign '{args.label}' created in {out}")
    print(f"  code snapshot: {', '.join(sorted(p.name for p in (out / 'code').glob('*.py')))}")
    print(f"  P list {camp.p_list()}, N0={cfg['n0']} x{cfg['growth']}, "
          f"rule: marginal >= {cfg['marginal']:.0%} and E >= {cfg['efficiency']:.0%}")
    print(f"  MPI options: {cfg['mpi_args'] or '(none: allgather, v1 kernel)'}")
    if args.submit:
        args.campaign = out
        args.from_job = None
        cmd_next(args)
    else:
        print(f"Start it with:  python3 {camp.tools / 'controller.py'} next --campaign {out}")


def cmd_finish(args: argparse.Namespace) -> None:
    camp = Campaign(args.campaign)
    with camp.lock():
        jobs = camp.jobs()
        for j in jobs:
            if j["jobid"] == args.job:
                j["finished"] = datetime.now().isoformat(timespec="seconds")
        camp.save_jobs(jobs)


def evaluate(camp: Campaign) -> Tuple[Dict[Tuple[int, int], Point], Decision, Tuple[float, float]]:
    points, samples = measurements(camp)
    cost = fit_cost(camp, samples)
    return points, replay(camp, points, cost), cost


def cmd_next(args: argparse.Namespace) -> None:
    camp = Campaign(args.campaign)
    local = camp.cfg["executor"] == "local"
    while True:
        with camp.lock():
            jobs = camp.jobs()
            # Jobs that never called 'finish' (killed at the walltime, crashed node, bkill)
            for j in jobs:
                if j.get("finished") or j["jobid"] == args.from_job or local:
                    continue
                state = lsf_state(j["jobid"])
                if state in ("PEND", "RUN", "PSUSP", "USUSP", "SSUSP", "WAIT", "PROV", "UNKNOWN"):
                    print(f"Job {j['jobid']} (N={j['n']} P={j['p']}) is still {state}; it submits the "
                          f"next point itself when it ends. Nothing to do.")
                    return
                j["finished"] = f"lsf:{state}"
            camp.save_jobs(jobs)

            points, decision, cost = evaluate(camp)
            write_reports(camp, points, decision, cost)
            if decision.next_point is None:
                finalize(camp, decision)
                return
            n, p = decision.next_point
            measured_jobs = {r.job for pt in points.values() for r in pt.runs}
            failed = [j for j in jobs if j["n"] == n and j["p"] == p and j["jobid"] not in measured_jobs]
            reason = None
            if len(failed) >= int(camp.cfg["max_attempts"]):
                reason = (f"point N={n} P={p} failed {len(failed)} times "
                          f"(see logs/ for jobs {', '.join(j['jobid'] for j in failed)})")
            elif len(jobs) >= int(camp.cfg["max_jobs"]):
                reason = f"max_jobs={camp.cfg['max_jobs']} reached"
            if reason:
                decision.log.append(f"DONE: {reason}")
                decision.reason, decision.next_point = reason, None
                write_reports(camp, points, decision, cost)
                finalize(camp, decision)
                return

            # Make the subset now (cheap here), so the job's walltime goes to measuring
            try:
                sys.path.insert(0, str(HERE))
                from make_subset import ensure_subsets
                ensure_subsets([n], Path(camp.cfg["gaia_csv"]), Path(camp.cfg["subsets_dir"]),
                               camp.cfg["subset_method"], int(camp.cfg["seed"]))
            except Exception as exc:  # the job tries again itself
                print(f"Could not make the subset here ({type(exc).__name__}: {exc}); the job will try")

            plan = plan_job(camp, n, p, cost, points)
            assert plan is not None  # replay only proposes points that fit
            attempt = len([j for j in jobs if j["n"] == n and j["p"] == p]) + 1
            name = f"iso_{camp.cfg['label']}_N{n}_P{p}_a{attempt}"
            script = job_script(camp, n, p, plan, name)
            path = camp.dir / "jobs" / f"{name}.sh"
            try:
                jobid = submit(camp, script, path)
            except RuntimeError as exc:
                print(f"{exc}\nThe chain is broken; resume from a login node with:\n"
                      f"  python3 {camp.tools / 'controller.py'} next --campaign {camp.dir}")
                sys.exit(1)
            jobs.append({"jobid": jobid, "n": n, "p": p, "attempt": attempt, "script": str(path),
                         "submitted": datetime.now().isoformat(timespec="seconds"),
                         "reason": decision.reason, "plan": vars(plan)})
            camp.save_jobs(jobs)
            print(f"Submitted job {jobid}: N={n} P={p} ({decision.reason}); "
                  f"serial {plan.serial_limit} s + {plan.reps} x MPI {plan.mpi_limit} s")
        if not local:
            return
        # local executor: run the job right here, then decide again
        env = dict(os.environ, ISO_CHAIN="0", ISO_LOCAL_JOBID=jobid)
        log = camp.dir / "logs" / f"{name}_{jobid}.out"
        with open(log, "w") as handle:
            subprocess.run(["bash", str(path)], env=env, stdout=handle, stderr=subprocess.STDOUT)
        args.from_job = None


def finalize(camp: Campaign, decision: Decision) -> None:
    print(f"Campaign '{camp.cfg['label']}' finished: {decision.reason}")
    (camp.dir / "DONE").write_text(decision.reason + "\n")
    try:
        sys.path.insert(0, str(HERE))
        import plot
        for path in plot.make_plots([camp.dir], camp.dir / "plots"):
            print(f"Saved plot: {path}")
    except Exception as exc:  # plots are a convenience; the CSVs are the result
        print(f"Plots not made ({type(exc).__name__}: {exc}); run isoeff/plot.py by hand")
    print(f"Results: {camp.dir / 'points.csv'}, {camp.dir / 'thresholds.csv'}, {camp.dir / 'decisions.log'}")


def cmd_status(args: argparse.Namespace) -> None:
    camp = Campaign(args.campaign)
    points, decision, cost = evaluate(camp)
    if not (camp.dir / "DONE").exists():
        write_reports(camp, points, decision, cost)
    print_status(camp, points, decision)
    if args.log:
        print("\n".join(decision.log))


def cmd_set(args: argparse.Namespace) -> None:
    camp = Campaign(args.campaign)
    with camp.lock():
        for item in args.settings:
            key, _, raw = item.partition("=")
            if key not in DEFAULTS:
                sys.exit(f"Unknown setting '{key}'. Known: {', '.join(DEFAULTS)}")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            camp.cfg[key] = value
            print(f"{key} = {value!r}")
        camp.save_config()
        (camp.dir / "DONE").unlink(missing_ok=True)
    print(f"Continue with:  python3 {camp.tools / 'controller.py'} next --campaign {camp.dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create a campaign (snapshot of the current code)")
    p_init.add_argument("--label", required=True, help="e.g. v2-ring; names the campaign folder")
    p_init.add_argument("--note", help="free text stored in config.json (what changed in this version)")
    p_init.add_argument("--submit", action="store_true", help="submit the first job straight away")
    p_init.add_argument("--gaia-csv", type=Path, help="default: <code>/gaia_cleaned.csv")
    p_init.add_argument("--subsets-dir", type=Path, help="default: <code>/gaia_subsets")
    p_init.add_argument("--no-confirm", action="store_true", help="accept a threshold without re-measuring")
    for key in ("n0", "max_nodes", "cores_per_node", "mpi_reps", "min_steps", "walltime_min",
                "reserve_s", "n_max", "max_jobs"):
        p_init.add_argument("--" + key.replace("_", "-"), dest=key, type=int)
    for key in ("growth", "marginal", "efficiency", "dt", "softening", "merge_radius", "serial_share"):
        p_init.add_argument("--" + key.replace("_", "-"), dest=key, type=float)
    for key in ("queue", "mpirun", "map_opts", "mpi_args", "executor", "at_max_p", "subset_method"):
        p_init.add_argument("--" + key.replace("_", "-"), dest=key)
    p_init.add_argument("--p-list", dest="p_list", type=lambda s: [int(x) for x in s.split(",")],
                        help="explicit P list, e.g. 1,2,4,8,16,24")
    p_init.set_defaults(func=cmd_init)

    for name, func, text in (("next", cmd_next, "decide the next point and submit it (also resumes)"),
                             ("status", cmd_status, "print the measured points and the decision")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--campaign", type=Path, required=True)
        p.set_defaults(func=func)
        if name == "next":
            p.add_argument("--from-job", default=None, help=argparse.SUPPRESS)
        else:
            p.add_argument("--log", action="store_true", help="also print the full decision log")

    p_fin = sub.add_parser("finish", help="mark a job finished (called by the job)")
    p_fin.add_argument("--campaign", type=Path, required=True)
    p_fin.add_argument("--job", required=True)
    p_fin.set_defaults(func=cmd_finish)

    p_set = sub.add_parser("set", help="change settings of a campaign, e.g. max_nodes=2")
    p_set.add_argument("--campaign", type=Path, required=True)
    p_set.add_argument("settings", nargs="+", metavar="key=value")
    p_set.set_defaults(func=cmd_set)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
