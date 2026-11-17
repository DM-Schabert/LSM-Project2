"""Scaling campaigns for the MPI N-body code: how many ranks/nodes and how many stars each version handles.

One campaign measures one code version. It is a chain of short LSF jobs; each job
measures one point (N stars, P ranks) and, at its end, asks this script what to
measure next and submits that job itself.

CAPACITY MODE (the default for new campaigns)
    Every run: `steps` (10 000) time steps of the MPI program in one `walltime_min` (7) minute
    job. It "fits" when it finishes all steps within its time limit. No serial runs.

    1. At P, grow N along the grid (300, 424, 600, 849, ... : x sqrt 2) until the run no longer
       fits: N_max(P), the most stars this P can simulate in the time budget.
    2. Step P up (1, 2, 4, 8, 12, 16, 24, then whole nodes 48, 72, ...) and time the new P on
       N_max of the previous P. The step pays off when
           (T(P_prev) / T(P)) / (P / P_prev) >= marginal      (default 0.8)
       If it does not (measured twice), P_prev is the last useful P and the campaign ends.
    3. Otherwise find N_max of the new P, and so on.

    --continue-from <previous campaign> starts the next version where the previous one
    stopped: --overlap (3) grid steps below its largest N, at the smallest P that held that
    N, with the same grid, steps and rule, so the two versions share points.

TIME AND STEP MODES (older campaigns; kept so they still replay)
    Speed-up against a serial run in the same job; P steps up while marginal efficiency
    >= 0.8 and E = S/P >= 0.5, and N doubles at a threshold (isoefficiency path).
    Time mode gives every run a time budget; step mode (--steps0) fixes the steps per N.

Everything is decided by replaying the rule over the measurements in results.csv, so the
campaign can be stopped, inspected, resumed or extended (e.g. more nodes) at any time.

Commands (run from the code folder, the one with ParallelImplementation.py):
    python3 isoeff/controller.py init  --label v1 --merge-radius 0.001 --submit
    python3 isoeff/controller.py init  --label v2-ring --continue-from isoeff/out/v1 --mpi-args "--decomposition ring" --submit
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
    # --- what the campaign measures
    "mode": "capacity",         # capacity (new campaigns) | time | steps (older campaigns; see the docstring)
    "steps": 10_000,            # capacity: steps of every run
    "run_reserve_s": 120,       # capacity: the run's time limit is walltime - this (300 s at 7 min, as batch.sh)
    "start_n": None,            # capacity: where to start (set by --continue-from)
    "start_p": None,
    # --- the sweep
    "n0": 500,                  # first N (capacity: 300)
    "growth": 2.0,              # N *= growth when a threshold is hit (capacity: sqrt(2) per probe)
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
    "walltime_min": 7,          # time mode: the walltime of every job
    "steps0": None,             # step mode: steps per run at N0, e.g. 10000 (None: time mode)
    "run_limit_min": 7,         # step mode: no single run may take longer than this
    "max_walltime_min": 30,     # step mode: longest job (several runs); fewer MPI repeats if needed
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

    @property
    def mode(self) -> str:
        """capacity | steps | time; campaigns made before the modes existed are time (or steps) mode."""
        return self.cfg.get("mode") or ("steps" if self.cfg.get("steps0") else "time")

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


# ---------------------------------------------------------------- steps per N (step mode)

def steps_for(camp: Campaign, n: int) -> Optional[int]:
    """Step mode: the number of steps every run at N does (serial and every MPI repeat, at every P).

    steps0 at N0; for other N, scaled so the serial run takes as long as at N0, with the cost
    model stored when the campaign was created (never refitted, so a point's step count never
    changes). Equal steps for serial and MPI mean the same trajectory, merges included, so the
    speed-up compares like with like. None in time mode.
    """
    s0 = camp.cfg.get("steps0")
    if not s0:
        return None
    a, b, n0 = float(camp.cfg["cost_a"]), float(camp.cfg["cost_b"]), int(camp.cfg["n0"])
    serial_step = lambda m: a * m + b * m * m  # noqa: E731
    floor = max(int(camp.cfg["min_steps"]) + 1, 10)
    return max(floor, int(round(int(s0) * serial_step(n0) / serial_step(n))))


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
    step_mode: bool = bool(camp.cfg.get("steps0"))
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
        if step_mode and str(iterations) != str(row.get("steps_requested", "")).split(".")[0]:
            continue  # stopped early: not the same trajectory as the other runs at this N
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
    steps: Optional[int] = None   # step mode: steps of every run
    walltime_min: int = 7


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

    steps = steps_for(camp, n)
    if steps is not None:  # step mode: fixed steps, the walltime follows from them
        run_cap = float(c["run_limit_min"]) * 60
        serial_run, mpi_run = steps * ts, steps * tm
        if serial_run * 1.3 + 10 > run_cap or mpi_run * 1.3 + 10 > run_cap:
            return None  # a single run would break the per-run limit
        serial_limit = int(min(run_cap, serial_run * 2 + 30))  # safety caps: a run that hits one does not count
        mpi_limit = int(min(run_cap, mpi_run * 3 + 30))
        for reps in range(int(c["mpi_reps"]), 0, -1):
            expected = (serial_run + reps * mpi_run) * 1.3 + (1 + reps) * overhead + float(c["reserve_s"])
            wall = math.ceil(expected / 60)
            if wall <= int(c["max_walltime_min"]):
                return Plan(serial_limit, mpi_limit, reps, ts, tm, steps, max(wall, 5))
        return None

    budget: float = float(c["walltime_min"]) * 60 - float(c["reserve_s"])
    for reps in range(int(c["mpi_reps"]), 0, -1):
        usable: float = budget - (1 + reps) * overhead
        serial_limit = max(float(c["serial_share"]) * usable, needed(ts))
        mpi_limit = (usable - serial_limit) / reps
        if mpi_limit >= needed(tm) and serial_limit <= usable:
            return Plan(int(serial_limit), int(mpi_limit), reps, ts, tm, None, int(c["walltime_min"]))
    return None


# ---------------------------------------------------------------- the rule

@dataclass
class Decision:
    next_point: Optional[Tuple[int, int]]   # (N, P) to measure next, or None when done
    reason: str
    log: List[str]
    verdicts: Dict[Tuple[int, int], str]
    thresholds: List[Dict]
    capacity: List[Dict] = field(default_factory=list)


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
    wall = int(plan.walltime_min)
    logs = camp.dir / "logs"
    q = shlex.quote
    sim = (f"--steps {plan.steps or 100000000} --dt {c['dt']} --softening {c['softening']} "
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
# Generated by isoeff/controller.py for campaign '{c['label']}': N={n}, P={p}{f", {plan.steps} steps per run" if plan.steps else ""}.
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

def write_reports(camp: Campaign, points: Dict, decision: Decision, cost: Tuple[float, float]) -> None:
    if camp.mode == "capacity":
        write_capacity_reports(camp, points, decision)
        return
    ps = camp.p_list()
    with open(camp.dir / "points.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["N", "P", "jobs", "serial_ms", "mpi_ms", "speedup", "efficiency",
                    "marginal_eff", "karp_flatt", "verdict", "hosts", "steps"])
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
                        decision.verdicts.get((n, p), ""), ";".join(sorted({r.host for r in pt.runs})),
                        steps_for(camp, n) or ""])
    with open(camp.dir / "thresholds.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["N", "P_star", "P_fail", "marginal_eff", "E_at_fail", "E_at_P_star"])
        for t in decision.thresholds:
            w.writerow([t["n"], t["p_star"], t["p_fail"], f"{t['marginal']:.4f}",
                        f"{t['efficiency_at_fail']:.4f}", f"{t['efficiency_at_p_star']:.4f}"])
    lines = [f"# campaign {camp.cfg['label']} - {datetime.now().isoformat(timespec='seconds')}",
             f"# rule: marginal >= {camp.cfg['marginal']}, E >= {camp.cfg['efficiency']}, "
             f"P list {ps}, N0={camp.cfg['n0']} x{camp.cfg['growth']}",
             f"# steps per run: {('fixed per N, ' + str(camp.cfg['steps0']) + ' at N0') if camp.cfg.get('steps0') else 'time-limited'}"
             f", merge radius {camp.cfg['merge_radius']} pc",
             f"# serial cost fit: T(N) = {cost[0]:.4g}*N + {cost[1]:.4g}*N^2 s"] + decision.log
    (camp.dir / "decisions.log").write_text("\n".join(lines) + "\n")


def print_status(camp: Campaign, points: Dict, decision: Decision) -> None:
    if camp.mode == "capacity":
        print_capacity_status(camp, points, decision)
        return
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


# ---------------------------------------------------------------- capacity mode (fixed steps, fixed walltime)
#
# Every run: `steps` steps (10 000) of the MPI program in one job of `walltime_min` (7)
# minutes; it "fits" when it finishes all steps within its time limit (walltime − run_reserve_s,
# 300 s at 7 min, as in batch.sh). No serial runs: correctness is checked separately.
#
#   1. At P, grow N along the grid (×√2) until the run no longer fits: N_max(P).
#   2. Step P up and time the same N_max(P) on the new P. The step pays off when
#        (T(P) / T(P_new)) / (P_new / P) >= marginal      (default 0.8)
#      Otherwise P is the last useful P (re-measured once first) and the campaign ends.
#   3. If it pays off, find N_max of the new P, and so on through p_list.
#
# A campaign made with --continue-from starts a few grid steps (--overlap, 3) below the
# largest N of the previous version, at the smallest P that held that N there, so the
# versions share points to compare.

@dataclass
class CapRun:
    job: str
    fits: bool         # finished all steps within the run's time limit
    time_s: float      # time of all `steps` steps (estimated from the step time if it did not finish)
    step_ms: float
    host: str


def capacity_runs(camp: "Campaign") -> Dict[Tuple[int, int], List[CapRun]]:
    """Every MPI run of the campaign by (N, P); crashed or interrupted runs say nothing and are left out."""
    steps = int(camp.cfg["steps"])
    out: Dict[Tuple[int, int], List[CapRun]] = {}
    for row in read_rows(camp.results):
        m = RUN_RE.search(row.get("run_name", ""))
        if not m or m[3] != "mpi":
            continue
        n, p, job = int(m[1]), int(m[2]), m[4]
        try:
            iterations, step_ms = int(float(row["iterations"])), float(row["step_time_mean_ms"])
        except (KeyError, ValueError):
            continue
        if not (math.isfinite(step_ms) and step_ms > 0):
            continue
        if iterations >= steps:
            try:
                total = float(row.get("simulation_time_s") or "nan")
            except ValueError:
                total = float("nan")
            fits, total = True, total if math.isfinite(total) else step_ms * iterations / 1000
        elif row.get("stop_reason") == "time_limit":
            fits, total = False, step_ms * steps / 1000
        else:
            continue
        out.setdefault((n, p), []).append(CapRun(job, fits, total, step_ms, row.get("hostname", "")))
    return out


def replay_capacity(camp: "Campaign", runs: Dict[Tuple[int, int], List[CapRun]]) -> "Decision":
    c = camp.cfg
    ps: List[int] = camp.p_list()
    grid: List[int] = list(camp.n_grid())
    steps, th = int(c["steps"]), float(c["marginal"])
    log: List[str] = []
    verdicts: Dict[Tuple[int, int], str] = {}
    thresholds: List[Dict] = []
    capacity: List[Dict] = []

    def done(reason: str) -> "Decision":
        log.append(f"DONE: {reason}")
        return Decision(None, reason, log, verdicts, thresholds, capacity)

    def need(n: int, p: int, why: str) -> "Decision":
        log.append(f"NEXT: N={n} P={p} ({why})")
        return Decision((n, p), why, log, verdicts, thresholds, capacity)

    def state(n: int, p: int) -> Optional[Tuple[bool, float, int]]:
        rs = runs.get((n, p), [])
        if not rs:
            return None
        fit = [r for r in rs if r.fits]
        ok = 2 * len(fit) >= len(rs)
        return ok, statistics.median(r.time_s for r in (fit if ok else rs)), len(rs)

    start_n, start_p = c.get("start_n"), c.get("start_p")
    if start_n is not None and int(start_n) not in grid:
        return done(f"start_n={start_n} is not on the N grid")
    i = grid.index(int(start_n)) if start_n is not None else 0
    k = max([j for j, p in enumerate(ps) if p <= int(start_p)] or [0]) if start_p is not None else 0
    if start_n is not None:
        log.append(f"Continuing from {c.get('continued_label', 'an earlier campaign')}: start at N={grid[i]}, P={ps[k]}")
    marginal_in: Optional[float] = None
    while True:
        p = ps[k]
        while True:  # 1. grow N at this P
            s = state(grid[i], p)
            if s is None:
                return need(grid[i], p, "capacity probe")
            if s[0]:
                verdicts.setdefault((grid[i], p), "fits")
                log.append(f"P={p} N={grid[i]}: {steps} steps in {s[1]:.0f} s -> fits")
                if i + 1 >= len(grid):
                    capacity.append({"P": p, "N_max": grid[i], "time_s": s[1], "marginal": marginal_in,
                                     "verdict": "useful (N limit)"})
                    return done(f"N reached n_max={c['n_max']} at P={p}; N_max = {grid[i]}")
                nxt = state(grid[i + 1], p)
                if nxt is None:
                    return need(grid[i + 1], p, "capacity probe")
                if nxt[0]:
                    i += 1
                    continue
                if verdicts.get((grid[i + 1], p)) != "too big":
                    verdicts[(grid[i + 1], p)] = "too big"
                    log.append(f"P={p} N={grid[i + 1]}: {steps} steps would take ~{nxt[1]:.0f} s -> too big")
                break
            verdicts[(grid[i], p)] = "too big"
            log.append(f"P={p} N={grid[i]}: {steps} steps would take ~{s[1]:.0f} s -> too big, trying a smaller N")
            if i == 0:
                return done(f"even N={grid[0]} does not finish {steps} steps in time at P={p}")
            i -= 1
        n_max, t_max = grid[i], state(grid[i], p)[1]
        capacity.append({"P": p, "N_max": n_max, "time_s": t_max, "marginal": marginal_in, "verdict": "useful"})
        log.append(f"N_max(P={p}) = {n_max}")
        if k + 1 >= len(ps):  # 2. step P up
            return done(f"reached the largest P ({p}) with every step still paying off; N_max = {n_max}")
        p_new = ps[k + 1]
        s = state(n_max, p_new)
        if s is None:
            return need(n_max, p_new, f"marginal test {p}->{p_new}")
        m = (t_max / s[1]) / (p_new / p)
        text = f"step {p}->{p_new} at N={n_max}: {t_max:.0f} s -> {s[1]:.0f} s, marginal {m:.0%}"
        if not s[0] or m < th:
            if c["confirm"] and s[2] < 2:
                log.append(text + " -> below the rule, measuring again to confirm")
                return need(n_max, p_new, "confirm a failing step")
            verdicts[(n_max, p_new)] = "threshold"
            thresholds.append({"P_last": p, "P_fail": p_new, "N": n_max, "marginal": m})
            capacity[-1]["verdict"] = "last useful P"
            log.append(text + f" -> does NOT pay off (< {th:.0%})")
            return done(f"adding ranks beyond P={p} no longer pays off: the step {p}->{p_new} at N={n_max} "
                        f"gives {m:.0%} (< {th:.0%}); N_max = {n_max}")
        verdicts[(n_max, p_new)] = "step ok"
        log.append(text + " -> pays off")
        marginal_in = m
        k += 1  # 3. N_max(p) is measured on p_new already; grow from there


def capacity_plan(camp: "Campaign") -> "Plan":
    c = camp.cfg
    wall = int(c["walltime_min"])
    limit = wall * 60 - int(c["run_reserve_s"])
    return Plan(0, limit, 1, 0.0, 0.0, int(c["steps"]), wall)


def capacity_job_script(camp: "Campaign", n: int, p: int, plan: "Plan", name: str) -> str:
    c = camp.cfg
    cpn = int(c["cores_per_node"])
    if p <= cpn:
        slots, span = (cpn if c["full_node"] else p), "span[hosts=1]"
    else:
        slots, span = math.ceil(p / cpn) * cpn, f"span[ptile={cpn}]"
    wall = plan.walltime_min
    q = shlex.quote
    sim = (f"--steps {plan.steps} --dt {c['dt']} --softening {c['softening']} "
           f"--merge-radius {c['merge_radius']} --seed {c['seed']}")
    return f"""#!/bin/bash
#BSUB -J {name}
#BSUB -q {c['queue']}
#BSUB -n {slots}
#BSUB -R "{span}"
#BSUB -R "rusage[mem={c['mem_per_slot']}]"
#BSUB -W {wall // 60:02d}:{wall % 60:02d}
#BSUB -o {camp.dir / 'logs'}/{name}_%J.out
#BSUB -e {camp.dir / 'logs'}/{name}_%J.err
# Generated by isoeff/controller.py (capacity mode) for campaign '{c['label']}': N={n}, P={p},
# one MPI run of {plan.steps} steps; it fits if it finishes within {plan.mpi_limit} s.
set -u
CAMPAIGN={q(str(camp.dir))}
JOB=${{LSB_JOBID:-${{ISO_LOCAL_JOBID:-local}}}}
cd "$CAMPAIGN/code"
if command -v module >/dev/null 2>&1; then module load {" ".join(c["modules"])}; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

echo "=== {name} job $JOB on $(hostname) started $(date)"
echo "slots: ${{LSB_DJOB_NUMPROC:-?}}   hosts/slots: ${{LSB_MCPU_HOSTS:-?}}"
echo "cpu:   $(lscpu 2>/dev/null | grep 'Model name' | sed 's/Model name:[[:space:]]*//')"

DATA={q(str(camp.subset(n)))}
python3 "$CAMPAIGN/isoeff/make_subset.py" --n {n} --method {c['subset_method']} \\
    --csv {q(str(c['gaia_csv']))} --out {q(str(c['subsets_dir']))}

echo "--- MPI, P={p}, N={n}, {plan.steps} steps"
timeout -k 10 {plan.mpi_limit + 60} {c['mpirun']} -np {p} {c['map_opts']} python3 ParallelImplementation.py \\
    --data $DATA {sim} --results $CAMPAIGN/results.csv {c.get('mpi_args', '')} \\
    --time-limit {plan.mpi_limit} --run-name iso_{c['label']}_N{n}_P{p}_mpi_r1_j$JOB

python3 "$CAMPAIGN/isoeff/controller.py" finish --campaign "$CAMPAIGN" --job "$JOB"
if [ "${{ISO_CHAIN:-1}}" = 1 ]; then
    echo "--- deciding and submitting the next point"
    python3 "$CAMPAIGN/isoeff/controller.py" next --campaign "$CAMPAIGN" --from-job "$JOB"
fi
echo "=== job $JOB finished $(date)"
"""


def write_capacity_reports(camp: "Campaign", runs: Dict[Tuple[int, int], List[CapRun]], decision: "Decision") -> None:
    with open(camp.dir / "cap_points.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["N", "P", "runs", "fits", "time_s", "ms_per_step", "verdict", "hosts"])
        for (n, p) in sorted(runs, key=lambda k: (k[1], k[0])):
            rs = runs[(n, p)]
            fit = [r for r in rs if r.fits]
            ok = 2 * len(fit) >= len(rs)
            use = fit if ok else rs
            w.writerow([n, p, len(rs), ok, f"{statistics.median(r.time_s for r in use):.2f}",
                        f"{statistics.median(r.step_ms for r in use):.4f}", decision.verdicts.get((n, p), ""),
                        ";".join(sorted({r.host for r in rs}))])
    with open(camp.dir / "capacity.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["P", "N_max", "time_s", "marginal_in", "verdict"])
        for e in decision.capacity:
            w.writerow([e["P"], e["N_max"], f"{e['time_s']:.2f}",
                        "" if e["marginal"] is None else f"{e['marginal']:.4f}", e["verdict"]])
    with open(camp.dir / "thresholds.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["P_last", "P_fail", "N", "marginal"])
        for t in decision.thresholds:
            w.writerow([t["P_last"], t["P_fail"], t["N"], f"{t['marginal']:.4f}"])
    c = camp.cfg
    lines = [f"# campaign {c['label']} (capacity mode) - {datetime.now().isoformat(timespec='seconds')}",
             f"# every run: {c['steps']} steps in a {c['walltime_min']}-min job (fits within "
             f"{int(c['walltime_min']) * 60 - int(c['run_reserve_s'])} s); P list {camp.p_list()}; "
             f"N grid {c['n0']} x{float(c['growth']):.4g}; rule: marginal >= {c['marginal']}; "
             f"merge radius {c['merge_radius']} pc"] + decision.log
    (camp.dir / "decisions.log").write_text("\n".join(lines) + "\n")


def print_capacity_status(camp: "Campaign", runs: Dict[Tuple[int, int], List[CapRun]], decision: "Decision") -> None:
    c = camp.cfg
    print(f"Campaign {c['label']} (capacity mode: {c['steps']} steps per {c['walltime_min']}-min job)  ({camp.dir})")
    print(f"{'P':>4} {'N':>7} {'runs':>4} {'time (s)':>9} {'ms/step':>9}  verdict")
    for (n, p) in sorted(runs, key=lambda k: (k[1], k[0])):
        rs = runs[(n, p)]
        fit = [r for r in rs if r.fits]
        use = fit if 2 * len(fit) >= len(rs) else rs
        print(f"{p:>4} {n:>7} {len(rs):>4} {statistics.median(r.time_s for r in use):>9.1f} "
              f"{statistics.median(r.step_ms for r in use):>9.2f}  {decision.verdicts.get((n, p), '')}")
    if decision.capacity:
        print("Capacity (largest N that finishes in time): " +
              ", ".join(f"P={e['P']}: {e['N_max']}" for e in decision.capacity))
    jobs = camp.jobs()
    open_jobs = [j for j in jobs if not j.get("finished")]
    print(f"Jobs: {len(jobs)} submitted, {len(open_jobs)} not finished"
          + (f" ({', '.join(j['jobid'] for j in open_jobs)})" if open_jobs else ""))
    done_file = camp.dir / "DONE"
    if done_file.exists():
        print(f"Finished - {done_file.read_text().strip()}")
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
    given = {k: v for k, v in vars(args).items() if k in DEFAULTS and v is not None}
    cfg.update(given)
    prev: Optional[Campaign] = None
    if args.continue_from:
        prev = Campaign(Path(args.continue_from))
        if prev.mode != "capacity":
            sys.exit(f"{prev.dir} is not a capacity campaign; --continue-from needs one")
        cfg["mode"] = "capacity"
        # same grid, steps and rule as the previous version, unless given here
        for key in ("n0", "growth", "steps", "marginal", "confirm", "walltime_min", "run_reserve_s", "dt",
                    "softening", "merge_radius", "seed", "subset_method", "cores_per_node", "max_nodes", "n_max"):
            if key not in given and key in prev.cfg:
                cfg[key] = prev.cfg[key]
    elif cfg["mode"] == "capacity":
        if "n0" not in given:
            cfg["n0"] = 300
        if "growth" not in given:
            cfg["growth"] = math.sqrt(2)
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
    for name in ("controller.py", "make_subset.py", "plot.py", "make_page.py", "results.html"):
        if (HERE / name).exists():
            shutil.copy2(HERE / name, out / "isoeff" / name)
    (out / "logs").mkdir()
    (out / "jobs").mkdir()
    write_json(out / "config.json", cfg)
    camp = Campaign(out)
    if prev is not None:  # start a few grid steps below where the previous version stopped
        _, prev_decision, _ = evaluate(prev)
        if not prev_decision.capacity:
            shutil.rmtree(out)
            sys.exit(f"{prev.dir} has no capacity results yet")
        grid = list(camp.n_grid())
        top = max(e["N_max"] for e in prev_decision.capacity)
        start = grid[max(0, grid.index(top) - int(args.overlap))]
        camp.cfg.update({"start_n": start,
                         "start_p": min(e["P"] for e in prev_decision.capacity if e["N_max"] >= start),
                         "continued_from": str(prev.dir), "continued_label": prev.cfg["label"]})
        camp.save_config()
        cfg = camp.cfg
    print(f"Campaign '{args.label}' created in {out}")
    print(f"  code snapshot: {', '.join(sorted(p.name for p in (out / 'code').glob('*.py')))}")
    if camp.mode == "capacity":
        grid = list(camp.n_grid())
        print(f"  capacity mode: every run {cfg['steps']} steps in a {cfg['walltime_min']}-min job; "
              f"P list {camp.p_list()}; N grid {', '.join(map(str, grid[:9]))}, ...; "
              f"rule: marginal >= {cfg['marginal']:.0%}")
        if cfg.get("start_n") is not None:
            print(f"  continues '{cfg['continued_label']}': starts at N={cfg['start_n']}, P={cfg['start_p']} "
                  f"({args.overlap} grid steps below its largest N)")
    else:
        print(f"  P list {camp.p_list()}, N0={cfg['n0']} x{cfg['growth']}, "
              f"rule: marginal >= {cfg['marginal']:.0%} and E >= {cfg['efficiency']:.0%}")
    print(f"  MPI options: {cfg['mpi_args'] or '(none: allgather, v1 kernel)'}")
    if cfg.get("steps0"):
        grid = list(camp.n_grid())[:6]
        print("  step mode: every run at one N does the same steps (serial and MPI, merges included): "
              + ", ".join(f"N={n}: {steps_for(camp, n)}" for n in grid)
              + f"; at most {cfg['run_limit_min']} min per run, {cfg['max_walltime_min']} min per job")
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


def evaluate(camp: Campaign) -> Tuple[Dict, Decision, Tuple[float, float]]:
    if camp.mode == "capacity":
        runs = capacity_runs(camp)
        return runs, replay_capacity(camp, runs), (float(camp.cfg["cost_a"]), float(camp.cfg["cost_b"]))
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
            if camp.mode == "capacity":
                measured_jobs = {r.job for rs in points.values() for r in rs}
            else:
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

            plan = capacity_plan(camp) if camp.mode == "capacity" else plan_job(camp, n, p, cost, points)
            assert plan is not None  # replay only proposes points that fit
            attempt = len([j for j in jobs if j["n"] == n and j["p"] == p]) + 1
            name = f"iso_{camp.cfg['label']}_N{n}_P{p}_a{attempt}"
            script = (capacity_job_script if camp.mode == "capacity" else job_script)(camp, n, p, plan, name)
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
            if camp.mode == "capacity":
                print(f"Submitted job {jobid}: N={n} P={p} ({decision.reason}); "
                      f"{plan.steps} steps, fits if done within {plan.mpi_limit} s")
            else:
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
    first = "capacity.csv" if camp.mode == "capacity" else "points.csv"
    print(f"Results: {camp.dir / first}, {camp.dir / 'thresholds.csv'}, {camp.dir / 'decisions.log'}")
    template = HERE / "results.html"
    if template.exists() and (HERE / "make_page.py").exists():
        page = camp.dir / "results.html"
        try:  # this campaign's own results page; compare versions with isoeff/make_page.py
            shutil.copy2(template, page)
            subprocess.run([sys.executable, str(HERE / "make_page.py"), str(camp.dir), "--page", str(page)],
                           check=True, capture_output=True, text=True, timeout=120)
            print(f"Results page: {page}")
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"Results page not made ({type(exc).__name__}); run isoeff/make_page.py by hand")


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
    p_init.add_argument("--continue-from", help="capacity mode: the previous version's campaign folder; "
                        "start where it stopped (same grid, steps and rule)")
    p_init.add_argument("--overlap", type=int, default=3,
                        help="with --continue-from: start this many N grid steps below its largest N (default 3)")
    for key in ("n0", "max_nodes", "cores_per_node", "mpi_reps", "min_steps", "walltime_min",
                "steps0", "run_limit_min", "max_walltime_min", "steps", "run_reserve_s",
                "reserve_s", "n_max", "max_jobs"):
        p_init.add_argument("--" + key.replace("_", "-"), dest=key, type=int)
    for key in ("growth", "marginal", "efficiency", "dt", "softening", "merge_radius", "serial_share"):
        p_init.add_argument("--" + key.replace("_", "-"), dest=key, type=float)
    for key in ("queue", "mpirun", "map_opts", "mpi_args", "executor", "at_max_p", "subset_method", "mode"):
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
