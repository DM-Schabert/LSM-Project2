"""Problem sizes and time budget of the Gustafson (weak-scaling) sweeps.

Only the Python standard library, so it runs on the login node without modules.

Weak scaling grows the problem with the number of ranks P, so that every rank keeps
the work of the base problem with N0 stars. Two rules for N(P):

    time   (default) every rank keeps the serial run time of the base problem:
           T_serial(N(P)) = P * T_serial(N0), with the measured serial cost per step
           T_serial(N) = a*N + b*N^2   (a: per star, NumPy's per-row overhead; b: per pair)
    pairs  every rank keeps the number of pair interactions of the base problem:
           N(P) = N0 * sqrt(P)   (the rule of code/plot_results.py)

Direct summation costs a*N + b*N^2 per step, not b*N^2. With the pairs rule a rank
of the scaled problem gets fewer but longer rows than the base problem, which NumPy
handles faster per pair, so its share costs less time than the base problem did and
the weak efficiency can come out above 100 % with no parallel effect at all. The
time rule gives every rank the same serial time, which is what Gustafson's law
assumes ("the total parallel time in the scaled problem is p").

    python3 gustafson/plan.py --n0 100                       # sizes and time estimate
    python3 gustafson/plan.py --n0 100 --sizes               # only the N list (for scripts)
    python3 gustafson/plan.py --n0 100 --steps 6000 --repeats 3 --walltime 30
    python3 gustafson/plan.py --fit amdahl/out/v1-numpy/by_N_v1-numpy.csv   # a and b of a code version
"""
import argparse
import csv
import math
import sys
from pathlib import Path
from typing import List, Tuple

# Fit of the serial step times of v1-numpy (Amdahl campaign v1-numpy_20261111_0949, Xeon E5-2650 v4,
# N = 50, 100, 1 000, 5 000; within 3.4 % of every point). Refit for a new code version with --fit.
A_US_PER_STAR = 23.47
B_NS_PER_PAIR = 113.5

P_LIST = [1, 2, 4, 8, 12, 16, 24]
SERIAL_MIN_STEPS = 10
# rough costs for the time estimate (from the Amdahl campaign)
STARTUP_SERIAL_S = 1.4
STARTUP_MPI_S, STARTUP_MPI_PER_RANK_S = 1.8, 0.028
OVERHEAD_MS_PER_STEP = 0.15        # Allgather, Python and waiting per step at P > 1, about constant


def step_ms(n: float, a_us: float = A_US_PER_STAR, b_ns: float = B_NS_PER_PAIR) -> float:
    """Serial time per step in ms of the cost model a*N + b*N^2."""
    return a_us * 1e-3 * n + b_ns * 1e-6 * n * n


def n_of_p(n0: int, p: int, rule: str, a_us: float = A_US_PER_STAR, b_ns: float = B_NS_PER_PAIR) -> int:
    if p == 1:
        return n0
    if rule == "pairs":
        return int(round(n0 * math.sqrt(p)))
    if rule == "time":
        a, b = a_us * 1e-3, b_ns * 1e-6            # ms per star, ms per pair
        target = p * step_ms(n0, a_us, b_ns)       # b N^2 + a N = target
        return int(round((-a + math.sqrt(a * a + 4 * b * target)) / (2 * b)))
    raise ValueError(f"unknown rule {rule!r}: use time or pairs")


def sizes(n0: int, ps: List[int], rule: str, a_us: float, b_ns: float) -> List[int]:
    ns = [n_of_p(n0, p, rule, a_us, b_ns) for p in ps]
    if len(set(ns)) != len(ns):
        raise SystemExit(f"Two P values give the same N ({ns}); use a larger N0.")
    return ns


def serial_steps(steps: int, p: int, minimum: int = SERIAL_MIN_STEPS) -> int:
    """Steps of the serial run on the problem of P: the scaled problem takes about P times longer
    per step, so ~steps/P steps keep the run as long as the base run. Only the time per step is used."""
    return steps if p == 1 else max(minimum, -(-steps // p))


def estimate(n0: int, ps: List[int], ns: List[int], steps: int, a_us: float, b_ns: float,
             serial: bool = True) -> Tuple[float, List[Tuple[int, int, float, float, int, float]]]:
    """Seconds per repeat, and per P: (P, N, serial ms/step, MPI run s, serial steps, serial run s)."""
    rows, total = [], STARTUP_SERIAL_S + steps * step_ms(n0, a_us, b_ns) / 1000   # serial base run
    for p, n in zip(ps, ns):
        t_serial = step_ms(n, a_us, b_ns)
        mpi_ms = t_serial / p + (OVERHEAD_MS_PER_STEP if p > 1 else 0.0)
        mpi_s = STARTUP_MPI_S + STARTUP_MPI_PER_RANK_S * p + steps * mpi_ms / 1000
        k = serial_steps(steps, p)
        ser_s = STARTUP_SERIAL_S + k * t_serial / 1000 if (serial and p > 1) else 0.0
        rows.append((p, n, t_serial, mpi_s, k, ser_s))
        total += mpi_s + ser_s
    return total, rows


def fit(path: Path) -> Tuple[float, float]:
    """Least-squares a, b (relative errors) from a by_N_<label>.csv of amdahl_analysis.py."""
    with open(path, newline="") as f:
        pts = [(float(r["n_stars"]), float(r["serial_ms_per_step"])) for r in csv.DictReader(f)]
    if len(pts) < 2:
        raise SystemExit(f"Need at least two N in {path}")
    # t/N = a + b N, weighted by 1/(t/N): minimises the relative error of t
    sw = sxx = sx = sy = sxy = 0.0
    for n, t in pts:
        y, w = t / n, (n / t) ** 2
        sw += w; sx += w * n; sy += w * y; sxx += w * n * n; sxy += w * n * y
    b = (sw * sxy - sx * sy) / (sw * sxx - sx * sx)
    a = (sy - b * sx) / sw
    for n, t in pts:
        print(f"  N = {n:>7,.0f}: measured {t:9.3f} ms/step, model {a * n + b * n * n:9.3f} ms "
              f"({(a * n + b * n * n) / t - 1:+.1%})", file=sys.stderr)
    return a * 1e3, b * 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n0", type=int, help="stars of the base problem (P = 1)")
    ap.add_argument("--rule", choices=["time", "pairs"], default="time")
    ap.add_argument("--p", type=int, nargs="+", default=P_LIST)
    ap.add_argument("--a-us", type=float, default=A_US_PER_STAR, help="serial cost per star and step (µs)")
    ap.add_argument("--b-ns", type=float, default=B_NS_PER_PAIR, help="serial cost per pair and step (ns)")
    ap.add_argument("--steps", type=int, help="steps of every MPI run and of the serial base run")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--walltime", type=float, help="minutes; warns if the estimate does not fit")
    ap.add_argument("--no-serial", action="store_true", help="estimate without the serial runs at N(P)")
    ap.add_argument("--sizes", action="store_true", help="print only the N list, one per P")
    ap.add_argument("--fit", type=Path, metavar="BY_N_CSV", help="fit a and b to an Amdahl by_N table")
    args = ap.parse_args()

    if args.fit:
        a, b = fit(args.fit)
        print(f"--a-us {a:.2f} --b-ns {b:.1f}")
        return
    if args.n0 is None:
        ap.error("--n0 is required")
    ns = sizes(args.n0, args.p, args.rule, args.a_us, args.b_ns)
    if args.sizes:
        print(" ".join(map(str, ns)))
        return

    steps = args.steps or max(1, round(20_000 / step_ms(args.n0, args.a_us, args.b_ns)))
    per_rep, rows = estimate(args.n0, args.p, ns, steps, args.a_us, args.b_ns, not args.no_serial)
    print(f"N0 = {args.n0:,}, rule {args.rule}, {steps:,} steps, cost model {args.a_us:.2f} µs·N + "
          f"{args.b_ns:.1f} ns·N² per step")
    print(f"  {'P':>3} {'N':>7} {'N/P':>7} {'pairs/rank':>11} {'serial ms/step':>15} {'MPI run':>8} "
          f"{'serial steps':>13} {'serial run':>11}")
    pairs0 = args.n0 * (args.n0 - 1)
    for p, n, t, mpi_s, k, ser_s in rows:
        print(f"  {p:>3} {n:>7,} {n / p:>7.1f} {n * (n - 1) / p / pairs0:>10.2f}× {t:>15.2f} {mpi_s:>7.0f}s "
              f"{k if p > 1 else '':>13} {(f'{ser_s:.0f}s' if p > 1 and ser_s else ''):>11}")
    total = per_rep * args.repeats
    print(f"  one repeat ≈ {per_rep / 60:.1f} min, {args.repeats} repeats ≈ {total / 60:.1f} min "
          f"(MPI runs assumed at ~100 % efficiency, so allow some margin)")
    if args.walltime and total > 0.85 * args.walltime * 60:
        print(f"  WARNING: that is more than 85 % of the {args.walltime:.0f} min walltime. "
              f"Fewer steps or repeats, or a longer walltime.")


if __name__ == "__main__":
    main()
