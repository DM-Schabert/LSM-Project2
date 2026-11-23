#!/usr/bin/env python3
"""Summarize benchmarking CSV files. No MPI needed; plotting is optional."""

import csv
from collections import defaultdict
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main():
    input_files = sorted(RESULTS.glob("bench_*.csv"))
    if not input_files:
        print("No results/bench_*.csv found. Submit jobs/benchmark.lsf first.")
        return
    data = defaultdict(list)
    for file in input_files:
        with file.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (int(row["n"]), int(row["ranks"]), row["method"])
                data[key].append(float(row["us_per_call"]))

    summaries = []
    print(f"Analysing {len(input_files)} job file(s). Negative Δ means XYZ slower.\n")
    print(f"{'N':>8} {'P':>3} {'DOUBLE (us)':>13} {'XYZ (us)':>12} {'XYZ faster %':>13}")
    print("-" * 56)
    for n, ranks in sorted({(n, p) for n, p, method in data}):
        a = data.get((n, ranks, "double"))
        b = data.get((n, ranks, "xyz_custom"))
        if not a or not b:
            continue
        standard_us = statistics.median(a)
        xyz_us = statistics.median(b)
        faster_pct = (standard_us - xyz_us) / standard_us * 100 if standard_us else 0
        print(f"{n:8d} {ranks:3d} {standard_us:13.3f} {xyz_us:12.3f} {faster_pct:13.2f}")
        summaries.append(dict(n=n, ranks=ranks, double_us=standard_us,
                              xyz_us=xyz_us, xyz_faster_pct=faster_pct))

    if not summaries:
        print("No complete DOUBLE/XYZ comparisons yet.")
        return
    summary = RESULTS / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summaries[0].keys())
        w.writeheader()
        w.writerows(summaries)
    print(f"\nSaved {summary}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib locally if you'd also like a PNG plot.")
        return

    # Plot independent curves per rank count; do not imply a benchmark conclusion.
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ranks_set = sorted({d["ranks"] for d in summaries})
    for ranks in ranks_set:
        subset = sorted((d for d in summaries if d["ranks"] == ranks),
                        key=lambda d: d["n"])
        x = [d["n"] for d in subset]
        ax.plot(x, [d["double_us"] for d in subset], marker="o",
                label=f"DOUBLE, P={ranks}")
        ax.plot(x, [d["xyz_us"] for d in subset], marker="x", linestyle="--",
                label=f"XYZ, P={ranks}")
    ax.set_xlabel("Total number of stars (N)")
    ax.set_ylabel("Median Allgatherv time per call (microseconds)")
    ax.set_title("MPI.DOUBLE vs custom contiguous XYZ datatype")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    output = RESULTS / "benchmark_comparison.png"
    fig.savefig(output, dpi=170)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
