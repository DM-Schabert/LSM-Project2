"""Readable paired-policy findings, plots and raw CSVs from a passed job."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import verify_manifest


def main():
    root = Path(__file__).resolve().parents[1]
    verify_manifest(root)
    config = json.loads((root / "config.json").read_text())
    ranks = config["ranks"][0]
    job = root / "jobs" / f"p{ranks}"
    status_file = job / "status.json"
    if not status_file.exists():
        raise RuntimeError("Job has no final status yet. Check bjobs and the LSF logs.")
    status = json.loads(status_file.read_text())
    if not status.get("passed"):
        raise RuntimeError(f"Job failed; no accepted timing report: {status.get('error', 'see logs')}")
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    rows, step_rows, event_rows, stage_rows = [], [], [], []
    diagnostics = {}
    for policy in config["policies"]:
        folder = job / "runs" / f"{policy}_diagnostic"
        steps = pd.read_csv(folder / "rank_steps.csv")
        events = pd.read_csv(folder / "merge_events.csv")
        diagnostics[policy] = steps
        for step, frame in steps.groupby("step"):
            step_rows.append({"policy": policy, "step": step, "n": frame.n_after.iloc[0],
                              "owned_min": frame.owned_after.min(), "owned_max": frame.owned_after.max(),
                              "work_max_over_mean": frame.owned_after.max() / frame.owned_after.mean(),
                              "force_time_max_over_mean": frame.forces_s.max() / frame.forces_s.mean(),
                              "mean_position_exchange_ms": frame.position_exchange_s.mean() * 1000})
        for step, frame in events.groupby("step"):
            event_rows.append({"policy": policy, "step": step, "n_before": frame.n_before.iloc[0],
                               "n_after": frame.n_after.iloc[0],
                               "cross_rank_pairs": frame.cross_rank_pairs_local.sum(),
                               "logical_owner_changes": frame.logical_owner_changes_global.iloc[0],
                               "mean_event_ms": frame.event_total_s.mean() * 1000})
        times = events.groupby("rank")[["event_exchange_s", "merge_compute_s", "layout_rebuild_s"]].sum().mean() * 1000
        stage_rows.append({"policy": policy, **times.to_dict()})
        for repeat in range(1, config["repeats"] + 1):
            run = job / "runs" / f"{policy}_repeat{repeat}"
            summary = json.loads((run / "summary.json").read_text())
            validation = json.loads((run / "validation.json").read_text())
            if not validation["passed"] or not summary["completed"] or summary["trace"]:
                raise RuntimeError(f"Unaccepted timing run: {run}")
            rows.append({"policy": policy, "repeat": repeat,
                         "simulation_wall_s": summary["simulation_wall_s"],
                         "mean_step_ms": summary["mean_step_ms"], "n_final": summary["n_final"],
                         "pair_interactions": summary["pair_interactions"]})
    runs = pd.DataFrame(rows)
    if runs.pair_interactions.nunique() != 1 or runs.n_final.nunique() != 1:
        raise RuntimeError("Policies did not perform identical scheduled work")
    runs.to_csv(output / "timing_runs.csv", index=False)
    summary = runs.groupby("policy").agg(repeats=("repeat", "size"), mean_wall_s=("simulation_wall_s", "mean"),
                                         std_wall_s=("simulation_wall_s", "std"),
                                         mean_step_ms=("mean_step_ms", "mean"), std_step_ms=("mean_step_ms", "std"))
    summary.to_csv(output / "timing_summary.csv")
    paired = runs.pivot(index="repeat", columns="policy", values="simulation_wall_s")
    paired["saved_seconds_immediate"] = paired.retained - paired.immediate
    paired["retained_over_immediate"] = paired.retained / paired.immediate
    paired.to_csv(output / "paired_comparison.csv")
    per_step = pd.DataFrame(step_rows)
    per_step.to_csv(output / "diagnostic_steps.csv", index=False)
    pd.DataFrame(event_rows).to_csv(output / "merge_events.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(output / "event_stage_mean_rank_ms.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, policy in zip(axes[0], config["policies"]):
        for rank, frame in diagnostics[policy].groupby("rank"):
            x = np.r_[0, frame.step.to_numpy()]
            y = np.r_[frame.owned_before.iloc[0], frame.owned_after.to_numpy()]
            ax.step(x, y, where="post", label=f"rank {rank}")
        for step in config["event_steps"]:
            ax.axvline(step, color="gray", alpha=0.15, linewidth=1)
        ax.set(title=f"{policy}: actual ownership over time", xlabel="Step", ylabel="Owned particles")
        ax.set_ylim(0, max(config["n_input"] / ranks * 1.15, 1))
        ax.legend(fontsize=8)
    for policy, frame in per_step.groupby("policy"):
        axes[1, 0].plot(frame.step, frame.force_time_max_over_mean, label=policy, alpha=0.8)
    axes[1, 0].set(title="Observed force-time imbalance (diagnostic)", xlabel="Step", ylabel="Max / mean across ranks")
    axes[1, 0].legend()
    ax = axes[1, 1]
    for policy, color in (("immediate", "#2563eb"), ("retained", "#ea8a23")):
        ax.plot(paired.index, paired[policy], marker="o", label=policy, color=color)
    ax.set(title="Actual timing repeats: same work, same node", xlabel="Paired repeat", ylabel="Simulation wall time (seconds)")
    ax.set_xticks(paired.index)
    ax.legend()
    fig.suptitle(f"{config['name']}: scheduled coalescence, P={ranks}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output / "FINDINGS.png", dpi=160)
    plt.close(fig)
    event_stages = pd.DataFrame(stage_rows).set_index("policy")
    ax = event_stages.rename(columns={"event_exchange_s": "Full-state exchange", "merge_compute_s": "Merge computation",
                                      "layout_rebuild_s": "Layout rebuilding"}).plot.bar(stacked=True, figsize=(7, 4))
    ax.set(xlabel="Policy", ylabel="Total ms per average rank over all scheduled events")
    ax.tick_params(axis="x", rotation=0)
    ax.figure.tight_layout()
    ax.figure.savefig(output / "event_stages.png", dpi=160)
    plt.close(ax.figure)
    last_step = config["steps"]
    owner_text, imbalance_text = [], []
    for policy in config["policies"]:
        last = diagnostics[policy].query("step == @last_step").sort_values("rank")
        owner_text.append(f"- **{policy}:** {last.owned_after.astype(int).tolist()} particles per rank at the end.")
        tail = per_step[(per_step.policy == policy) & (per_step.step > config["event_steps"][-1])]
        imbalance_text.append(f"- {policy}: mean force-time max/mean after the last event = "
                              f"{tail.force_time_max_over_mean.mean():.4f}; theoretical work max/mean = "
                              f"{last.owned_after.max() / last.owned_after.mean():.4f}.")
    speedup = summary.loc["retained", "mean_wall_s"] / summary.loc["immediate", "mean_wall_s"]
    benefit = "immediate" if speedup > 1 else "retained"
    lines = [f"# {config['name']}: results", "", "![Ownership, imbalance and paired timing](FINDINGS.png)", "",
             "## Result", "",
             f"Both policies passed the same {len(config['event_steps'])}-event schedule, matching event checkpoint states "
             f"and final states. Each timing run completed {config['steps']} steps and {int(runs.pair_interactions.iloc[0]):,} interactions.", "",
             *owner_text, "", f"The ratio mean(retained walltime) / mean(immediate walltime) is **{speedup:.4f}**. "
             f"The lower observed mean belongs to **{benefit}** in this allocation. Values above 1 favour immediate redistribution.", "",
             "## Five timing repeats (or configured repeat count)", "",
             "| Policy | Repeats | Mean walltime (s) | SD between runs (s) | Mean ms/step |",
             "|---|---:|---:|---:|---:|"]
    for policy, row in summary.iterrows():
        lines.append(f"| {policy} | {int(row.repeats)} | {row.mean_wall_s:.6f} | {row.std_wall_s:.6f} | {row.mean_step_ms:.4f} |")
    lines += ["", "[Each paired measurement](paired_comparison.csv). "
              f"Mean paired time saved by immediate: {paired.saved_seconds_immediate.mean():.6f} s; "
              f"sample SD of paired differences: {paired.saved_seconds_immediate.std():.6f} s.", "",
              "## Why the timings may differ", "", *imbalance_text, "",
              "The force work is proportional to owned_particles*(N-1). Equal blocks reduce the slowest rank's "
              "interaction count. MPI communication and synchronization include waiting for that slowest rank. "
              "Exchange time therefore does not measure pure network latency.", "",
              "![Event-stage costs](event_stages.png)", "",
              "Both policies exchange the full state for each scheduled event and reconstruct it on every rank. "
              "Immediate redistribution changes ownership using this already replicated state; there is no extra "
              "Scatterv or dedicated migration transfer. Event-stage times are separate diagnostic measurements, "
              "summed per rank then averaged; they do not directly subtract from the uninstrumented timing repeats.", "",
              "## Scope of the conclusion", "",
              "This is a synthetic prescribed workload using Gaia initial positions, not naturally occurring stellar "
              "collisions. A schedule intentionally concentrated in one initial owner's block tests an unfavourable "
              "case for retained ownership. It does not establish how often this pattern occurs in natural simulations.", "",
              f"Both policies ran on {status['hostname']} in job {status['job_id']}; order alternated across repeats. "
              "Five launches within one allocation do not measure variation between nodes/days. If timing differences "
              "are small relative to run-to-run variation, repeat in a new allocation before claiming an improvement.", "",
              "A distributed merge implementation could have different migration costs. The result applies to this "
              "replicated-state implementation and does not justify a universal load-balancing policy.", "",
              "## Evidence", "", "- [Frozen configuration](../config.json)", "- [Fixed particle-ID schedule](../data/schedule.json)",
              "- [Source and input hashes](../manifest.json)", "- [Raw timing repeats](timing_runs.csv)",
              "- [Per-step imbalance](diagnostic_steps.csv)", "- [Merge events and ownership changes](merge_events.csv)",
              "- [Event-stage timings](event_stage_mean_rank_ms.csv)", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "REPORT.md")
    print(summary.to_string())
    print(f"retained/immediate mean-walltime ratio: {speedup:.4f}")


if __name__ == "__main__":
    main()
