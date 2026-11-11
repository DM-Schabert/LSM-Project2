"""One allocation, both ownership policies, validated repeated measurements."""
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common import compare_states, sha256, verify_manifest


def check_diagnostic(folder, policy, ranks, reference):
    a = json.loads((reference / "checkpoints.json").read_text())
    b = json.loads((folder / "checkpoints.json").read_text())
    if a != b:
        raise RuntimeError("Event checkpoint states differ from the one-process reference")
    steps = pd.read_csv(folder / "rank_steps.csv")
    for _, frame in steps.groupby("step"):
        if len(frame) != ranks or frame.n_after.nunique() != 1 or frame.owned_after.sum() != frame.n_after.iloc[0]:
            raise RuntimeError("Invalid per-step particle ownership")
        if policy == "immediate" and frame.owned_after.max() - frame.owned_after.min() > 1:
            raise RuntimeError("Immediate policy failed to restore equal blocks")
    events = pd.read_csv(folder / "merge_events.csv")
    if policy == "retained" and not events.logical_owner_changes_global.eq(0).all():
        raise RuntimeError("Retained policy changed surviving owners")


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranks", type=int, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    verify_manifest(root)
    config = json.loads((root / "config.json").read_text())
    if args.ranks not in config["ranks"]:
        p.error("Use the configured rank count")
    job = root / "jobs" / f"p{args.ranks}"
    job.mkdir(parents=True, exist_ok=True)
    runs = job / "runs"
    runs.mkdir(exist_ok=False)
    deadline = time.monotonic() + config["walltime_min"] * 60 - 120
    status = {"passed": False, "ranks": args.ranks, "hostname": socket.gethostname(),
              "job_id": os.getenv("LSB_JOBID", "local"),
              "started_utc": datetime.now(timezone.utc).isoformat()}

    def launch(name, policy, *, trace=False, fixture=False, repeat=0):
        remaining = deadline - time.monotonic()
        if remaining < 25:
            raise RuntimeError("Insufficient remaining job budget; prepare a new campaign with more walltime")
        folder = runs / name
        folder.mkdir()
        command = [sys.executable, str(root / "code/engine.py"), "--policy", policy,
                   "--out", str(folder), "--time-limit", str(min(120, remaining - 20))]
        if trace:
            command.append("--trace")
        if fixture:
            command.append("--fixture")
        if policy != "reference":
            command = ["mpirun", "--report-bindings", "--bind-to", "core", "-np", str(args.ranks), *command]
        metadata = {"command": command, "policy": policy, "repeat": repeat,
                    "role": "fixture" if fixture else "reference" if policy == "reference" else "diagnostic" if trace else "timing",
                    "job_id": status["job_id"], "host": status["hostname"],
                    "manifest_sha256": sha256(root / "manifest.json")}
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"RUN {name}", flush=True)
        with (folder / "stdout.log").open("w") as out, (folder / "stderr.log").open("w") as err:
            subprocess.run(command, stdout=out, stderr=err, cwd=root, check=True, timeout=remaining - 5)
        result = json.loads((folder / "summary.json").read_text())
        if not result["completed"] or result["steps"] != (6 if fixture else config["steps"]):
            raise RuntimeError(f"Incomplete run: {name}")
        if len(result["hosts"]) != 1:
            raise RuntimeError("This comparison must run on one node")
        return folder, result

    def validate(reference, candidate):
        report = compare_states(reference / "final.csv", candidate / "final.csv")
        a = json.loads((reference / "summary.json").read_text())
        b = json.loads((candidate / "summary.json").read_text())
        if a["pair_interactions"] != b["pair_interactions"] or a["steps"] != b["steps"]:
            raise RuntimeError("Different work or completed step count")
        (candidate / "validation.json").write_text(json.dumps(report, indent=2))

    try:
        reference, ref = launch("fixture_reference", "reference", trace=True, fixture=True)
        if ref["n_final"] != 12:
            raise RuntimeError("Fixture should end with 12 particles")
        for policy in config["policies"]:
            candidate, _ = launch(f"fixture_{policy}", policy, trace=True, fixture=True)
            validate(reference, candidate)
            check_diagnostic(candidate, policy, args.ranks, reference)
        reference, ref = launch("reference", "reference", trace=True)
        if ref["n_final"] != config["n_final_expected"]:
            raise RuntimeError("Reference does not match the expected scheduled particle count")
        for policy in config["policies"]:
            candidate, _ = launch(f"{policy}_diagnostic", policy, trace=True)
            validate(reference, candidate)
            check_diagnostic(candidate, policy, args.ranks, reference)
        # Prove the diagnostic actually exercises the planned imbalance.
        retained = pd.read_csv(runs / "retained_diagnostic/rank_steps.csv")
        last = retained[retained.step == config["steps"]]
        if last.owned_after.max() - last.owned_after.min() <= 1:
            raise RuntimeError("No meaningful ownership imbalance developed")
        for repeat in range(1, config["repeats"] + 1):
            order = config["policies"] if repeat % 2 else config["policies"][::-1]
            for policy in order:
                candidate, _ = launch(f"{policy}_repeat{repeat}", policy, repeat=repeat)
                validate(reference, candidate)
        status["passed"] = True
        print("PASS: both policies agree with reference at all diagnostic events and final states; timing repeats validated", flush=True)
    except BaseException as exc:
        status["error"] = str(exc)
        traceback.print_exc()
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (job / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
