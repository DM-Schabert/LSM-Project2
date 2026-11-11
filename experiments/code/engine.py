"""Scheduled coalescence: identical canonical state, two ownership policies.

Array rows always follow immutable particle IDs. Ownership only determines which
rows a rank updates. Allgatherv packs by rank and unpacks back to canonical order,
so changing ownership cannot change the force summation order.
"""
import argparse
import hashlib
import json
import os
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from mpi4py import MPI

import SerialImplementation as serial
from common import verify_manifest


def balanced_owners(n, p):
    counts = np.full(p, n // p, dtype=np.int64)
    counts[:n % p] += 1
    return np.repeat(np.arange(p, dtype=np.int64), counts)


class State:
    def __init__(self, stars, ids, owners, comm):
        self.stars, self.ids, self.owners = stars, ids, owners
        self.rank, self.size = comm.Get_rank(), comm.Get_size()
        self.indices = np.flatnonzero(owners == self.rank)
        self.local_positions = stars.positions[self.indices].copy()
        self.local_velocities = stars.velocities[self.indices].copy()
        blocks = [np.flatnonzero(owners == r) for r in range(self.size)]
        self.packed_indices = np.concatenate(blocks)
        self.counts = np.array([len(block) * 3 for block in blocks], dtype=np.int64)
        self.displacements = np.r_[0, np.cumsum(self.counts)[:-1]]
        self.receive = np.empty((stars.count, 3), dtype=np.float64)
        # Shared kernel accepts contiguous ranges. Split arbitrary ownership into
        # contiguous runs while preserving canonical row order within the rank.
        self.ranges = []
        for block in np.split(self.indices, np.flatnonzero(np.diff(self.indices) != 1) + 1):
            if len(block):
                self.ranges.append((int(block[0]), int(block[-1]) + 1))

    def exchange(self, mine, out, comm):
        comm.Allgatherv(mine, [self.receive, self.counts, self.displacements, MPI.DOUBLE])
        out[self.packed_indices] = self.receive

    def accelerations(self, softening):
        result = np.empty_like(self.local_positions)
        cursor = 0
        for start, stop in self.ranges:
            block, _ = serial.compute_accelerations(self.stars.positions, self.stars.masses,
                                                     softening, 0.0, start, stop)
            result[cursor:cursor + len(block)] = block
            cursor += len(block)
        return result


def checkpoint(state, step):
    digest = hashlib.sha256()
    for array in (state.ids, state.stars.positions, state.stars.velocities,
                  state.stars.masses, state.stars.members, state.stars.owner):
        digest.update(np.ascontiguousarray(array).tobytes())
    return {"step": step, "n": state.stars.count, "state_sha256": digest.hexdigest()}


def merge_event(state, pair_ids, policy, comm):
    start = time.perf_counter()
    n_before, owned_before = state.stars.count, len(state.indices)
    positions = np.empty((n_before, 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    state.exchange(state.local_positions, positions, comm)
    state.exchange(state.local_velocities, velocities, comm)
    exchange_s = time.perf_counter() - start
    full = state.stars
    full.positions, full.velocities = positions, velocities
    index = {int(value): i for i, value in enumerate(state.ids)}
    pairs = []
    for first, second in pair_ids:
        if first not in index or second not in index or first == second:
            raise RuntimeError(f"Scheduled IDs are not alive and distinct: {first}, {second}")
        pairs.append(tuple(sorted((index[first], index[second]))))
    pairs_local = sum(state.owners[i] == state.rank for i, _ in pairs)
    cross_local = sum(state.owners[i] == state.rank and state.owners[i] != state.owners[j]
                      for i, j in pairs)
    start_compute = time.perf_counter()
    labels = serial.group_close_stars(n_before, pairs)
    representative = np.full(int(labels.max()) + 1, n_before, dtype=np.int64)
    np.minimum.at(representative, labels, np.arange(n_before))
    merged = serial.merge_stars(full, pairs)
    compute_s = time.perf_counter() - start_compute
    start_layout = time.perf_counter()
    ids = state.ids[representative]
    retained = state.owners[representative]
    owners = balanced_owners(merged.count, state.size) if policy == "immediate" else retained.copy()
    changed = int(np.count_nonzero(owners != retained))
    result = State(merged, ids, owners, comm)
    rebuild_s = time.perf_counter() - start_layout
    row = {"rank": state.rank, "n_before": n_before, "n_after": merged.count,
           "owned_before": owned_before, "owned_after": len(result.indices),
           "scheduled_pairs_local": int(pairs_local), "cross_rank_pairs_local": int(cross_local),
           "logical_owner_changes_global": changed,
           "event_exchange_s": exchange_s, "merge_compute_s": compute_s,
           "layout_rebuild_s": rebuild_s, "event_total_s": time.perf_counter() - start}
    return result, row


def run(config, data, schedule, policy, comm, *, trace, time_limit):
    rank, size = comm.Get_rank(), comm.Get_size()
    stars = serial.load_subset(data, np.random.default_rng(config["seed"])) if rank == 0 else None
    # One setup-only pickled broadcast. Every hot-path exchange uses NumPy buffers.
    stars = comm.bcast(stars, root=0)
    initial_mass = float(stars.masses.sum())
    state = State(stars, np.arange(stars.count, dtype=np.int64), balanced_owners(stars.count, size), comm)
    records, events, checkpoints = [], [], []
    local_interactions = 0
    requested = config["steps"]
    steps_done = 0
    stop_local = np.zeros(1, dtype=np.int32)
    stop_global = np.zeros_like(stop_local)
    comm.Barrier()
    simulation_start = time.perf_counter()
    for step in range(1, requested + 1):
        step_start = time.perf_counter()
        stop_local[0] = int(time.perf_counter() - simulation_start > time_limit)
        comm.Allreduce(stop_local, stop_global, op=MPI.MAX)
        if stop_global[0]:
            break
        n_before, owned_before = state.stars.count, len(state.indices)
        event_s = 0.0
        if step in schedule:
            state, event = merge_event(state, schedule[step], policy, comm)
            event_s = event["event_total_s"]
            if trace:
                events.append({"step": step, **event})
                if rank == 0:
                    checkpoints.append(checkpoint(state, step))
        start = time.perf_counter()
        state.exchange(state.local_positions, state.stars.positions, comm)
        communication_s = time.perf_counter() - start
        start = time.perf_counter()
        acc = state.accelerations(config["softening"])
        forces_s = time.perf_counter() - start
        interactions = len(state.indices) * (state.stars.count - 1)
        local_interactions += interactions
        start = time.perf_counter()
        state.local_velocities += acc * config["dt"]
        state.local_positions += state.local_velocities * config["dt"]
        integrate_s = time.perf_counter() - start
        steps_done = step
        if trace:
            records.append({"rank": rank, "step": step, "n_before": n_before,
                            "n_after": state.stars.count, "owned_before": owned_before,
                            "owned_after": len(state.indices), "pair_interactions_local": interactions,
                            "forces_s": forces_s, "position_exchange_s": communication_s,
                            "event_total_s": event_s, "integrate_s": integrate_s,
                            "step_s": time.perf_counter() - step_start})
    comm.Barrier()
    elapsed = time.perf_counter() - simulation_start
    # All output, validation and trace gathering are outside the timed interval.
    state.exchange(state.local_positions, state.stars.positions, comm)
    state.exchange(state.local_velocities, state.stars.velocities, comm)
    elapsed = comm.reduce(elapsed, op=MPI.MAX, root=0)
    interactions = comm.reduce(local_interactions, op=MPI.SUM, root=0)
    hosts = comm.gather(socket.gethostname(), root=0)
    gathered = comm.gather((records, events), root=0) if trace else None
    if rank:
        return None
    expected = 0
    n = stars.count
    for step in range(1, steps_done + 1):
        if step in schedule:
            # Preparation requires disjoint, alive pairs within each event.
            n -= len(schedule[step])
        expected += n * (n - 1)
    if interactions != expected:
        raise RuntimeError(f"Interaction count {interactions} differs from schedule prediction {expected}")
    final = state.stars
    np.testing.assert_allclose(final.masses.sum(), initial_mass, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(serial.total_momentum(final), 0, rtol=0, atol=1e-8)
    numeric = np.r_[final.positions.ravel(), final.velocities.ravel(), final.masses]
    if not np.isfinite(numeric).all():
        raise RuntimeError("Non-finite final state")
    return {"state": state, "records": gathered, "checkpoints": checkpoints,
            "summary": {"policy": policy, "ranks": size, "steps": steps_done,
                        "steps_requested": requested, "completed": steps_done == requested,
                        "n_initial": stars.count, "n_final": final.count,
                        "pair_interactions": interactions, "expected_pair_interactions": expected,
                        "simulation_wall_s": elapsed, "mean_step_ms": elapsed * 1000 / max(steps_done, 1),
                        "hosts": sorted(set(hosts)), "job_id": os.getenv("LSB_JOBID", "local"),
                        "trace": trace}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=["reference", "immediate", "retained"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--time-limit", type=float, default=120)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    verify_manifest(root)
    config = json.loads((root / "config.json").read_text())
    if args.policy == "reference" and MPI.COMM_WORLD.Get_size() != 1:
        p.error("Reference must use one process")
    if args.fixture:
        config["steps"], config["dt"] = 6, 0.001
    schedule_file = "fixture_schedule.json" if args.fixture else "schedule.json"
    schedule = {int(event["step"]): event["pairs"]
                for event in json.loads((root / "data" / schedule_file).read_text())}
    result = run(config, root / "data" / ("fixture.csv" if args.fixture else "input.csv"),
                 schedule, args.policy, MPI.COMM_WORLD, trace=args.trace, time_limit=args.time_limit)
    if MPI.COMM_WORLD.Get_rank():
        return
    args.out.mkdir(parents=True, exist_ok=True)
    state = result["state"]
    stars = state.stars
    pd.DataFrame({"particle_id": state.ids, "x_pc": stars.positions[:, 0], "y_pc": stars.positions[:, 1],
                  "z_pc": stars.positions[:, 2], "vx_kms": stars.velocities[:, 0],
                  "vy_kms": stars.velocities[:, 1], "vz_kms": stars.velocities[:, 2],
                  "mass_msun": stars.masses, "merged_from": stars.members,
                  "sptype": stars.sptypes, "sptype_guessed": stars.sptype_guessed}).to_csv(args.out / "final.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(result["summary"], indent=2))
    if args.trace:
        for index, name in enumerate(("rank_steps", "merge_events")):
            rows = [row for block in result["records"] for row in block[index]]
            if rows:
                pd.DataFrame(rows).to_csv(args.out / (name + ".csv"), index=False)
        (args.out / "checkpoints.json").write_text(json.dumps(result["checkpoints"], indent=2))
    print(json.dumps(result["summary"], indent=2), flush=True)
    if not result["summary"]["completed"]:
        raise RuntimeError("Incomplete run; excluded from accepted results")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
