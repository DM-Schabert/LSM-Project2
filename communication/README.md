# Communication schemes: how the positions get to the ranks

Every star feels every other star, so in every time step each rank needs positions it does not own. This folder compares ways of moving them, one at a time, against the same baseline and with the same measurements.

**Question.** How does the communication scheme change the strong scaling of the direct-summation N-body code, for small and large N, on one node and on two?

What we know from the Amdahl campaign (`amdahl/results.html`): on one node and with large N, communication costs almost nothing (N = 5 000, P = 24: about 0.3 ms per step against about 120 ms of computing). With small N it is a large part of each step (N = 50: about a quarter of the step loop). So differences between the schemes will show **at small N, across nodes and at large P**, and hardly at all on one node with large N. Finding that is a result too.

## Layout

| Folder | What goes in |
|---|---|
| `schemes/` | one module per scheme. The baseline (v1, `Allgatherv` in `ParallelImplementation.py`) stays untouched |
| `tests/` | correctness: every scheme gives the same accelerations as v1, apart from summation order |
| `bench/` | communication alone, no physics: time per call against message size and P, to fit `T = μ + s/B` |
| `jobs/` | LSF job scripts (one node, two nodes) |
| `results/` | raw CSV rows of every run (in git) |
| `out/` | plots and tables, made again from `results/` (not in git) |

## Rules

1. **Expectation before measurement.** For every scheme, write down below what it should cost (data per rank per step, messages per step, memory per rank) before running it. Afterwards, compare.
2. **Correct before fast.** A scheme is only timed once its accelerations match v1 (`np.allclose`) for several N and P, including N not divisible by P.
3. **One change per iteration.** Each scheme, and each improvement of a scheme (e.g. overlap), is its own iteration with its own label, measured against the previous one on the same experiments.
4. **Everything else fixed.** Same force kernel, same N, steps, `dt`, softening, merging off, one thread per rank, same rank placement (`--report-bindings` in the `.err` file), 3 repeats, median and spread. Serial baseline and all P of one sweep on the same node.
5. **Traceable.** Label in the results = git tag of the code (`v1-numpy`, `v2-ring`, …). Every results row also carries the code fingerprint.
6. **Raw data in git, plots not.** `results/` is committed; `out/` is made again by the analysis.

## Schemes and what we expect

N stars, P ranks, 24 bytes per position (3 doubles). "Received" is per rank and time step.

| Scheme | Label | Received | Messages per step | Positions held per rank |
|---|---|---|---|---|
| Allgather (v1) | `v1-numpy` | 24 N (P − 1)/P | one collective, algorithm chosen by MPI | N |
| Ring (1D periodic Cartesian, `Sendrecv_replace`) | `v2-ring` | 24 N (P − 1)/P | P − 1 point-to-point | 2 N/P (own + travelling block) |
| Ring with overlap (`Isend`/`Irecv`) | `v2b-ring-overlap` | as ring | as ring, hidden behind computing | 3 N/P |
| 2D force decomposition (p_r × p_c grid, row and column communicators) | `v3-2d` | about 24 N/p_r + 24 N/p_c, plus the force `Allreduce` along the row | `Bcast` in column and row, `Allreduce` in the row | N/p_r + N/p_c |

Expectations to check:

- On one node, all schemes are about equally fast for large N.
- For small N the ring is slower than Allgather, because it pays P − 1 latencies instead of one collective.
- Overlap helps only where communication is a real share of the step: small N, or two nodes.
- 2D needs a grid close to square to pay off: at P = 24 (6 × 4) each rank still receives about 0.4 N positions, at P = 64 (8 × 8) about 0.25 N.
- Memory is not the limit at our sizes: all 626 000 Gaia stars are 15 MB of positions per rank.

## Experiments

| # | What | N | P | Nodes |
|---|---|---|---|---|
| C0 | Communication alone: `Allgatherv`, ring of `Sendrecv`, `Bcast` + `Allreduce`. Fit latency μ and bandwidth B | message sizes of 50 to 100 000 stars | 2 to 24 (48) | 1 and 2 |
| C1 | Correctness of every scheme against v1 | 7, 50, 1 000 (and N not divisible by P) | 1, 2, 3, 4, 8 | 1 |
| C2 | Strong scaling per scheme (with the Amdahl framework, one label per scheme) | 50, 100, 1 000, 5 000 | 1 … 24 | 1 |
| C3 | The same across two nodes | 1 000, 5 000, 20 000 | 8 … 48 | 2 |
| C4 | Overlap on and off (ring) | 50, 1 000 | 8, 24 | 1 and 2 |
| C5 | Memory per rank against N (peak RSS) | up to the largest that fits | 24 | 1 |

## Status

| Scheme | Expectation written | Implemented | Correct (C1) | Measured, 1 node | Measured, 2 nodes |
|---|---|---|---|---|---|
| Allgather (v1) | yes (`docs/Parallel_Plan_v1(1).md`, §3) | yes | yes (bit-identical to the serial code for P = 1, 2, 3, 4, 7) | yes (`amdahl/`) | no |
| Ring | table above | no | no | no | no |
| Ring with overlap | table above | no | no | no | no |
| 2D force decomposition | table above | no | no | no | no |
