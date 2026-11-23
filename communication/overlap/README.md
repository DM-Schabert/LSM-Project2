# overlap: why the ring's overlap does not help

The communication part's topic in depth. The ring can send the next block with `Isend`/`Irecv` and compute while it travels. We expected that to pay off where communication is a real share of the step: at small N or across two nodes. C4 says it does not: on one node no gain at any N, on two nodes 7–10 % slower at N ≤ 1 000 and the same at N ≥ 5 000. This folder finds out why, with two ranks and one message at a time, and checks the answer against the ring runs of C2 and C3. One job on two nodes, about 5 minutes.

```bash
cd ~/.../code                                          # the folder with ParallelImplementation.py
bsub < communication/overlap/overlap_batch.sh

# afterwards, on the laptop
scp '<cluster>:<repo>/communication/overlap/out/overlap_<job>.csv' communication/overlap/out/
scp '<cluster>:<repo>/communication/overlap/logs/overlap_<job>.*' communication/overlap/logs/
python3 communication/overlap/make_page.py             # fills communication/overlap/results.html
```

## Hypothesis and the alternatives

**Hypothesis (progress).** Open MPI moves a message only while the program is inside an MPI call; nothing runs in the background (no progress thread). How a message travels depends on its size:

- **Small (eager):** the sender copies it out at `Isend`; it has arrived by the time the computation ends, so `Waitall` only returns.
- **Large (rendezvous):** the two ranks first exchange a handshake, and the data moves only after it. Both steps need MPI calls. While the rank computes with NumPy, nothing happens, so the whole transfer takes place in `Waitall`, after the computation: nothing is hidden.

The limit between the two is set by the transport (UCX or the shared-memory BTL) and is typically a few KB; the job log prints it. Calling `Testall` between the kernel tiles makes MPI calls during the computation and should hide large messages after all.

**And for our runs:** even a perfect overlap can save at most the transfer time, (P − 1) · T_comm per step. In C2 and C3 that is tiny against the computation at large N, so overlap could not have shown there. At small N the loss comes from the extra calls (`Irecv` + `Isend` + `Waitall` instead of one `Sendrecv`).

**Alternatives that the measurement must separate:**

- **Overlap works, but there is nothing to hide:** then the small and the large messages would both be hidden in the microbenchmark, where the computation is chosen long enough.
- **The non-blocking calls cost more than they save:** shows as extra time for small messages, independent of the size limit.
- **The transfer slows the computation:** the kernel would take longer while a message is in flight (both compete for the memory).

## Experiment

Two ranks, each sends m bytes to the other and receives m bytes, as two neighbours of the ring do in every round. In between, both compute with the real force kernel (`ParallelImplementation.block_accelerations`, unchanged) in tiles of 2 × 1 000 pairs.

- **Sizes:** 64 B to 4 MB, factors of 2. The ring's messages are ⌈N/P⌉ · 24 bytes: 2.5 KB at N = 5 000 and 10 KB at N = 20 000 (P = 48).
- **Modes per size:**
  - `comm`: `Sendrecv` alone, giving T_comm, the transfer
  - `compute`: the kernel alone, giving T_comp, set to about 2 · T_comm and at least 4 tiles
  - `blocking`: `Sendrecv`, then compute (the ring with `--ring-comm blocking`)
  - `nonblocking`: `Irecv` + `Isend`, compute, `Waitall` (the ring with overlap, as Pelle wrote it)
  - `polling`: the same, with `Testall` after every tile
- **Ranks:** both on one node (cores 0 and 1), and one on each of two nodes.
- **Repeats:** 30 iterations per size and mode, each after a barrier; the slower rank counts; the median. All sizes three times, the second time in reverse order; the page takes the median of the three.
- **Recorded:** the time for posting the requests, for computing and in `Waitall`. Also the transport and its eager and rendezvous limits (`ompi_info`, `ucx_info`, in the log).

**Measures.** Hidden share = 1 − (time in `Waitall`) / T_comm: 1 = the message had arrived when the computation ended, 0 = the whole transfer happened in `Waitall`. Extra cost = non-blocking minus blocking total time. The most overlap could ever save in a ring run = (P − 1) · T_comm(⌈N/P⌉ · 24 B) / step time; for runs on two nodes the pair across the nodes counts, because the slowest hop sets the pace of a round.

## Expectations (written before the first measurement)

The page applies exactly these rules to the data and prints the verdict.

| # | Expectation | Rule for "yes" |
|---|---|---|
| 1 | Small messages have arrived when the computation ends. | Median time in `Waitall` for sizes ≤ 1 KB at most 3 µs, on one node and on two. |
| 2 | Large messages are not hidden without progress calls: the whole transfer happens in `Waitall`. | For sizes ≥ 64 KB, at most 20 % of the transfer hidden (median), on one node and on two. |
| 3 | `Testall` between the kernel tiles hides them. | For sizes ≥ 64 KB with `Testall`, at least 70 % hidden (median), on one node and on two. |
| 4 | The switch lies at the eager/rendezvous limit, a few KB. | The smallest size from which less than half is hidden at every larger size lies between 1 KB and 64 KB, on one node and on two. Compared by hand with the limits in the log. |
| 5 | A transfer in the background does not slow the kernel down. | Kernel time during a non-blocking transfer ≥ 64 KB at most ×1.05 the kernel alone (median), on one node and on two. |
| 6 | For small messages, `Irecv` + `Isend` + `Waitall` cost a few µs more than one `Sendrecv`. | Non-blocking minus blocking for sizes ≤ 1 KB between 0 and 15 µs per exchange (median), on one node and on two. |
| 7 | In our runs overlap could not have shown at large N, even if it had worked. | For every ring run of C2 and C3 with N ≥ 5 000, (P − 1) · T_comm below 3 % of the step. |
| 8 | The loss of the overlap ring at N = 50 is the extra cost of the non-blocking calls. | The measured loss per hop in C2 and C3 (N = 50, P ≥ 8) between half and twice the extra cost of expectation 6 (median). |

If 1–4 hold, the mechanism is the missing progress for rendezvous messages, and the fix would be `Testall` between the tiles of the ring's kernel call. If 7 holds as well, that fix would not change our step times, and overlap only becomes worth it where (P − 1) · T_comm is a real share of the step. Only then is it worth adding a progress option to the ring in `ParallelImplementation.py` (Pelle's code; to be agreed with him) and measuring it in the program.

## Commands

| What | Command |
|---|---|
| run | `bsub < communication/overlap/overlap_batch.sh` |
| only one placement | `sed 's/^PLACEMENTS=.*/PLACEMENTS="inter"/' communication/overlap/overlap_batch.sh \| bsub` |
| more iterations | `sed 's/^BENCH_ARGS=.*/BENCH_ARGS="--reps 60"/' communication/overlap/overlap_batch.sh \| bsub` |
| progress | `bjobs`, `tail -f communication/overlap/logs/overlap_<job>.out` |
| results page | `python3 communication/overlap/make_page.py [--job <job>]`, then open `communication/overlap/results.html` |

## Output

- `out/overlap_<job>.csv`: one row per placement, sweep, message size and mode: total time (median, quartiles), posting, computing and waiting, tiles, `Testall` calls, hosts and CPUs
- `logs/overlap_<job>.out`: the transport and its limits, the bindings, every size in words

## Results page

`results.html` is in the same style as `isoeff/`, `amdahl/`, `gustafson/` and `../results.html` and stays empty until `make_page.py` fills its data block.

- **Tiles:** the switch size on one node and on two, the share hidden for large messages without and with `Testall`, and the most overlap could have saved in our runs.
- **Charts:** the hidden share against message size, and the transfer time against the time left in `Waitall`.
- **Tables:** what overlap could have saved in every ring run of C2 and C3 next to what C4 measured; the other explanations; the expectations with their verdicts; every point.

The charts load d3 from cdnjs, so the page needs an internet connection.
