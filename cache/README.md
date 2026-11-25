# cache: what the caches and the memory bandwidth do to the kernel and to communication

**Question.** Our results say that the computation per step is the limit, not the network and not the memory size. But the force kernel itself slows down when the node is full, and the schemes differ because they cut the kernel's work into pieces of different size. Is that the caches and the shared memory bandwidth? What happens when we overload them on purpose, and does communication suffer from it too?

**What we already know** (from `communication/`, `memory/` and the presentation):

- The tiled kernel (`--tile-rows 0`, about 2^18 pairs per tile) is 30 % faster than the untiled one on one rank, but on 24 ranks the two are equally fast. At N = 5 000 the tiled v1 keeps only **74 %** of its own one-rank speed on 24 ranks; the untiled v1 of `amdahl/` keeps **97 %**.
- One tile needs about **20 MB** of temporaries (79 B per pair, measured in `memory/`). A core's L2 has 256 KB; the 30 MB L3 of a socket is shared by 12 cores, about 2.5 MB each.
- The ring calls the kernel P times per step on pieces of (N/P)² pairs. Where those pieces fit into a core's share of the cache it beats v1 (N = 1 000 with 12–16 ranks, N = 5 000 with 48), elsewhere it loses. At P = 24 and P = 48 every socket holds 12 busy ranks, so the competition for memory is the same; only the pieces shrink.
- On one node, MPI moves a message by a CPU copy through shared memory; inside the program even the fastest rank of the ring spends 2 ms per step communicating, against 0.05–0.17 ms for the bare transfers measured in C0.

**Hypothesis.** The tiled kernel is memory-bound when every core works: its temporaries stream through RAM, and the 12 cores of a socket share one memory bandwidth (lecture 2, slides 11–14: "blocking algorithms", "all threads fight for the same memory bandwidth"). Tiles that fit a core's cache share avoid that. Intra-node communication uses the same memory bandwidth, so it should slow down under memory load as well; between nodes the network cards move the data and should suffer less.

```bash
# on the cluster, from the repository root (each job stands alone)
bsub < cache/cache_node.sh      # K0; K1, K2 at N = 5 000    one node,  ~20 min
bsub < cache/cache_node20k.sh   # K1, K2 at N = 20 000       one node,  ~15 min
bsub < cache/cache_bign.sh      # K3                         two nodes, ~20 min
bsub < cache/cache_comm.sh      # K4                         two nodes,  ~5 min
# every job asks for at most 30 min, like all our other hpcintro jobs

# afterwards, on the laptop
ssh <user>@login.hpc.dtu.dk "cd <repo on the cluster> && tar czf - cache/out cache/logs" | tar xzvf -
python3 cache/analyze.py        # tables, plots and the verdicts of the expectations below

# check that the scripts run (laptop, tiny sizes, nothing here is a measurement)
for j in cache_node cache_node20k cache_bign cache_comm; do LOCAL=1 bash cache/$j.sh; done
python3 cache/analyze.py --local
```

Nothing in the program is changed: K1–K3 run `ParallelImplementation.py` with its own options (`--tile-rows`, `--decomposition`) and `mpirun` placement options; K4 calls its kernel `block_accelerations` unchanged.

## Experiments

| | Question | Settings |
|---|---|---|
| **K0** | How much memory bandwidth do the cores get, alone and together? The reference for everything else. | NumPy copy and triad on 64 MB arrays per rank; P = 1, 2, 4, 6, 8, 12 on one socket; 2, 4, 8, 12 spread over both; 24 |
| **K1** | The kernel's "performance staircase": time per pair against tile size, alone and on a full node | v1, `--tile-rows` 1, 2, 4, 8, 16, 0 (auto), whole block; N = 5 000 at P = 1, 12 (one socket), 24; N = 20 000 with 1, 2, 4, 0, 64, whole block at P = 12, 24 |
| **K2** | Is it the shared bandwidth? Same 12 ranks on one socket against 6 + 6 on two | v1 at P = 12, `--map-by ppr:6:package` against K1's `ppr:12:package`; all N = 5 000 tiles, and 1, 0, 64 at N = 20 000 |
| **K3** | Do the schemes part when the positions themselves no longer fit the cache? | N = 100 000, 3 steps, one repeat: P = 48 on two nodes: v1 (tiles 0, 1), ring (0, 8), 2D (0, 1); P = 24 on one node: v1 0, ring 0 and 4, 2D 0 |
| **K4** | Does communication slow down under memory load? | ping-pong of 8 B … 4 MB (incl. the ring's 5 016 and 10 008 B blocks) between two cores of one socket (shared memory) and between two nodes (InfiniBand), while all other ranks are idle, run the kernel with ~1.6 MB tiles, with ~20 MB tiles, or a NumPy triad |

Tile sizes in MB, at 79 B per pair (one tile = rows × columns pairs):

| N | 1 row | 2 | 4 | 8 | 16 | 0 (auto) | 64 | whole block, P = 24 |
|---|---|---|---|---|---|---|---|---|
| 5 000 | 0.40 | 0.79 | 1.6 | 3.2 | 6.3 | 20.5 (52 rows) | — | 82 (209 rows) |
| 20 000 | 1.6 | 3.2 | 6.3 | — | — | 20.5 (13 rows) | 101 | 1 318 (834 rows) |

`--tile-rows 1` runs the serial kernel's row loop (a different code path); the rules below leave it out where tile size alone is compared. The whole block at N = 20 000, P = 12 needs about 2.6 GB per rank (32 GB on the node of 257 GB), so the node job reserves 4 GB per slot.

**Placement.** `--map-by ppr:12:package --bind-to core` fills socket 0 first (P ≤ 12 on one socket, P = 24 on both); `ppr:6:package` puts 6 ranks on each socket, as in the week 2 Jacobi lab (`compact`/`spread`). `--report-bindings` writes the actual cores to the `.err` file; `bandwidth.py` also records the socket of every rank.

**Measured quantity.** For K1–K3: the force kernel's time of the slowest rank per step (`forces_time_s / iterations`), divided by that rank's pairs, in ns per pair; also the step time and the share of communication. Repeats: 3 in the node job (outermost loop, so a drift of the node hits every setting alike), 1 in K3 (each step takes 25–55 s). The analysis takes medians.

## Expectations (written before the first measurement)

`analyze.py` applies exactly these rules and prints yes / no / n.a. for each.

| # | Expectation | Rule for "yes" |
|---|---|---|
| E0.1 | One core alone gets 3–15 GB/s. | Triad, STREAM counting, P = 1: 3–15 GB/s |
| E0.2 | One socket saturates long before 12 cores. | Node bandwidth with 12 ranks on one socket < 1.2 × with 6 |
| E0.3 | A second socket brings its own bandwidth. | 12 ranks spread ≥ 1.5 × 12 ranks on one socket |
| E1.1 | On one rank the tile size hardly matters: a single core has the bandwidth to itself. | P = 1, N = 5 000, tiles 2, 4, 8, 16, 0: largest / smallest time per pair ≤ 1.15 |
| E1.2 | On a full node, tiles that fit a core's L3 share are clearly faster. | P = 24, each N: fastest tile ≤ 2.5 MB (not the row loop) ≤ 0.87 × the 2^18-pair tile |
| E1.3 | With such tiles the kernel scales almost perfectly. | N = 5 000: time per pair at P = 1 / at P = 24 ≥ 0.9 for the best tile ≤ 2.5 MB (2^18-pair tiles: about 0.75 expected) |
| E1.4 | The whole block in one tile is the worst case. | P = 24, each N: whole block ≥ 2^18-pair tile per pair |
| E2.1 | Large tiles gain from the second socket's bandwidth. | P = 12, tiles 0: one socket / two sockets ≥ 1.10 (N = 5 000, and 20 000 if measured) |
| E2.2 | Small tiles do not. | P = 12, tiles 2, 4 (N = 5 000) and 1 (N = 20 000): one socket / two within 5 % |
| E3.1 | Even at N = 100 000 the network is not the limit. | Communication < 2 % of a step for v1 and 2D, < 10 % for the ring (its share is mostly waiting) |
| E3.2 | Positions out of the cache cost v1 little: they are read once per tile row, the temporaries many times. | v1, tiles 0, P = 24: N = 100 000 within 15 % of N = 20 000 (K1) per pair |
| E3.3 | At large N the ring wins once its pieces fit the cache, which v1's rows of N stars never can. | At each P: ring with small pieces < best of v1 and 2D with tiles 0, per pair |
| E4.1 | Intra-node messages slow down under memory load: the CPU copy competes for bandwidth. | One node, sizes ≥ 64 KB: one-way time under stream and under big tiles ≥ 1.3 × idle (median over sizes) |
| E4.2 | A load that stays in the cache does not hurt. | One node, sizes ≥ 64 KB: small tiles ≤ 1.15 × idle |
| E4.3 | Across nodes the network cards move the data, so memory load matters less. | Stream load, sizes ≥ 64 KB: slowdown across two nodes < on one node |
| E4.4 | Small messages are latency, not bandwidth. | Sizes ≤ 1 KB, every load, both placements: ≤ 1.2 × idle |

If E1.2, E1.3 and E2 hold, the 74 % efficiency of the tiled kernel is a memory-bandwidth effect, and the fix is a tile size of about 1 MB instead of 20 MB (a change of `TILE_TARGET_ELEMENTS`, to be agreed with the team). If E1.1 already fails, something else depends on the tile shape (the rows of 1 000 stars were slower per pair than rows of 5 000 even on one rank in C3); that is the first thing to look at then.

## K5: where the bytes go (hardware counters)

**Question.** K1-K4 infer from times that the 20 MB tiles stream through RAM and small tiles stay in the cache. K5 counts it: how many bytes per pair the kernel really takes from RAM (and from L3), for each tile size and shape, on 1, 12 and 24 ranks.

**Model (written before measuring).** The NumPy steps of one tile read about 104 B and write about 80 B per pair (`dis`, `dis**2`, the sum, `+ eps²`, `**-1.5`, `* masses`, `einsum`; the 80 B written are the 79 B of temporaries measured in `memory/`). If all of it went to RAM that would be ~184 B per pair, ~264 B if every write first reads its cache line (write-allocate). At 124 ns per pair that is 1.5-2.1 GB/s per rank, against the 2.6 GB/s a rank gets on a full node (K0).

```bash
bsub < cache/counters.sh                 # one node, ~15 min; the log says which counters this node allows
python3 cache/counters_analyze.py        # out/k5_counters.csv, out/report_k5.txt, out/plots/k5_bytes.png
```

`counters.sh` runs `kernel_bench.py` (the kernel alone, all ranks at once, for 12 s) under the best counters it finds: `likwid-perfctr` (groups MEM and L3), `perf stat -a` on the memory controllers, or `perf stat` per rank (last-level cache misses × 64 B, a lower bound). A start-up-only run is subtracted. Without counters it still measures the times of the shape sweep.

| # | Expectation | Rule for "yes" |
|---|---|---|
| E5.1 | The automatic 20 MB tiles on 24 ranks stream through RAM. | ≥ 100 B per pair from RAM (5 000 columns, 52 rows) |
| E5.2 | 2-row tiles (0.8 MB) stay in the cache. | ≤ 20 B per pair from RAM on 24 ranks |
| E5.3 | Alone, a core has the whole 30 MB L3, so even the 20 MB tile stays in it. | ≤ 30 B per pair from RAM on 1 rank |
| E5.4 | The shape effect (52 × 5 000 slower than 13 × 20 000, same size) is not RAM traffic. | At 20 MB per tile on 24 ranks: RAM bytes per pair within 20 % between shapes, time per pair not |

## Files

| File | What it does |
|---|---|
| `cache_node.sh` | LSF job, one node: K0, and K1, K2 at N = 5 000 |
| `cache_node20k.sh` | LSF job, one node: K1, K2 at N = 20 000 (the whole block needs 4 GB per slot) |
| `cache_bign.sh` | LSF job, two nodes: K3 (makes the N = 100 000 subset if missing) |
| `cache_comm.sh` | LSF job, two nodes: K4 |
| `common.sh` | sourced by the jobs: modules, placement options, one run of the program, copying the job's rows |
| `bandwidth.py` | K0: NumPy copy and triad on every rank at once; node and per-rank GB/s, socket of every rank |
| `comm_load.py` | K4: ping-pong between rank 0 and a partner while the other ranks run a load |
| `analyze.py` | tables, plots, verdicts; reads `out/`, writes `out/k*.csv`, `out/report.txt`, `out/plots/` |
| `counters.sh` | K5 job: finds the counters the node allows and runs the kernel under them |
| `kernel_bench.py` | K5: the kernel alone on every rank at once, for a fixed time; one row per run |
| `perf_rank.sh` | K5: `perf stat` around one rank, when only per-process counters are allowed |
| `counters_analyze.py` | K5: bytes per pair from RAM and L3, against the model; `out/k5_counters.csv`, `out/report_k5.txt`, `out/plots/k5_bytes.png` |

## Output (`cache/out/`)

- `runs_<job>.csv`: the program's results rows of K1–K3 (also in `results.csv`, named `cache_<K>_<scheme>_N<N>_P<P>_t<tiles>_<placement>_r<repeat>`)
- `bandwidth_<job>.csv`: K0, one row per placement, P and test
- `comm_load_<job>.csv`: K4, one row per placement, load and message size, with the work the load ranks did
- made by `analyze.py`: `k0_bandwidth.csv`, `k1_tiles.csv` (K1 and K2), `k3_bign.csv`, `k4_comm.csv`, `report.txt`, `plots/` (not in git)

Files of `LOCAL=1` runs carry `local` in their name, are ignored by git and left out of the analysis unless `--local` is given.
