# memory: what fills a rank, and which limit comes first

The memory limit test, in depth. How much memory does a rank need, what is it made of, and does N run into the memory limit before it runs into the time limit? The program itself only reports the peak of its largest rank (`peak_memory_mb`); here every rank is read after every phase of the start-up and the force kernel, from 1 000 stars up to the whole catalogue (589 993 stars after the subset filter), for Allgather, ring and 2D. One job on two nodes, about 15 minutes.

```bash
cd ~/.../code                                   # the folder with ParallelImplementation.py
bsub < memory/memory_batch.sh

# afterwards, on the laptop, from the repository root
ssh <user>@login.hpc.dtu.dk "cd <repo on the cluster> && tar czf - memory/out memory/logs" | tar xzvf -
python3 memory/make_page.py                     # fills memory/results.html
```

## What we know, and the hypothesis

- `peak_memory_mb` in C2/C3: 177–226 MB per rank on one node, 270–300 MB on two, at every N up to 20 000.
- Rank 0 reads the CSV; then **every rank gets the whole catalogue by `Bcast`**: positions, velocities (24 B each per star), masses, members, owner (8 B each), class (`<U1`, 4 B), guessed (1 B), so 77 B per star, also for the ring and 2D.
- After loading, `main()` drops the full positions for the ring and 2D; rank 0 keeps a copy of the start positions for `--plot`.
- The kernel's temporaries are 56 B per pair of the current tile (`tracemalloc`, one tile call); `--tile-rows 0` caps a tile at 2^18 pairs.

**Hypothesis.** Memory of a rank ≈ M0 (Python, NumPy, pandas, matplotlib, MPI) + N · b (the whole catalogue, b ≈ 77 B per star, whatever the scheme) + 56 B × pairs per tile + on rank 0 alone, pandas while reading the CSV. Then a rank has room for millions of stars, while time allows far fewer: **time is the limit, not memory.** Memory would come first only for a kernel without tiles (one tile for the whole own block).

**Alternatives the measurement must separate:** MPI's memory grows with the number of ranks (per-peer buffers); the ring or 2D need less at the peak than v1; rank 0 needs so much more that it alone sets the limit.

## Experiments

`memory_probe.py` calls the program's own functions, unchanged (`parse_args`, `load_and_distribute`, the ring's buffers, the 2D grid, `block_accelerations`), and does what `main()` does in between. It reads VmHWM (peak so far) and VmRSS (now) from `/proc/self/status` on every rank after each phase:

| Phase | After |
|---|---|
| `start` | the imports and `MPI_Init` |
| `connected` | one `Alltoall` of one number: every pair of ranks has exchanged a message |
| `loaded` | `load_and_distribute`: rank 0 read the CSV, every rank got the catalogue |
| `state` | the set-up of `main()`: rank 0's copy for `--plot`; ring and 2D drop the full positions; the ring's two buffers, the 2D column block |
| `kernel` | the force kernel: one tile (`--kernel tile`, M1) or the whole share of one time step (`--kernel block`, M2) |

| | Question | Settings |
|---|---|---|
| M3 | MPI's share against P | v1, N = 1 000, P = 1, 2, 4, 8, 12, 16, 24 (one node), 32, 48 (two nodes) |
| M1 | memory against N | v1, ring, 2D; P = 24; N = 1 000, 5 000, 20 000, 50 000, 100 000, 200 000, 500 000, 589 993 |
| M2 | the kernel against tile size | v1, N = 20 000, P = 24, whole share of a step, `--tile-rows` 0 (13 rows), 32, 104, 208, 417 |

The largest tile (417 rows × 20 000 = 8.3 M pairs, about 470 MB) stays below the reservation of 1 GB per slot: the job measures up to about half the limit and the page extrapolates with the measured model, instead of provoking a kill (a killed job saves nothing). The log records the node's memory (`free -m`), the LSF limits and, at the end, LSF's "Max Memory" of the whole job. The time per star pair for the time limit comes from the C2/C3 runs of v1 (`../results.csv`).

## Expectations (written before the first measurement)

The page applies exactly these rules to the data and prints the verdict. "Other ranks": the median of ranks 1 … P − 1.

| # | Expectation | Rule for "yes" |
|---|---|---|
| 1 | Before any star, a rank needs 150–250 MB. | `start` on one node between 150 and 250 MB (median over P). |
| 2 | MPI needs little per peer on one node; a second node adds 50–100 MB per rank (the network transport). | One node: (`connected` − `start`) / (P − 1) at most 1 MB (median over P ≥ 2); `connected` at P = 48 minus P = 24 between +50 and +100 MB. |
| 3 | Every rank holds the whole catalogue after the `Bcast`, about 77 B per star, whatever the scheme. | Slope of (`loaded` − `connected`) peak against N between 70 and 110 B per star for v1, ring and 2D, R² ≥ 0.99, ring and 2D within ±10 % of v1. |
| 4 | After the set-up, v1 holds about 55 B per star, the ring about 33 and 2D about 43. | Slope of (`state` − `connected`) VmRSS against N within ±20 % of 55 / 33 / 43. |
| 5 | Rank 0 needs 150–600 B per star more while pandas reads the CSV. | Rank 0's `loaded` peak minus the other ranks', per star, at the largest N, between 150 and 600 B. |
| 6 | The kernel's temporaries are 56 B per pair of the tile, linear in the tile size. | Slope of (`kernel` peak − `state` VmRSS) against pairs per tile between 42 and 70 B, R² ≥ 0.99 (M2). |
| 7 | Time comes first: even rank 0 has room for at least 3 times as many stars as one time step in 30 minutes allows at P = 24. | N at which rank 0 fills 1 GB (model of 1, 3, 5) ≥ 3 × the N of one step in 30 min at P = 24 (time per pair from C2/C3). |
| 8 | Without tiles, memory would come first: a kernel on the whole own block fills 1 GB per rank at P = 24 from 10 000–30 000 stars, below our largest run. | N for 1 GB per rank with the naive kernel (model of 1 and 6) between 10 000 and 30 000. |
| 9 | LSF's Max Memory of the job agrees with the sum over the ranks. | Max Memory between 0.6 and 1.1 times the largest sum of the ranks' peaks in one launch. |

Expectation 4 also depends on whether the memory allocator gives freed arrays back to the system; if 3 holds and 4 does not, that is the first suspect.

## Commands

| What | Command |
|---|---|
| run | `bsub < memory/memory_batch.sh` |
| only some N | `sed 's/^M1_N=.*/M1_N="1000 20000 589993"/' memory/memory_batch.sh \| bsub` |
| progress | `bjobs`, `tail -f memory/logs/memory_<job>.out` |
| results page | `python3 memory/make_page.py [--job <job>] [--reservation-mb 1024]`, then open `memory/results.html` |
| one probe by hand | `mpirun -np 4 python3 memory/memory_probe.py --experiment T --scheme ring --data gaia_subsets/gaia_nearest_1000.csv --kernel block` |

## Output

- `out/memory_<job>.csv`: one row per launch, rank and phase: experiment, scheme, N, P, nodes, host, tile rows, kernel tile, VmHWM, VmRSS, seconds
- `logs/memory_<job>.out`: node memory, limits, every launch in one line, and LSF's job summary with "Max Memory" at the end

## Results page

`results.html` is in the same style as `isoeff/`, `amdahl/`, `gustafson/` and `communication/results.html`, and stays empty until `make_page.py` fills its data block: tiles (memory before any star, bytes per star, rank 0's extra, memory against time), memory against N (peak or held after the set-up), every phase for one scheme and N, the kernel's temporaries against tile size, MPI's share against P, which limit comes first, the expectations with their verdicts, and every launch. The charts load d3 from cdnjs, so the page needs an internet connection.

## Result (job 30267003, n-62-21-77 and n-62-21-47, 257 GB per node)

All 38 launches finished in 3 minutes. Verdicts on `results.html`: 7 of 9 yes (6 and 9 no).

- **Before any star: 187 MB per rank** (Python, NumPy, pandas, matplotlib, mpi4py). MPI needs almost nothing per peer on one node (0–34 KB); the first contact with the other node adds **76 MB per rank** (the network transport), which is the step from about 220 to 270–300 MB seen in C3. A single rank (P = 1) starts at 249 MB.
- **Every rank holds the whole catalogue: 85–88 B per star**, the same for v1, ring and 2D (R² 0.999). The ring's and 2D's smaller share does not show in the peak.
- **After the set-up** v1 holds 58, the ring 36 and 2D 46 B per star, as predicted from the code (55 / 33 / 43): the ring and 2D save only after the peak.
- **Rank 0 needs 321 B per star more while pandas reads the CSV**, so it sets the limit: 418 MB at 589 993 stars against 275 MB on the other ranks.
- **The kernel needs 79 B per pair, not 56 (expectation 6: no).** `tracemalloc` of one tile call alone gives 56 B; with several tiles per call the previous tile's arrays stay alive until the next tile's arrays replace them (`tracemalloc`: 56.1 B for one tile, 80.1 B from two tiles on). A `del` at the end of the tile loop in `block_accelerations` would take this back to 56 B.
- **Which limit comes first:** with 1 GB per rank (the reservation), rank 0 fills it at 2.2 M stars, the other ranks at 10.3 M; with the node's memory (10.7 GB per rank) at 27 M. Time allows 607 k stars for one step in 30 minutes at P = 24 (859 k at P = 48), and only 2 479 for 10 000 steps in 300 s. **Time comes first**, by ×3.6 in the most generous case and by orders of magnitude for a real run.
- **Without tiles memory would come first:** one tile for the whole own block fills 1 GB per rank at P = 24 from about 16 000 stars, below our 20 000-star runs.
- **LSF's "Max Memory" (2.5 GB) missed the real peak** (about 20 GB summed over the 24 ranks at `--tile-rows 417`; expectation 9: no): LSF samples the memory every few seconds, and each launch took about 2 s. It cannot be used to find short peaks.
