# v2: the ring decomposition — summary (15 Nov 2026)

Plan §8.1, iteration 7b. It was added to `code/ParallelImplementation.py` alongside v1, selected with `--decomposition ring`. The default is still `allgather` (v1), unchanged and still bit-identical to the serial code.

## How it works
- **Communicators.**
  - `MPI.COMM_WORLD.Clone()` (7a), used by both decompositions.
  - The ring adds `Create_cart([P], periods=[True], reorder=False)`. With `reorder=False`, rank r still owns block r.
  - `Shift(0, 1)` gives the left neighbour (receive from) and the right neighbour (send to).
- **Ownership.** Each rank keeps only its own positions. `state.positions` is `None` during the run. A copy of the own block travels around the ring: in round s, rank r holds the block of rank (r − s) mod P. After P − 1 transfers, every rank has seen every block once.
- **`--ring-comm overlap`** (default): `Irecv` the next block and `Isend` the current one, compute on the current one, then `Waitall`. **`--ring-comm blocking`**: `Sendrecv` after each block, for comparison.
- **Kernel.** `block_accelerations(i_block, j_block)` uses the same formula as `serial.compute_accelerations`, with global indices. The self-force is excluded where the blocks overlap, and close pairs (i, j > i) are still found once, by the owner of i.
- **Merging.**
  - The merge path gathers positions and velocities into temporary arrays, then runs the serial `merge_stars` on every rank.
  - The pair order differs from the serial run, but union-find doesn't depend on it.
  - Afterwards the state is rebuilt without full positions.
- **Timing sections.**
  - `communication` = exposed ring traffic (posting + `Waitall`, or `Sendrecv`).
  - `forces` = the kernel.
- **Results columns.** `decomposition` = `ring_overlap` / `ring_blocking`, plus the new `kernel_tile_rows`.

## `--tile-rows K` (both decompositions)
- K own stars per loop iteration, using 2-D broadcasting. 1 = the v1 kernel (default); 0 = automatic (about 2^18 pairs per tile, ~8 MB temporaries).
- **Why it matters for the ring:** with K = 1, each rank loops over its rows once per block, so the Python loop overhead (the a·N term) is **not** divided by P. At N = 1000, P = 2 (laptop):

  | Variant | ms/step |
  |---|---|
  | allgather | 32.6 |
  | ring | 39.8 |
  | ring, tiled | 23.7 |
  | allgather, tiled | 25.4 |
  | serial | 59.6 |

- The serial baseline always uses the v1 kernel, so tiled speed-ups include the kernel gain (about 2× at small N). Compare decompositions with the same kernel:
  - v1 vs. v2-ring
  - v1-tile vs. v2-ring-tile

## Validation (local, MPICH)
- **Setups covered:**
  - P = 1, 2, 3, 4, 7
  - N = 300, 1000, and 5 (N < P)
  - with and without merging
  - overlap and blocking
  - tile 1 / 7 / auto
  - a 46-star test with 9 merges happening partway through the run
- **Results:**
  - allgather: bit-identical (tiled too, in these tests).
  - ring: same merge steps, same `merged_from`; relative differences ≤ 2e-14.
  - Time limit, `--plot` and the results row work in ring mode.

## Job scripts
- **`batch.sh`/`submit.sh`:**
  - New `MPI_ARGS` (passed to the MPI run only), e.g. `MPI_ARGS="--decomposition ring --tile-rows 0"`.
  - Validate mode now prints relative position/velocity differences, checks that the merges are the same, and reports PASSED/FAILED against `TOL` (default 1e-9).
- **isoeff:** new setting `mpi_args`, e.g. `./isoeff/start.sh v2-ring --mpi-args "--decomposition ring"`.

## Next
- On the HPC:
  1. `./submit.sh N=1000 MODE=validate MPI_ARGS="--decomposition ring"`.
  2. Then four campaigns: v1, v2-ring, v1-tile, v2-ring-tile.
- Expect the ring to pay off mainly across nodes (memory per rank, overlap). On one node, v1's communication is already about 0.04 % of runtime.
