# Gaia N-body: direct-summation HPC project

Oct 27, 2026 · @Pelle Martin Hesketh

This project simulates gravity between the 626k Gaia DR3 stars by exact direct summation, with no grid. Every time step needs communication between ranks, so it is not embarrassingly parallel. It is built from the week 7 (communicators) and week 8 (scaling, custom datatypes) lectures of 02616.

## Setup

**Dataset:** [Gaia 626k stars from DR3 (Data release 3) V2](https://www.kaggle.com/datasets/realkiller69/gaia-stars-dataset-from-dr3-data-release-3) on Kaggle. Check its columns first; the plan below assumes right ascension, declination and parallax, and uses proper motions and radial velocities if present.

**Preprocessing (serial, once):**

- Keep stars with positive parallax and, if the column exists, `parallax_over_error > 5`.
- Distance in parsec = 1000 / parallax (mas). Convert (ra, dec, distance) to heliocentric Cartesian x, y, z.
- Velocities: with proper motions and radial velocity, build full 3D velocities. Without radial velocity, start every star at rest (a cold collapse). That is still a valid computational test.
- Masses are not in the data: give every star 1 solar mass. Estimating masses from colour and brightness is an optional extra.
- Units: parsec, million years (Myr), solar masses, so G = 4.50 × 10⁻³ pc³ M☉⁻¹ Myr⁻².

**Physics framing:** the stars around the Sun are not a self-gravitating system, and their mutual gravity is too weak to show up over Gaia's lifetime. Gaia has no later snapshot to compare against. The data gives realistic, strongly clustered starting positions; the scientific goal is computational, not astrophysical. Optionally, add a fixed analytic Milky Way potential; it is evaluated per star and needs no communication.

**Validation:**

- Two-body orbit against its exact analytic solution.
- A Plummer sphere (a standard model cluster) should stay in equilibrium.
- Total energy and momentum conserved within the integrator's error.
- The parallel result matches the serial result, apart from summation order.

**Reporting:** strong and weak scaling plots as described in the [HPC wiki scaling guide](https://hpc-wiki.info/hpc/Scaling) from the week 8 slides, with an Amdahl fit for the serial fraction and a breakdown of time per stage.

## Project: gridless direct-summation N-body

Every star feels every other star, computed exactly with no grid. Each step, every rank needs the current positions of all stars, so the project is about moving those positions efficiently: three communication schemes compared, ending with a 2D decomposition on row and column sub-communicators.

### One time step

1. Each rank owns N/P stars.
2. Compute the acceleration of each star from all others, with softening length ε to avoid infinite forces in close encounters.
3. Push with kick–drift–kick leapfrog; the time step is the global minimum from an `Allreduce`.
4. Diagnostics: kinetic and potential energy, the latter accumulated during the force loop, summed with `Allreduce`.

```latex
\mathbf{a}_i = \sum_{j \neq i} \frac{G m_j (\mathbf{r}_j - \mathbf{r}_i)}{\left( |\mathbf{r}_j - \mathbf{r}_i|^2 + \varepsilon^2 \right)^{3/2}}
```

### Three communication schemes to compare

1. **Baseline, `Allgather`:** every rank gathers all positions each step. Simple, but each rank holds and receives all N positions.
2. **Ring:** a periodic 1D Cartesian communicator, `Create_cart([P], periods=[True])`, with neighbours from `Shift(0, 1)`. A block of positions travels around the ring in P − 1 `Sendrecv_replace` steps; each rank computes forces from the block it holds. Memory per rank drops to N/P. Overlap the next transfer with the current computation using `Isend`/`Irecv`.
3. **2D force decomposition:** `Compute_dims(P, 2)` gives a √P × √P grid. Rank (i, j) computes the forces on star block i from star block j.
   - `Split` by row and by column (slide 3): block j's positions are broadcast down its column communicator, block i's along its row.
   - The partial forces on block i are summed with `row_comm.Allreduce`.
   - Data received per rank shrinks from about N to about N/√P. This is the slide 3 `y = A·x` pattern, with the force matrix in place of A.

&#91;embedded content: 2D force decomposition · 3 x 3 ranks, one row and one column communicator highlighted\]

Rank (1,2) receives block 2's positions through its column and contributes its partial forces on block 1 to its row's `Allreduce`; no rank ever holds all N positions.

**Why 2D, not 3D?** The 2D grid splits the N × N table of star pairs (forces on i from j), not physical space; the 3D positions only appear inside each force. Splitting space into 3D boxes would not help: gravity has no cutoff, so every box still needs every other box's stars, and Gaia's clustering would make the boxes' star counts very uneven. A third rank dimension would need a third loop index, as in matrix multiplication or three-body forces. Copying positions c times on a (P/c) × c grid spans the range from the ring (c = 1) to the 2D scheme (c = √P); varying c is an optional experiment on the memory–communication trade-off.

Optional: use Newton's third law to halve the arithmetic, at the cost of sending forces mmunication.

**Validation:**

- Two-body orbit against its exact analytic solution.
- A Plummer sphere (a standard model cluster) should stay in equilibrium.
- Total energy and momentum conserved within the integrator's error.
- The parallel result matches the serial result, apart from summation order.

**Reporting:** strong and weak scaling plots as described in the [HPC wiki scaling guide](https://hpc-wiki.info/hpc/Scaling) from the week 8 slides, with an Amdahl fit for the serial fraction and a breakdown of time per stage.

## Project: gridless direct-summation N-body

Every star feels every other star, computed exactly with no grid. Each step, every rank needs the current positions of all stars, so the project is about moving those positions efficiently: three communication schemes compared, ending with a 2D decomposition on row and column sub-communicators.

### One time step

1. Each rank owns N/P stars.
2. Compute the acceleration of each star from all others, with softening length ε to avoid infinite forces in close encounters.
3. Push with kick–drift–kick leapfrog; the time step is the global minimum from an `Allreduce`.
4. Diagnostics: kinetic and potential energy, the latter accumulated during the force loop, summed with `Allreduce`.

### Three communication schemes to compare

1. **Baseline, ****`Allgather`****:** every rank gathers all positions each step. Simple, but each rank holds and receives all N positions.
2. **Ring:** a periodic 1D Cartesian communicator, `Create_cart([P], periods=[True])`, with neighbours from `Shift(0, 1)`. A block of positions travels around the ring in P − 1 `Sendrecv_replace` steps; each rank computes forces from the block it holds. Memory per rank drops to N/P. Overlap the next transfer with the current computation using `Isend`/`Irecv`.
3. **2D force decomposition:** `Compute_dims(P, 2)` gives a √P × √P grid. Rank (i, j) computes the forces on star block i from star block j.
   - `Split` by row and by column (slide 3): block j's positions are broadcast down its column communicator, block i's along its row.
   - The partial forces on block i are summed with `row_comm.Allreduce`.
   - Data received per rank shrinks from about N to about N/√P. This is the slide 3 `y = A·x` pattern, with the force matrix in place of A.

&#91;embedded content\]

Rank (1,2) receives block 2's positions through its column and contributes its partial forces on block 1 to its row's `Allreduce`; no rank ever holds all N positions.

**Why 2D, not 3D?** The 2D grid splits the N × N table of star pairs (forces on i from j), not physical space; the 3D positions only appear inside each force. Splitting space into 3D boxes would not help: gravity has no cutoff, so every box still needs every other box's stars, and Gaia's clustering would make the boxes' star counts very uneven. A third rank dimension would need a third loop index, as in matrix multiplication or three-body forces. Copying positions c times on a (P/c) × c grid spans the range from the ring (c = 1) to the 2D scheme (c = √P); varying c is an optional experiment on the memory–communication trade-off.

Optional: use Newton's third law to halve the arithmetic, at the cost of sending forces back.

### Custom datatypes

- Stars are stored aback.

### Custom datatypes

- Stars are stored as a NumPy structured record (int64 ID, 3 positions, 3 velocities, mass). To send only position and mass from full records without packing, build a `Create_struct` over those four fields at their byte offsets, then `Create_resized` to the record size (slides 13–14). MPI then skips the velocities and ID in place.
- Simpler alternative for comparison: a separate position-and-mass array with `MPI.DOUBLE.Create_contiguous(4)`, the slide 11 "xyz" idea.
- When N is not divisible by P, blocks differ in size: receivers check `status.Get_count` (slide 15).

### Why it is not embarrassingly parallel

Every star's acceleration depends on the current positions of all other stars. That all-to-all dependency recurs every time step.

### Experiments

- **Cost:** work grows as N². All 626k stars mean 3.9 × 10¹¹ pairs per step, so time steps on subsamples of 50k to 200k stars and use Numba or tiled NumPy for the force loop.
- **Strong scaling:** fixed N (for example 100k), P = 1 up to the allocation, all three schemes. Expect `Allgather` and the ring to hit a communication floor, since each rank still receives all N positions; the 2D scheme should scale further.
- **Weak scaling:** with N² work, constant work per rank means N grows as √P, not P. Doubling the ranks adds only 41% more stars; discuss this against Gustafson's law.
- **Load balance:** equal star counts give equal work, so balance is trivial here.
- **Ensembles (slide 4):** once the 2D scheme stops scaling, run several random Gaia subsamples or softening lengths on `Split` sub-communicators.

### Additional validation

- Forces must agree across all three schemes and P values, apart from summation order.

### Milestones

1. Serial force loop with Numba; pass the two-body and Plummer tests.
2. `Allgather` baseline with the struct datatype.
3. Ring with overlap.
4. 2D force decomposition with row and column communicators.
5. Scaling runs comparing the three schemes; report.

### Risks

- Pure Python loops are far too slow: plan on Numba from the start.
- N² cost limits how many stars and steps fit in the allocation; agree on subsample sizes early.
- Close encounters need either a small time step or a larger ε; fix ε before timing runs.

## Sources

- [Gaia 626k stars from DR3 (Data release 3) V2](https://www.kaggle.com/datasets/realkiller69/gaia-stars-dataset-from-dr3-data-release-3), Kaggle
- [Scaling](https://hpc-wiki.info/hpc/Scaling), HPC wiki, referenced in the week 8 slides
- 02616 Large-scale Modelling, week 7 (Communicators) and week 8 (Scaling and data-types) slides
