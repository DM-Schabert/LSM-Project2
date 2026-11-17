"""Pure Python helpers for the N-body communication layout (no MPI needed)."""


def block_partition(n_stars: int, ranks: int) -> tuple[list[int], list[int]]:
    """Return local star counts and 0-based star displacements for each rank.

    Works when N is not divisible by P, including N < P.
    """
    if n_stars < 0 or ranks < 1:
        raise ValueError("n_stars must be >= 0 and ranks must be >= 1")
    q, r = divmod(n_stars, ranks)
    counts = [q + (i < r) for i in range(ranks)]
    displacements = [0] * ranks
    for i in range(1, ranks):
        displacements[i] = displacements[i - 1] + counts[i - 1]
    return [int(v) for v in counts], displacements
