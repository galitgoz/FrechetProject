"""
Enhanced Greedy Algorithm for Fréchet Decision Problem

This module implements an enhanced greedy algorithm for the Fréchet decision problem
using matrix partitioning and multiple path options within submatrices.
"""

import numpy as np
import itertools
from typing import List, Tuple
from .computations import euclidean_distance

Point = Tuple[float, float]
Curve = List[Point]


def free_space_matrix(P: Curve, Q: Curve, epsilon: float) -> np.ndarray:
    """
    Construct a boolean matrix indicating which point pairs are within epsilon distance.

    Args:
        P, Q: Input curves as lists of (x, y) points
        epsilon: Distance threshold

    Returns:
        Boolean matrix F where F[i,j] = True if distance(P[i], Q[j]) <= epsilon
    """
    n, m = len(P), len(Q)
    F = np.zeros((n, m), dtype=bool)
    for i in range(n):
        for j in range(m):
            F[i, j] = euclidean_distance(P[i], Q[j]) <= epsilon
    return F


def partition_indices(n: int, k: int) -> List[Tuple[int, int]]:
    """
    Partition range [0, n) into k approximately equal-sized blocks.

    Args:
        n: Total number of elements
        k: Number of partitions

    Returns:
        List of (start, end) index pairs for each partition
    """
    sizes = [n // k + (1 if x < n % k else 0) for x in range(k)]
    idx = [0]
    for s in sizes:
        idx.append(idx[-1] + s)
    return [(idx[i], idx[i+1]) for i in range(k)]


def enhanced_greedy_decision(P: Curve, Q: Curve, epsilon: float, k: int = 4) -> bool:
    """
    Enhanced greedy decision algorithm for Fréchet distance threshold.

    Partitions the free space diagram into k×k submatrices and tries all possible
    combinations of diagonal vs edge traversal within each submatrix.

    Args:
        P, Q: Input curves as lists of (x, y) points
        epsilon: Distance threshold to test
        k: Number of partitions (k×k grid)

    Returns:
        True if Fréchet distance <= epsilon, False otherwise
    """
    n, m = len(P), len(Q)
    F = free_space_matrix(P, Q, epsilon)

    # Check if start and end points are reachable
    if not F[0, 0] or not F[n-1, m-1]:
        return False

    row_blocks = partition_indices(n, k)
    col_blocks = partition_indices(m, k)

    # Try all combinations of path options for each block
    # 0 = diagonal traversal, 1 = edge traversal
    num_blocks = k * k
    block_options = list(itertools.product([0, 1], repeat=num_blocks))

    for option in block_options:
        if _try_path_option(F, row_blocks, col_blocks, option, k):
            return True

    return False


def _try_path_option(F: np.ndarray, row_blocks: List[Tuple[int, int]],
                    col_blocks: List[Tuple[int, int]], option: Tuple[int, ...], k: int) -> bool:
    """
    Try a specific path option through the partitioned matrix.

    Args:
        F: Free space matrix
        row_blocks, col_blocks: Partition boundaries
        option: Tuple specifying traversal mode for each block (0=diagonal, 1=edge)
        k: Number of partitions

    Returns:
        True if this path option reaches the target, False otherwise
    """
    n, m = F.shape
    reachable = np.zeros_like(F, dtype=bool)
    reachable[0, 0] = True

    for block_idx, (rb, cb) in enumerate(itertools.product(range(k), range(k))):
        r0, r1 = row_blocks[rb]
        c0, c1 = col_blocks[cb]

        if r0 >= n or c0 >= m:
            continue

        mode = option[block_idx]

        if mode == 0:
            # Diagonal traversal
            _traverse_diagonal(F, reachable, r0, r1, c0, c1, n, m)
        else:
            # Edge traversal (rightmost column and bottom row)
            _traverse_edge(F, reachable, r0, r1, c0, c1, n, m)

    return reachable[n-1, m-1]


def _traverse_diagonal(F: np.ndarray, reachable: np.ndarray,
                      r0: int, r1: int, c0: int, c1: int, n: int, m: int) -> None:
    """Traverse along the main diagonal of a submatrix."""
    for d in range(min(r1 - r0, c1 - c0)):
        i, j = r0 + d, c0 + d
        if i < n and j < m and F[i, j]:
            # Check if reachable from diagonal predecessor or if starting point
            if (i > 0 and j > 0 and reachable[i-1, j-1]) or (i == 0 and j == 0):
                reachable[i, j] = True


def _traverse_edge(F: np.ndarray, reachable: np.ndarray,
                  r0: int, r1: int, c0: int, c1: int, n: int, m: int) -> None:
    """Traverse along the rightmost column and bottom row of a submatrix."""
    # Traverse rightmost column
    j = c1 - 1
    if j < m:
        for i in range(r0, r1):
            if i < n and F[i, j]:
                if (_is_reachable_from_neighbors(reachable, i, j) or
                    (i == 0 and j == 0)):
                    reachable[i, j] = True

    # Traverse bottom row
    i = r1 - 1
    if i < n:
        for j in range(c0, c1):
            if j < m and F[i, j]:
                if (_is_reachable_from_neighbors(reachable, i, j) or
                    (i == 0 and j == 0)):
                    reachable[i, j] = True


def _is_reachable_from_neighbors(reachable: np.ndarray, i: int, j: int) -> bool:
    """Check if position (i,j) is reachable from valid neighboring positions."""
    return ((i > 0 and reachable[i-1, j]) or
            (j > 0 and reachable[i, j-1]) or
            (i > 0 and j > 0 and reachable[i-1, j-1]))


def enhanced_greedy_etd(P: Curve, Q: Curve, k: int = 4,
                       tolerance: float = 1e-3, max_iterations: int = 50) -> float:
    """
    Compute the Enhanced Trajectory Distance (ETD) using binary search.

    Args:
        P, Q: Input curves as lists of (x, y) points
        k: Number of partitions for the enhanced greedy algorithm
        tolerance: Convergence tolerance for binary search
        max_iterations: Maximum number of binary search iterations

    Returns:
        Estimated Fréchet distance (ETD)
    """
    if len(P) == 0 or len(Q) == 0:
        raise ValueError("Curves must have at least one point each.")

    # Binary search bounds
    d_lo = 0.0
    d_hi = max(euclidean_distance(p, q) for p in P for q in Q)

    for _ in range(max_iterations):
        d_mid = (d_lo + d_hi) / 2.0

        if enhanced_greedy_decision(P, Q, d_mid, k):
            d_hi = d_mid
        else:
            d_lo = d_mid

        if d_hi - d_lo < tolerance:
            break

    return d_hi


def compute_approximation_quality(etd: float, exact_distance: float) -> float:
    """
    Compute the approximation quality ratio ETD/d_F.

    Args:
        etd: Enhanced Trajectory Distance
        exact_distance: Exact Fréchet distance

    Returns:
        Approximation ratio (ETD / exact_distance)
    """
    if exact_distance == 0:
        return np.inf if etd > 0 else 1.0
    return etd / exact_distance


def is_decision_correct(etd: float, exact_distance: float, threshold: float,
                       tolerance: float = 1e-6) -> bool:
    """
    Check if the enhanced greedy decision is correct for a given threshold.

    Args:
        etd: Enhanced Trajectory Distance
        exact_distance: Exact Fréchet distance
        threshold: Distance threshold for decision problem
        tolerance: Numerical tolerance for comparisons

    Returns:
        True if decision is correct, False otherwise
    """
    etd_decision = etd <= threshold + tolerance
    exact_decision = exact_distance <= threshold + tolerance
    return etd_decision == exact_decision
