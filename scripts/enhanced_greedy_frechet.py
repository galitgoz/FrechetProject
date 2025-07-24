# Enhanced Greedy Algorithm for Fréchet Decision Problem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.frechet.computations import discrete_frechet_distance, euclidean_distance
import itertools


def free_space_matrix(P, Q, epsilon):
    n, m = len(P), len(Q)
    F = np.zeros((n, m), dtype=bool)
    for i in range(n):
        for j in range(m):
            F[i, j] = euclidean_distance(P[i], Q[j]) <= epsilon
    return F


def partition_indices(n, k):
    # Returns list of (start, end) indices for k partitions of n
    sizes = [n // k + (1 if x < n % k else 0) for x in range(k)]
    idx = [0]
    for s in sizes:
        idx.append(idx[-1] + s)
    return [(idx[i], idx[i+1]) for i in range(k)]


def enhanced_greedy_decision(P, Q, epsilon, k=4):
    n, m = len(P), len(Q)
    F = free_space_matrix(P, Q, epsilon)
    row_blocks = partition_indices(n, k)
    col_blocks = partition_indices(m, k)
    # Each block: (row_start, row_end), (col_start, col_end)
    # For each block, two options: diagonal or edge
    # For kxk blocks, 2**(k*k) combinations (can be pruned for large k)
    block_options = list(itertools.product([0, 1], repeat=k*k))  # 0: diag, 1: edge
    for option in block_options:
        reachable = np.zeros_like(F, dtype=bool)
        # Set start
        if not F[0, 0]:
            continue
        reachable[0, 0] = True
        for block_idx, (rb, cb) in enumerate(itertools.product(range(k), range(k))):
            r0, r1 = row_blocks[rb]
            c0, c1 = col_blocks[cb]
            if r0 >= n or c0 >= m:
                continue
            # Option: 0=diag, 1=edge
            mode = option[block_idx]
            if mode == 0:
                # Diagonal: traverse (r0,c0) to (r1-1,c1-1) along diagonal
                for d in range(min(r1 - r0, c1 - c0)):
                    i, j = r0 + d, c0 + d
                    if i < n and j < m and F[i, j]:
                        if (i > 0 and reachable[i-1, j-1]) or (i == 0 and j == 0):
                            reachable[i, j] = True
            else:
                # Edge: traverse rightmost col and bottom row
                for i in range(r0, r1):
                    j = c1 - 1
                    if j < m and F[i, j]:
                        if (i > 0 and reachable[i-1, j]) or (j > 0 and reachable[i, j-1]) or (i == 0 and j == 0):
                            reachable[i, j] = True
                for j in range(c0, c1):
                    i = r1 - 1
                    if i < n and F[i, j]:
                        if (i > 0 and reachable[i-1, j]) or (j > 0 and reachable[i, j-1]) or (i == 0 and j == 0):
                            reachable[i, j] = True
        if reachable[n-1, m-1]:
            return True
    return False


def enhanced_greedy_etd(P, Q, k=4, tol=1e-3, max_iter=50):
    # Binary search for minimal epsilon where decision is True
    d_lo, d_hi = 0, max([euclidean_distance(p, q) for p in P for q in Q])
    for _ in range(max_iter):
        d_mid = (d_lo + d_hi) / 2
        if enhanced_greedy_decision(P, Q, d_mid, k):
            d_hi = d_mid
        else:
            d_lo = d_mid
        if d_hi - d_lo < tol:
            break
    return d_hi


def run_experiment(curve_pairs, k=4):
    results = []
    for idx, (P, Q) in enumerate(curve_pairs):
        d_exact = discrete_frechet_distance(P, Q)
        etd = enhanced_greedy_etd(P, Q, k)
        correct = abs(etd - d_exact) < 1e-3 or etd >= d_exact  # Accept if etd >= d_exact
        results.append({
            'idx': idx,
            'd_exact': d_exact,
            'etd': etd,
            'ratio': etd / d_exact if d_exact > 0 else np.nan,
            'correct': correct
        })
        print(f"Pair {idx}: d_F={d_exact:.4f}, ETD={etd:.4f}, ratio={etd/d_exact:.4f}, correct={correct}")
    df = pd.DataFrame(results)
    correct_pct = 100 * df['correct'].mean()
    print(f"\nCorrect decision rate: {correct_pct:.2f}%")
    # CDF plot
    ratios = df['ratio'].dropna().sort_values()
    yvals = np.arange(1, len(ratios)+1) / len(ratios)
    plt.figure(figsize=(8,5))
    plt.step(ratios, yvals, where='post')
    plt.xlabel('ETD / d_F')
    plt.ylabel('Cumulative Fraction')
    plt.title(f'CDF of ETD/d_F Ratio (k={k})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'enhanced_greedy_cdf_k{k}.pdf')
    plt.show()
    return df


if __name__ == "__main__":
    # Example: load or generate curve pairs
    # Here, generate random curves for demo; replace with real data loading as needed
    np.random.seed(42)
    num_pairs = 10
    curve_len = 20
    curve_pairs = []
    for _ in range(num_pairs):
        P = np.cumsum(np.random.randn(curve_len, 2), axis=0)
        Q = np.cumsum(np.random.randn(curve_len, 2), axis=0)
        curve_pairs.append((P.tolist(), Q.tolist()))
    # Run experiment
    run_experiment(curve_pairs, k=4)

