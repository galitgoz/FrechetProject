import math
import numpy as np
import pandas as pd
from scipy.fft import fft
from typing import List, Tuple


Point = Tuple[float, float]
Curve = List[Point]

def euclidean_distance(p1: Point, p2: Point) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# Frechet Distance functions
def discrete_frechet_distance(P: Curve, Q: Curve) -> float:
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        raise ValueError("Curves must have at least one point each.")
    dp = np.zeros((n, m))
    dp[0][0] = euclidean_distance(P[0], Q[0])
    for i in range(1, n):
        dp[i][0] = max(dp[i-1][0], euclidean_distance(P[i], Q[0]))
    for j in range(1, m):
        dp[0][j] = max(dp[0][j-1], euclidean_distance(P[0], Q[j]))
    for i in range(1, n):
        for j in range(1, m):
            dp[i][j] = max(
                min(dp[i-1][j], dp[i-1][j-1], dp[i][j-1]),
                euclidean_distance(P[i], Q[j])
            )
    return dp[-1][-1]

def greedy_frechet_distance(curveA: Curve, curveB: Curve) -> Tuple[float, List[Tuple[int, int]]]:
    i, j = 0, 0
    path = [(i, j)]
    max_dist = euclidean_distance(curveA[0], curveB[0])
    while i < len(curveA) - 1 and j < len(curveB) - 1:
        da = euclidean_distance(curveA[i + 1], curveB[j])
        db = euclidean_distance(curveA[i], curveB[j + 1])
        if da < db:
            i += 1; current = da
        else:
            j += 1; current = db
        max_dist = max(max_dist, current)
        path.append((i, j))
    while i < len(curveA) - 1:
        i += 1; current = euclidean_distance(curveA[i], curveB[j]); max_dist = max(max_dist, current); path.append((i, j))
    while j < len(curveB) - 1:
        j += 1; current = euclidean_distance(curveA[i], curveB[j]); max_dist = max(max_dist, current); path.append((i, j))
    return max_dist, path

# Curve Analysis functions

def turning_angle_histogram(curve: Curve, num_bins: int = 32) -> np.ndarray:
    #    Compute histogram of heading changes (in radians) between successive segments.
    headings = []
    for i in range(1, len(curve)):
        dx = curve[i][0] - curve[i-1][0]
        dy = curve[i][1] - curve[i-1][1]
        headings.append(math.atan2(dy, dx))
    deltas = []
    for i in range(1, len(headings)):
        d = headings[i] - headings[i-1]
        d = (d + math.pi) % (2*math.pi) - math.pi
        deltas.append(d)
    hist, _ = np.histogram(deltas, bins=num_bins, range=(-math.pi, math.pi), density=True)
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist


def fourier_descriptor(curve: Curve, k: int = 20) -> np.ndarray:
    #    Compute first k Fourier coefficients (magnitudes) of the complex trajectory,
    #       padding or truncating so the output always has length k.
    z = np.array([x + 1j*y for x, y in curve])
    if len(z) < 2:
        mag = np.zeros(k)
    else:
        Z = fft(z)
        coeffs = Z[1:k+1] if len(Z) >= k+1 else Z[1:]
        mag = np.abs(coeffs)
        if len(mag) < k:
            mag = np.pad(mag, (0, k - len(mag)))
    norm = np.linalg.norm(mag)
    return mag / norm if norm > 0 else mag

def compute_distance_matrix(curveA: Curve, curveB: Curve) -> np.ndarray:
    """
    Construct a matrix of pairwise distances between points of two curves.
    """
    nA, nB = len(curveA), len(curveB)
    dist_matrix = np.zeros((nA, nB))
    for i in range(nA):
        for j in range(nB):
            dist_matrix[i, j] = euclidean_distance(curveA[i], curveB[j])
    return dist_matrix


def compute_jerk(curve: Curve, dt: float = 1.0) -> np.ndarray:
    curve = np.asarray(curve)
    v = np.diff(curve, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    j = np.diff(a, axis=0) / dt
    return j  #  (N-3, 2)
