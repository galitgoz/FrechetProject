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

def segment_curve_by_jerk(curve: Curve, jerk_threshold: float) -> List[Curve]:
    """
    Split the curve into sub-curves whenever jerk norm exceeds the given threshold.
    Returns a list of sub-curves.
    """
    if len(curve) < 4:
        return [curve]

    jerks = compute_jerk(curve)
    jerk_norms = np.linalg.norm(jerks, axis=1)

    # Pad with zeros at start so that indices match original curve
    split_indices = np.where(jerk_norms > jerk_threshold)[0] + 2  # +2 for alignment with curve indices

    # Always start from zero
    prev = 0
    segments = []
    for idx in split_indices:
        # Avoid degenerate segments
        if idx - prev > 2:
            segments.append(curve[prev:idx+1])
            prev = idx
    # Last segment
    if prev < len(curve) - 1:
        segments.append(curve[prev:])

    return segments

def rdp_simplify(curve: Curve, epsilon: float) -> Curve:
    """
    Ramer–Douglas–Peucker simplification for 2D curves.
    Args:
        curve: List of (x, y) points.
        epsilon: Tolerance (max distance from original curve).
    Returns:
        Simplified curve as a list of points.
    """
    if len(curve) < 3:
        return curve
    def point_line_distance(point, start, end):
        if start == end:
            return euclidean_distance(point, start)
        x0, y0 = point
        x1, y1 = start
        x2, y2 = end
        num = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        den = math.hypot(y2-y1, x2-x1)
        return num / den
    dmax = 0.0
    index = 0
    for i in range(1, len(curve)-1):
        d = point_line_distance(curve[i], curve[0], curve[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax > epsilon:
        rec1 = rdp_simplify(curve[:index+1], epsilon)
        rec2 = rdp_simplify(curve[index:], epsilon)
        return rec1[:-1] + rec2
    else:
        return [curve[0], curve[-1]]

