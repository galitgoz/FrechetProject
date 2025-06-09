import math
import numpy as np
import pandas as pd
from scipy.fft import fft
from typing import List, Tuple
from rdp import rdp

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

def align_curves(curveA: Curve, curveB: Curve, align_start=True, align_end=False) -> Tuple[Curve, Curve]:
    """
    Align two curves by their start and/or end points.
    If align_start: translate both so their first points coincide.
    If align_end: after aligning starts, rotate/scale so ends match.
    Returns aligned copies of curveA, curveB.
    """
    a = np.array(curveA)
    b = np.array(curveB)
    if align_start:
        offset = a[0] - b[0]
        b = b + offset
    if align_end:
        vec_a = a[-1] - a[0]
        vec_b = b[-1] - b[0]
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a > 0 and norm_b > 0:
            scale = norm_a / norm_b
            b = (b - b[0]) * scale + b[0]
            angle_a = np.arctan2(vec_a[1], vec_a[0])
            angle_b = np.arctan2(vec_b[1], vec_b[0])
            angle = angle_a - angle_b
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            b = (b - b[0]) @ rot.T + b[0]
    return a.tolist(), b.tolist()

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


def compute_jerk(curve: Curve, dt: float = 1.0, times: np.ndarray = None) -> np.ndarray:
    """
    Compute jerk (third derivative) for a curve.
    If times is provided, use non-uniform time intervals.
    curve: list of (x, y)
    times: array-like of timestamps (same length as curve)
    Returns: jerk array of shape (N-3, 2)
    """
    curve = np.asarray(curve)
    if times is not None:
        times = np.asarray(times)
        # Compute velocities with variable dt
        v = np.diff(curve, axis=0) / np.diff(times)[:, None]
        a = np.diff(v, axis=0) / np.diff(times[1:])[:, None]
        j = np.diff(a, axis=0) / np.diff(times[2:])[:, None]
        return j
    else:
        v = np.diff(curve, axis=0) / dt
        a = np.diff(v, axis=0) / dt
        j = np.diff(a, axis=0) / dt
        return j

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

def filter_outlier_points(curve, jerks, sigma=3):
    """
    Remove points from the curve where jerk norm is above mean + sigma*std.
    Returns filtered curve and indices kept.
    """
    jerk_norms = np.linalg.norm(jerks, axis=1)
    threshold = jerk_norms.mean() + sigma * jerk_norms.std()
    # Indices of points to keep (pad to match curve length)
    keep_idx = [0, 1]  # always keep first two points
    keep_idx += [i+2 for i, jn in enumerate(jerk_norms) if jn <= threshold]
    filtered_curve = [curve[i] for i in keep_idx]
    return filtered_curve, keep_idx

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


def simplify_curve_grid_sampling(curve, num_points=None):
    """
    Simplify curve by sampling sqrt(n) points from grid cells.
    """
    curve = np.array(curve)
    n = len(curve)

    if num_points is None:
        num_points = int(np.sqrt(n))

    # Compute bounding box
    min_x, min_y = curve.min(axis=0)
    max_x, max_y = curve.max(axis=0)

    # Define grid
    x_bins = np.linspace(min_x, max_x, num_points+1)
    y_bins = np.linspace(min_y, max_y, num_points+1)

    # Store sampled points
    simplified_points = []

    # For each grid cell, pick a representative point
    for i in range(num_points):
        for j in range(num_points):
            # Points inside the current cell
            in_cell = curve[
                (curve[:,0] >= x_bins[i]) & (curve[:,0] < x_bins[i+1]) &
                (curve[:,1] >= y_bins[j]) & (curve[:,1] < y_bins[j+1])
            ]

            if len(in_cell) > 0:
                # Cell center
                cell_center = np.array([
                    (x_bins[i] + x_bins[i+1]) / 2,
                    (y_bins[j] + y_bins[j+1]) / 2
                ])

                # Select the point closest to cell center
                distances = np.linalg.norm(in_cell - cell_center, axis=1)
                representative_point = in_cell[np.argmin(distances)]

                simplified_points.append(representative_point)

    return np.array(simplified_points)

def combined_simplification(curve, jerk_sigma=3, angle_threshold=np.pi/8):
    """
    Combines jerk-based and angular-based simplification.

    Params:
    - curve: original curve as array (n,2).
    - jerk_sigma: threshold for jerk filtering (higher means fewer points).
    - angle_threshold: threshold (radians) for angular simplification.

    Returns:
    - simplified_curve: simplified points preserving jerk and angular features.
    """
    curve = np.array(curve)

    # Step 1: Jerk-based simplification
    jerks = compute_jerk(curve)
    jerk_norms = np.linalg.norm(jerks, axis=1)
    jerk_threshold = jerk_norms.mean() + jerk_sigma * jerk_norms.std()

    # Keep indices with high jerk
    jerk_high_idx = set(np.where(jerk_norms > jerk_threshold)[0] + 2)  # offset by 2 due to jerk computation

    # Step 2: Angular-based simplification
    def compute_angles(curve):
        v1 = curve[1:-1] - curve[:-2]
        v2 = curve[2:] - curve[1:-1]
        norm1 = np.linalg.norm(v1, axis=1)
        norm2 = np.linalg.norm(v2, axis=1)
        cos_angle = np.einsum('ij,ij->i', v1, v2) / (norm1 * norm2 + 1e-9)
        angles = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return angles

    angles = compute_angles(curve)
    angle_high_idx = set(np.where(angles > angle_threshold)[0] + 1)  # offset by 1 due to angle computation

    # Combine indices from both criteria
    key_points_idx = jerk_high_idx.union(angle_high_idx)
    key_points_idx.update({0, len(curve)-1})  # Always keep first and last points

    # Sort indices and generate simplified curve
    simplified_idx = sorted(key_points_idx)
    simplified_curve = curve[simplified_idx]

    return simplified_curve
