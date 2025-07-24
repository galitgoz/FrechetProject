import math
import numpy as np
import pandas as pd
from scipy.fft import fft
from typing import List, Tuple
from rdp import rdp
from geopy.distance import geodesic
from pyproj import Transformer
from geopy.distance import geodesic

Point = Tuple[float, float]
Curve = List[Point]
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)


def euclidean_distance(p1: Point, p2: Point) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def convert_curve_lonlat_to_xy(curve_lonlat):
    """
    Input: List of (lon, lat) pairs in degrees
    Output: List of (x, y) pairs in meters (projected)
    """
    return [transformer.transform(lon, lat) for lon, lat in curve_lonlat]


def convert_curve_xy_to_lonlat(curve_xy):
    """
    Convert projected XY coordinates (meters) back to geographic coordinates (lon, lat in degrees).
    Input:
        curve_xy: List or array-like of (x, y) pairs in meters (projected)
    Output:
        List of (lon, lat) pairs in degrees
    """
    return [transformer.transform(x, y, direction='INVERSE') for x, y in curve_xy]


# Frechet Distance functions
def discrete_frechet_distance(P: Curve, Q: Curve) -> float:
    """
    Compute the discrete Fréchet distance between two curves in meters.
    Input points are (x, y) in projected coordinates (meters).
    """

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
            dist = euclidean_distance(P[i], Q[j])
            dp[i][j] = max(
                min(dp[i-1][j], dp[i-1][j-1], dp[i][j-1]),
                dist
            )
    return dp[-1][-1]

def continuous_frechet_distance(P: Curve, Q: Curve, precision: float = 1e-3) -> float:
    """
    Compute the continuous Fréchet distance between two curves using binary search.

    The continuous Fréchet distance allows for parameterizations along curve segments,
    providing a more flexible matching than the discrete version.

    Args:
        P, Q: Curves as lists of (x, y) points in meters
        precision: Convergence tolerance for binary search

    Returns:
        Continuous Fréchet distance in meters
    """
    if len(P) == 0 or len(Q) == 0:
        raise ValueError("Curves must have at least one point each.")

    # Binary search bounds
    min_dist = 0.0
    max_dist = max(euclidean_distance(p, q) for p in P for q in Q)

    # Binary search for minimum feasible distance
    while max_dist - min_dist > precision:
        mid_dist = (min_dist + max_dist) / 2.0
        if _is_continuous_frechet_feasible(P, Q, mid_dist):
            max_dist = mid_dist
        else:
            min_dist = mid_dist

    return max_dist

def _is_continuous_frechet_feasible(P: Curve, Q: Curve, epsilon: float) -> bool:
    """
    Check if continuous Fréchet distance <= epsilon using free space diagram.

    This implements the feasibility test for continuous Fréchet distance by
    constructing the free space diagram and checking for a monotone path.
    """
    n, m = len(P), len(Q)

    # Construct free space intervals for each cell
    free_space = {}

    # Process each cell (i,j) representing segments P[i]->P[i+1] and Q[j]->Q[j+1]
    for i in range(n - 1):
        for j in range(m - 1):
            # Get free space interval for this cell
            interval = _compute_free_space_interval(P[i], P[i+1], Q[j], Q[j+1], epsilon)
            if interval is not None:
                free_space[(i, j)] = interval

    # Check if there's a feasible path from (0,0) to (n-1,m-1)
    return _has_feasible_path(free_space, n-1, m-1, P, Q, epsilon)

def _compute_free_space_interval(p1: Point, p2: Point, q1: Point, q2: Point, epsilon: float) -> Tuple[float, float]:
    """
    Compute the free space interval for a cell defined by segments p1->p2 and q1->q2.

    Returns the parameter interval [t_min, t_max] where points on the segments
    are within distance epsilon, or None if no such interval exists.
    """
    # Parameterize segments: P(s) = p1 + s*(p2-p1), Q(t) = q1 + t*(q2-q1)
    # We need |P(s) - Q(t)| <= epsilon for some s,t in [0,1]

    dp = (p2[0] - p1[0], p2[1] - p1[1])  # Direction vector for P
    dq = (q2[0] - q1[0], q2[1] - q1[1])  # Direction vector for Q
    d0 = (p1[0] - q1[0], p1[1] - q1[1])  # Initial offset

    # We need to solve |d0 + s*dp - t*dq|² <= epsilon²
    # This is a quadratic in s and t

    # Coefficients for the quadratic equation
    a = dp[0]**2 + dp[1]**2
    b = dq[0]**2 + dq[1]**2
    c = -2 * (dp[0]*dq[0] + dp[1]*dq[1])
    d = 2 * (d0[0]*dp[0] + d0[1]*dp[1])
    e = -2 * (d0[0]*dq[0] + d0[1]*dq[1])
    f = d0[0]**2 + d0[1]**2 - epsilon**2

    # For fixed t, solve as*s² + (c*t + d)*s + (b*t² + e*t + f) <= 0
    valid_intervals = []

    # Sample t values and find valid s intervals
    num_samples = 100
    for i in range(num_samples + 1):
        t = i / num_samples

        # Quadratic in s: as² + (ct + d)s + (bt² + et + f) <= 0
        coeff_s2 = a
        coeff_s1 = c * t + d
        coeff_s0 = b * t**2 + e * t + f

        # Solve quadratic inequality
        if abs(coeff_s2) < 1e-12:  # Linear case
            if abs(coeff_s1) < 1e-12:
                if coeff_s0 <= 0:
                    s_min, s_max = 0.0, 1.0
                else:
                    continue
            else:
                s_root = -coeff_s0 / coeff_s1
                if coeff_s1 > 0:
                    s_min, s_max = max(0, s_root), 1.0
                else:
                    s_min, s_max = 0.0, min(1, s_root)
        else:  # Quadratic case
            discriminant = coeff_s1**2 - 4 * coeff_s2 * coeff_s0
            if discriminant < 0:
                if coeff_s2 < 0:  # Always negative quadratic
                    s_min, s_max = 0.0, 1.0
                else:
                    continue
            else:
                sqrt_disc = math.sqrt(discriminant)
                s1 = (-coeff_s1 - sqrt_disc) / (2 * coeff_s2)
                s2 = (-coeff_s1 + sqrt_disc) / (2 * coeff_s2)

                if coeff_s2 > 0:  # Upward parabola
                    s_min, s_max = max(0, s1), min(1, s2)
                else:  # Downward parabola
                    s_min = 0.0
                    s_max = 1.0
                    if s1 > 1 or s2 < 0:
                        continue
                    if 0 <= s1 <= 1:
                        s_max = min(s_max, s1)
                    if 0 <= s2 <= 1:
                        s_min = max(s_min, s2)

        if s_min <= s_max and s_min <= 1.0 and s_max >= 0.0:
            valid_intervals.append((max(0, s_min), min(1, s_max)))

    if not valid_intervals:
        return None

    # Merge overlapping intervals and find the overall range
    valid_intervals.sort()
    merged = [valid_intervals[0]]
    for start, end in valid_intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    if merged:
        return (merged[0][0], merged[-1][1])
    return None

def _has_feasible_path(free_space: dict, n: int, m: int, P: Curve, Q: Curve, epsilon: float) -> bool:
    """
    Check if there's a monotone path through the free space from (0,0) to (n,m).

    Uses dynamic programming to check connectivity through free space intervals.
    """
    # Check if start and end points are within epsilon
    if euclidean_distance(P[0], Q[0]) > epsilon:
        return False
    if euclidean_distance(P[-1], Q[-1]) > epsilon:
        return False

    # Dynamic programming table: reachable[(i,j)] = True if cell (i,j) is reachable
    reachable = {}

    # Base case: (0,0) is reachable if the first segments have free space
    if (0, 0) in free_space:
        reachable[(0, 0)] = True

    # Fill the DP table
    for i in range(n):
        for j in range(m):
            if (i, j) not in free_space:
                continue

            # Check if this cell is reachable from previous cells
            cell_reachable = False

            # From left cell (i, j-1)
            if j > 0 and (i, j-1) in reachable and reachable[(i, j-1)]:
                cell_reachable = True

            # From bottom cell (i-1, j)
            if i > 0 and (i-1, j) in reachable and reachable[(i-1, j)]:
                cell_reachable = True

            # From diagonal cell (i-1, j-1)
            if i > 0 and j > 0 and (i-1, j-1) in reachable and reachable[(i-1, j-1)]:
                cell_reachable = True

            # Special case for starting cell
            if i == 0 and j == 0:
                cell_reachable = True

            if cell_reachable:
                reachable[(i, j)] = True

    # Check if the target cell (n-1, m-1) is reachable
    return (n-1, m-1) in reachable and reachable[(n-1, m-1)]


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

def compute_jerk(curve_xy, times):
    """
    Compute jerk (third derivative) for a curve.
    curve: list of (x, y) in meters .
    times: array-like of timestamps (same length as curve)
    Returns: jerk array of shape (N-3, 2) [m/s^3] (X and Y components)
    Skips intervals with zero or NaN time difference to avoid divide-by-zero.
    """
    times = pd.to_datetime(times)
    times_sec = (times - times[0]).total_seconds()
    if isinstance(times_sec, pd.Series):
        times_sec = times_sec.values
    dt = np.diff(times_sec)

    valid_indices = dt > 0
    if not np.all(valid_indices):
        print("Warning: found zero or negative time intervals, these will be ignored")
        curve_xy = np.array(curve_xy)[np.append(True, valid_indices)]
        dt = dt[valid_indices]

    v = np.diff(curve_xy, axis=0) / dt[:, None]
    dt_a = dt[1:]
    a = np.diff(v, axis=0) / dt_a[:, None]
    dt_j = dt_a[1:]
    j = np.diff(a, axis=0) / dt_j[:, None]

    return j

## curve manipulation functions ##

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

def filter_outlier_points(curve_xy, jerk_norms, sigma=3):
    """
    Remove points from the curve where jerk norm is above mean + sigma*std.
    Returns filtered curve and indices kept.
    """

    threshold = np.nanmean(jerk_norms) + sigma * np.nanstd(jerk_norms)
    keep_idx = np.where(jerk_norms <= threshold)[0] + 2  # +2 to align correctly
    keep_idx = np.concatenate(([0, 1], keep_idx))
    outlier_idx = np.where(jerk_norms > threshold)[0] + 2
    filtered_curve_xy = np.array(curve_xy)[keep_idx]
    return filtered_curve_xy, keep_idx, outlier_idx

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

def simplify_curve_grid_sampling(curve_xy, num_points=None):
    """
    Simplify curve by sampling sqrt(n) points from grid cells. - need to fix if used, sample is num_points^2
    """
    curve = np.array(curve_xy)
    n = len(curve)

    if num_points is None:
        num_points = int(np.sqrt(n))

    if num_points < 2:
        return curve.copy()  # Don't reduce to nothing!

    # Compute bounding box
    min_x, min_y = curve.min(axis=0)
    max_x, max_y = curve.max(axis=0)

    # Define grid
    x_bins = np.linspace(min_x, max_x, num_points+1)
    y_bins = np.linspace(min_y, max_y, num_points+1)

    # Map: (cell_x, cell_y) -> index of chosen representative
    cell_to_idx = dict()

    for idx, (x, y) in enumerate(curve):
        # Find which cell
        cell_x = np.searchsorted(x_bins, x, side='right') - 1
        cell_y = np.searchsorted(y_bins, y, side='right') - 1
        # Boundaries correction (if x==max_x or y==max_y)
        cell_x = min(cell_x, num_points-1)
        cell_y = min(cell_y, num_points-1)
        cell = (cell_x, cell_y)
        # If cell not yet filled, pick the first point in curve order
        if cell not in cell_to_idx:
            cell_to_idx[cell] = idx

    # Sort indices as they appeared in the original curve
    ordered_indices = sorted(cell_to_idx.values())
    simplified_curve = curve[ordered_indices]

    if ordered_indices[-1] != n - 1:
        ordered_indices.append(n - 1)
        simplified_curve = curve[ordered_indices]

    return simplified_curve

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

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    #print("Haversine distance:", R * c, "km", f"Geodesic (geopy) distance: {geodesic((lat1, lon1), (lat2, lon2)).km:.6f} km" )
    return R * c


def parse_datetime_column(df):
    """
    Adds a 'datetime' column to the DataFrame from 'date' and 'hour' columns.
    Drops rows with invalid datetimes and sorts by id and datetime.
    """
    df['datetime'] = df.apply(lambda row: pd.to_datetime(str(row['date']) + str(row['hour']).zfill(6), format='%Y%m%d%H%M%S', errors='coerce'), axis=1)
    df = df.dropna(subset=['datetime'])
    df = df.sort_values(['id', 'datetime']).reset_index(drop=True)
    return df

def velocity_grid_simplify(curve, times=None, velocity_threshold_kmh=10, c=None):
    """
    Simplify a curve by:
    1. Adding points so that the velocity between consecutive points corresponds to velocity_threshold_kmh (km/h), using interpolate_curve_by_velocity.
    2. Then, sample sqrt(n)/c points (where n is the original curve length, c=log(n) by default) using uniform sampling.

    Args:
        curve: List/array of (x, y) coordinates in meters
        times: Array-like of timestamps (optional)
        velocity_threshold_kmh: Maximum velocity threshold in km/h
        c: Simplification factor (default: log(n))

    Returns:
        tuple: (simplified_curve, augmented_curve) as numpy arrays

    Raises:
        ValueError: If curve has less than 2 points
    """
    if len(curve) < 2:
        raise ValueError("Curve must have at least 2 points for simplification")

    # Convert curve to DataFrame for interpolation
    df = pd.DataFrame(curve, columns=['x', 'y'])
    # If times are provided, add a datetime column
    if times is not None:
        #df['datetime'] = pd.to_datetime(times)
        df['datetime'] = times
    else:
        # Use a dummy datetime index if not provided
        df['datetime'] = pd.date_range('2000-01-01', periods=len(df), freq='h')
    df['id'] = 0  # dummy id for grouping
    # Use interpolate_curve_by_velocity to pad points at constant velocity
    interpolated_df = interpolate_curve_xy_by_velocity(df, velocity_kmh=velocity_threshold_kmh)
    arr_vel = interpolated_df[['x', 'y']].to_numpy()
    n0 = len(curve)
    print(f"Original curve length: {n0}, Interpolated curve length: {len(arr_vel)}")
    if c is None:
        c = math.log(n0) if n0 > 1 else 1
    num_points = max(2, int(np.sqrt(n0) / c))
    if num_points >= len(arr_vel):
        simplified = arr_vel
    else:
        simplified =uniform_sample_curve_points(arr_vel, num_points=num_points)
    print(f"Number of points after uniform sampling: {len(simplified)}")
    return simplified, arr_vel

def interpolate_curve_by_velocity(df, velocity_kmh=10):
    """
    Interpolates points along each trajectory so that the time between consecutive points corresponds to traveling at `velocity_kmh`.
    All original points are preserved, and new points are inserted at every interval corresponding to `velocity_kmh` along the segment.
    Args:
        df (DataFrame): DataFrame with columns ['id', 'datetime', 'lat', 'lon'].
        velocity_kmh (float): Velocity in kilometers per hour for interpolation.
    Returns:
        DataFrame: DataFrame with original and interpolated points, sorted by id and datetime.
    """


    interpolated_rows = []
    for id_val, group in df.groupby('id'):
        group = group.sort_values('datetime').reset_index(drop=True)
        for i in range(len(group) - 1):
            row_start = group.iloc[i]
            row_end = group.iloc[i + 1]
            start_coords = (row_start['lat'], row_start['lon'])
            end_coords = (row_end['lat'], row_end['lon'])
            total_dist = geodesic(start_coords, end_coords).km
            total_time = (row_end['datetime'] - row_start['datetime']).total_seconds() / 3600.0  # hours
            # Always include the starting point
            interpolated_rows.append(row_start)
            if total_dist == 0 or total_time == 0:
                continue
            # Time interval for 10 km at velocity_kmh
            interval_hours = 10.0 / velocity_kmh
            num_intervals = int(np.floor(total_time / interval_hours))
            for interp_idx in range(1, num_intervals + 1):
                fraction = (interp_idx * interval_hours) / total_time
                if fraction >= 1:
                    break
                interp_lat = row_start['lat'] + fraction * (row_end['lat'] - row_start['lat'])
                interp_lon = row_start['lon'] + fraction * (row_end['lon'] - row_start['lon'])
                interp_time = row_start['datetime'] + pd.to_timedelta(fraction * total_time, unit='h')
                interpolated_rows.append({
                    'id': id_val,
                    'datetime': interp_time,
                    'lat': interp_lat,
                    'lon': interp_lon,
                })
        # Always include the last point
        interpolated_rows.append(group.iloc[-1])
    result_df = pd.DataFrame(interpolated_rows)
    result_df = result_df.sort_values(['id', 'datetime']).reset_index(drop=True)
    return result_df


def interpolate_curve_xy_by_velocity(df, velocity_kmh=10):
        """
        Interpolates points along each trajectory so that the velocity between consecutive points does not exceed `velocity_kmh`.
        All original points are preserved, and new points are inserted whenever velocity exceeds the threshold.
        Args:
            df (DataFrame): DataFrame with columns ['id', 'datetime', 'x', 'y'] (meters and datetime).
            velocity_kmh (float): Velocity threshold in kilometers per hour.
        Returns:
            DataFrame: DataFrame with original and interpolated points, sorted by id and datetime.
        """

        interpolated_rows = []
        velocity_ms = velocity_kmh * 1000 / 3600  # convert velocity to m/s

        for id_val, group in df.groupby('id'):
            group = group.sort_values('datetime').reset_index(drop=True)
            for i in range(len(group) - 1):
                row_start = group.iloc[i]
                row_end = group.iloc[i + 1]

                start_coords = np.array([row_start['x'], row_start['y']])
                end_coords = np.array([row_end['x'], row_end['y']])

                total_dist = np.linalg.norm(end_coords - start_coords)
                total_time_sec = (row_end['datetime'] - row_start['datetime']).total_seconds()

                # Always include the starting point as dictionary
                interpolated_rows.append(row_start.to_dict())

                if total_time_sec <= 0 or total_dist == 0:
                    continue

                current_velocity = total_dist / total_time_sec

                # Check if velocity exceeds threshold and interpolate accordingly
                if current_velocity > velocity_ms:
                    num_intervals = int(np.ceil(current_velocity / velocity_ms))
                    for interp_idx in range(1, num_intervals):
                        fraction = interp_idx / num_intervals
                        interp_x = start_coords[0] + fraction * (end_coords[0] - start_coords[0])
                        interp_y = start_coords[1] + fraction * (end_coords[1] - start_coords[1])
                        interp_time = row_start['datetime'] + pd.to_timedelta(fraction * total_time_sec, unit='s')

                        interpolated_rows.append({
                            'id': id_val,
                            'datetime': interp_time,
                            'x': interp_x,
                            'y': interp_y,
                            'interpolated': True
                        })

            # Always include the last point as dictionary
            if len(group) > 0:
                interpolated_rows.append(group.iloc[-1].to_dict())

        result_df = pd.DataFrame(interpolated_rows)
        result_df.sort_values(['id', 'datetime'], inplace=True)
        result_df.reset_index(drop=True, inplace=True)

        return result_df

def uniform_sample_curve_points(curve_xy, num_points):
    """
    Uniformly samples a subset of points from the given curve,
    using only the original points (no interpolation),
    so that the sampled points are as evenly spaced as possible along the curve length.

    Parameters:
        curve_xy (array-like): Nx2 array of (x, y) coordinates in meters.
        num_points (int): The number of points to sample.

    Returns:
        np.ndarray: num_points x 2 array of sampled points from curve_xy.
    """
    curve_xy = np.asarray(curve_xy)
    n = len(curve_xy)
    if num_points >= n:
        # If requested more points than available, return all points
        return curve_xy.copy()

    # Compute cumulative arc-length (distance from the start for each point)
    segment_lengths = np.linalg.norm(np.diff(curve_xy, axis=0), axis=1)
    cumlen = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumlen[-1]

    # Determine desired distances along the curve for sampling
    target_distances = np.linspace(0, total_length, num_points)

    # For each desired distance, find the index of the nearest original point
    chosen_idx = np.searchsorted(cumlen, target_distances)

    # Remove duplicates if any (possible if many points are close together)
    unique_idx = np.unique(chosen_idx)

    # If we have fewer than num_points after deduplication, pad with extra indices from remaining points
    if len(unique_idx) < num_points:
        pad = num_points - len(unique_idx)
        extra_idx = np.setdiff1d(np.arange(n), unique_idx)
        unique_idx = np.concatenate([unique_idx, extra_idx[:pad]])
        unique_idx = np.sort(unique_idx)

    # Return the selected points
    return curve_xy[unique_idx]
