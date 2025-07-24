import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple, Optional
from geopy.distance import geodesic
import matplotlib.dates as mdates

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from src.frechet.computations import Curve, compute_jerk, convert_curve_lonlat_to_xy, convert_curve_xy_to_lonlat


def plot_curves(
    curve_a: Curve,
    curve_b: Curve,
    label_a: str = "Curve A",
    label_b: str = "Curve B",
    title: str = "",
    path: Optional[List[Tuple[int, int]]] = None,
    d_exact: float = None,
    d_greedy: float = None,
    xlabel: str = "X",
    ylabel: str = "Y"
):
    """
    Plot two curves with optional matching path and distance info.
    """
    fig, ax = plt.subplots()
    ax.plot(*zip(*curve_a), '-o', color='blue', label=label_a)
    ax.plot(*zip(*curve_b), '-o', color='orange', label=label_b)

    if path is not None:
        for (i, j) in path:
            ax.plot([curve_a[i][0], curve_b[j][0]], [curve_a[i][1], curve_b[j][1]], 'k--', alpha=0.3)

    info = []
    if d_exact is not None:
        info.append(f"Exact = {d_exact:.4f}")
    if d_greedy is not None:
        info.append(f"Greedy = {d_greedy:.4f}")
    if info:
        full_title = title + " " + ", ".join(info) if title else ", ".join(info)
        ax.set_title(full_title)
    else:
        ax.set_title(title)

    ax.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def scatter_exact_vs_greedy(names, exact_dists, greedy_dists, ref_name):
    """
    Plot scatter of exact vs greedy Fréchet distances.
    """
    df_scatter = dict(Exact=exact_dists, Greedy=greedy_dists, Curve=names)
    fig = px.scatter(df_scatter, x='Exact', y='Greedy', text='Curve',
                     title=f'Exact vs Greedy Fréchet (ref: {ref_name})')
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis_title='Exact Fréchet', yaxis_title='Greedy Fréchet')
    fig.show()

def runtime_boxplot(exact_times, greedy_times):
    """
    Plot a boxplot of runtimes for each method.
    """
    df_time = {
        'Method': ['Exact'] * len(exact_times) + ['Greedy'] * len(greedy_times),
        'Runtime': exact_times + greedy_times
    }
    fig = px.box(df_time, x='Method', y='Runtime', log_y=True,
                 title='Runtime Distribution (log scale)')
    fig.show()

def plot_distance_matrix_with_path(dist_matrix: np.ndarray, path: List[Tuple[int, int]], ref_name: str, other_name: str):
    """
    Plot a heatmap of the pairwise distance matrix with path overlay.
    """
    fig = px.imshow(dist_matrix, origin='lower', color_continuous_scale='Viridis',
                    labels={'x': 'Curve B Index', 'y': 'Curve A Index', 'color': 'Distance'},
                    title=f'Distance Matrix + Path (ref: {ref_name} vs {other_name})')
    path_i, path_j = zip(*path)
    fig.add_scatter(x=path_j, y=path_i, mode='lines+markers',
                    line=dict(color='white', width=2),
                    marker=dict(size=4, color='white'),
                    showlegend=False)
    fig.show()

def plot_maply_curves(ref_curve: Curve, other_curve: Curve, ref_name: str, other_name: str):
    """
    Plot two curves on an interactive Plotly map.
    """
    lon_ref = [p[0] for p in ref_curve]
    lat_ref = [p[1] for p in ref_curve]
    lon_cur = [p[0] for p in other_curve]
    lat_cur = [p[1] for p in other_curve]

    fig_map = go.Figure()
    fig_map.add_trace(go.Scatter(
        x=lon_ref, y=lat_ref,
        mode='lines+markers',
        name='Reference',
        line=dict(color='blue'),
        marker=dict(size=6)
    ))
    fig_map.add_trace(go.Scatter(
        x=lon_cur, y=lat_cur,
        mode='lines+markers',
        name=other_name,
        line=dict(color='orange'),
        marker=dict(size=6)
    ))
    fig_map.update_layout(
        title=f'Curve Locations (ref: {ref_name} vs {other_name})',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        dragmode='zoom'
    )
    fig_map.show()

def plot_segmented_curves(segments: List[List[Tuple[float, float]]], title: str = "Segmented curves by jerk"):
    """
    Plot a list of sub-curves, each in a different color.

    Parameters:
        segments (List[List[Tuple[float, float]]]): List of sub-curves (each sub-curve is a list of points).
        title (str): Plot title.
    """
    colors = plt.cm.rainbow(np.linspace(0, 1, len(segments)))
    plt.figure(figsize=(8, 6))
    for seg, color in zip(segments, colors):
        arr = np.array(seg)
        plt.plot(arr[:, 0], arr[:, 1], '-o', color=color)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_time_and_distance_histograms(df, id_col='id', date_col='date', hour_col='hour', lat_col='lat', lon_col='lon'):

    df[hour_col] = df[hour_col].astype(str).str.zfill(6)
    df['datetime'] = pd.to_datetime(df[date_col].astype(str) + df[hour_col], format='%Y%m%d%H%M%S', errors='coerce')
    df = df.dropna(subset=['datetime', lat_col, lon_col])
    df = df.sort_values([id_col, 'datetime'])

    time_deltas = []
    distance_deltas = []

    for _, group in df.groupby(id_col):
        group = group.sort_values('datetime')
        times = group['datetime'].values
        coords = list(zip(group[lat_col], group[lon_col]))

        # Time differences in minutes
        td = np.diff(times).astype('timedelta64[s]').astype(float) / 60.0
        time_deltas.extend(td)

        # Distance differences in meters
        dd = [geodesic(coords[i], coords[i + 1]).meters for i in range(len(coords) - 1)]
        distance_deltas.extend(dd)

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Time intervals
    axs[0].hist(time_deltas, bins=50, color='red', density=True)
    axs[0].set_xlabel('minutes')
    axs[0].set_ylabel('proportion')
    axs[0].set_title('(a) time interval')

    # Distance intervals
    axs[1].hist(distance_deltas, bins=50, color='red', density=True)
    axs[1].set_xlabel('meters')
    axs[1].set_ylabel('proportion')
    axs[1].set_title('(b) distance interval')

    plt.tight_layout()
    plt.show()

def plot_curve_with_special_points(
    curve: Curve,
    special_points: list = None,
    special_labels: list = None,
    color: str = 'blue',
    title: str = '',
    xlabel: str = 'X',
    ylabel: str = 'Y',
    show: bool = True,
    save_path: str = None
):
    """
    Plot a curve and mark special points (e.g., start/end, outliers) with custom markers and labels.
    special_points: list of indices or (x, y) tuples to mark.
    special_labels: list of labels for each special point (optional).
    """
    arr = np.array(curve)
    fig, ax = plt.subplots()
    ax.plot(arr[:, 0], arr[:, 1], '-o', color=color, label='Curve')
    if special_points is not None:
        for i, pt in enumerate(special_points):
            if isinstance(pt, int):
                x, y = arr[pt, 0], arr[pt, 1]
            else:
                x, y = pt
            label = special_labels[i] if special_labels and i < len(special_labels) else None
            ax.scatter(x, y, s=100, marker='*', label=label, zorder=10)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.5)
    if special_points is not None and special_labels:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    else:
        ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)

def plot_processing_stages(original, filtered, augmented, simplified, curve_id=None, save_path=None):
    """
    Visualize trajectory processing stages side-by-side.
    - original, filtered, augmented, simplified: arrays/lists of (x, y) points
    - curve_id: optional string for title
    - save_path: if given, save the figure
    """

    stages = [
        ("Original", original, None, None),
        ("Filtered", filtered, 'red', 'removed'),
        ("Augmented", augmented, 'green', 'added'),
        ("Simplified", simplified, 'purple', 'removed')
    ]
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
    orig_arr = np.array(original)
    xlim = (orig_arr[:,0].min()-0.01, orig_arr[:,0].max()+0.01)
    ylim = (orig_arr[:,1].min()-0.01, orig_arr[:,1].max()+0.01)
    for i, (stage, curve, highlight_color, change_type) in enumerate(stages):
        ax = axs[i]
        arr = np.array(curve)
        # Plot original in muted gray for reference
        ax.plot(orig_arr[:,0], orig_arr[:,1], '-o', color='lightgray', label='Original', zorder=1)
        # Plot current stage
        ax.plot(arr[:,0], arr[:,1], '-o', label=stage, zorder=2)
        # Highlight added/removed points
        if change_type == 'removed':
            removed_pts = set(map(tuple, original)) - set(map(tuple, curve))
            if removed_pts:
                removed_pts = np.array(list(removed_pts))
                ax.scatter(removed_pts[:,0], removed_pts[:,1], color=highlight_color, s=60, marker='x', label='Removed', zorder=3)
        elif change_type == 'added':
            orig_set = set(map(tuple, original))
            added_pts = [pt for pt in curve if tuple(pt) not in orig_set]
            if added_pts:
                added_pts = np.array(added_pts)
                ax.scatter(added_pts[:,0], added_pts[:,1], color=highlight_color, s=60, marker='*', label='Added', zorder=3)
        # Annotate number of points
        ax.text(0.05, 0.95, f'N={len(curve)}', transform=ax.transAxes, fontsize=12, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        # Annotate total path length
        path_len = np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1))
        ax.text(0.05, 0.85, f'Len={path_len:.2f}', transform=ax.transAxes, fontsize=10, va='top', ha='left', color='gray')
        ax.set_title(stage)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
    if curve_id:
        plt.suptitle(f'Curve {curve_id}: Processing Stages', fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


def plot_alignment(ax, curve1_xy, curve2_xy, label1, label2, title):
    curve1_lonlat = convert_curve_xy_to_lonlat(curve1_xy)
    curve2_lonlat = convert_curve_xy_to_lonlat(curve2_xy)
    curve1 = np.array(curve1_lonlat)
    curve2 = np.array(curve2_lonlat)
    # Plot trajectories
    ax.plot(curve1[:,0], curve1[:,1], '-o', label=label1, color='navy', alpha=0.8, linewidth=2, markersize=3)
    ax.plot(curve2[:,0], curve2[:,1], '-o', label=label2, color='forestgreen', alpha=0.8, linewidth=2, markersize=3)
    # Start/End markers
    ax.scatter(curve1[0,0], curve1[0,1], color='navy', s=60, marker='o', edgecolor='black', zorder=5, label=f'{label1} start')
    ax.scatter(curve1[-1,0], curve1[-1,1], color='navy', s=60, marker='*', edgecolor='black', zorder=5, label=f'{label1} end')
    ax.scatter(curve2[0,0], curve2[0,1], color='forestgreen', s=60, marker='o', edgecolor='black', zorder=5, label=f'{label2} start')
    ax.scatter(curve2[-1,0], curve2[-1,1], color='forestgreen', s=60, marker='*', edgecolor='black', zorder=5, label=f'{label2} end')
    # Optional: draw arrows for direction
    ax.arrow(curve1[-2,0], curve1[-2,1],
             curve1[-1,0]-curve1[-2,0], curve1[-1,1]-curve1[-2,1],
             head_width=0.02, head_length=0.03, fc='navy', ec='navy', alpha=0.7)
    ax.arrow(curve2[-2,0], curve2[-2,1],
             curve2[-1,0]-curve2[-2,0], curve2[-1,1]-curve2[-2,1],
             head_width=0.02, head_length=0.03, fc='forestgreen', ec='forestgreen', alpha=0.7)
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_aspect('equal', 'box')
    # Clean x/y ticks
    ax.tick_params(axis='both', which='both', labelsize=9)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

# --- GPS POINT DENSITY AND INTERVAL HISTOGRAMS ---
def plot_gps_density_and_intervals(df):
    """
    Visualize the density distribution of GPS points, and plot histograms of time intervals and distances between consecutive points.
    Args:
        df (pd.DataFrame): DataFrame with columns ['id', 'datetime', 'lat', 'lon']
    """

    # Drop rows with missing values
    df = df.dropna(subset=['id', 'datetime', 'lat', 'lon'])
    print(f"Total GPS points: {len(df)}, first 5 rows:\n{df.head()}")
    # Sort by id and datetime
    df = df.sort_values(['id', 'datetime'])

    # Histograms of time interval and distance between consecutive points
    time_deltas = []
    distance_deltas = []
    for _, group in df.groupby('id'):
        group = group.sort_values('datetime')
        times = group['datetime'].values
        coords = list(zip(group['lat'], group['lon']))
        # Time differences in minutes
        td = np.diff(times).astype('timedelta64[s]').astype(float) / 60.0
        time_deltas.extend(td)
        # Distance differences in meters
        dd = [geodesic(coords[i], coords[i + 1]).meters for i in range(len(coords) - 1)]
        distance_deltas.extend(dd)
    # Convert to numpy arrays and filter
    time_deltas = np.array(time_deltas)
    #time_deltas = time_deltas[time_deltas <= 12] # Filter out intervals > 12 minutes
    distance_deltas = np.array(distance_deltas)
    #distance_deltas = distance_deltas[distance_deltas <= 8000] # Filter out distances > 8000 meters
    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].hist(time_deltas, bins=np.arange(0, 13, 1), color='royalblue', density=True)
    axs[0].set_xlabel('Time Interval (minutes)')
    axs[0].set_ylabel('Proportion')
    axs[0].set_title('Histogram of Time Intervals Between Points')
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[1].hist(distance_deltas, bins=np.arange(0, 8200, 400), color='darkorange', density=True)
    axs[1].set_xlabel('Distance (meters)')
    axs[1].set_ylabel('Proportion')
    axs[1].set_title('Histogram of Distances Between Points')
    axs[1].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('gps_time_distance_histograms.pdf')
    plt.close()




def plot_jerk_norms_with_outliers(
        jerk_norms, outlier_idx, id_val,
        SIGMA=3,
        filtered_jerk_norms=None,
        pdf_path=None,
        original_curve_lonlat=None
):
    """
    Plots jerk norms with outliers and threshold, and (optionally) filtered jerk norms.
    Also plots the original curve with outlier points highlighted if original_curve is provided.
    Saves all to a PDF. If filtered_curve jerk is provided, plots the filtered jerk norm as well.
    """
    threshold= np.nanmean(jerk_norms) + SIGMA * np.nanstd(jerk_norms)
    if pdf_path is None:
        pdf_path = f'jerk_trend_and_filtered_id_{id_val}.pdf'

    with PdfPages(pdf_path) as pdf:
        # --- Main jerk norm plot ---
        plt.figure(figsize=(10, 4))
        plt.plot(jerk_norms, label='Jerk Norm ')
        if len(outlier_idx) > 0:
            jerk_outlier_idx = np.asarray(outlier_idx, dtype=int)
            jerk_outlier_idx=jerk_outlier_idx-2
            plt.scatter(jerk_outlier_idx, np.array(jerk_norms)[jerk_outlier_idx], color='red', s=60, label='Outliers')

        plt.axhline(threshold, color='orange', linestyle='--', label=f'Outlier threshold ({SIGMA}\u03c3)')
        plt.annotate('Outliers detected as points above orange dashed line (3\u03c3)',
                     xy=(len(jerk_norms) * 0.7, threshold),
                     xytext=(len(jerk_norms) * 0.5, threshold + np.nanstd(jerk_norms)),
                     arrowprops=dict(facecolor='orange', shrink=0.05, width=1, headwidth=8),
                     fontsize=9, color='orange',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='orange', alpha=0.7))
        plt.title(f'Jerk Norm Trend (id={id_val})')
        plt.xlabel('Point Index')
        plt.ylabel('Jerk Norm (m/sec³)')
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # --- Optional: filtered jerk norm plot ---
        if  len(filtered_jerk_norms) > 0:
            plt.figure(figsize=(10, 4))
            plt.plot(filtered_jerk_norms, label='Filtered Jerk Norm', color='green')
            plt.title(f'Filtered Jerk Norm Trend (id={id_val})')
            plt.xlabel('Point Index (Filtered)')
            plt.ylabel('Jerk Norm (m/sec³)')
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # --- Optional: original curve with outlier points in red ---
        if original_curve_lonlat is not None and len(original_curve_lonlat) >= 4:
            arr = np.array(original_curve_lonlat)
            plt.figure(figsize=(8, 6))
            plt.plot(arr[:, 0], arr[:, 1], '-o', color='blue', label='Original Curve')
            if len(outlier_idx) > 0:
                plt.scatter(arr[outlier_idx, 0], arr[outlier_idx, 1], color='red', s=80, label='Outlier Points')
            plt.title(f"Original Curve with Outlier Points (id={id_val})")
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            pdf.savefig(); plt.close()

    print(f"Plots saved to {pdf_path}")
