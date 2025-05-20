import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple, Optional

from .computations import Curve

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
    fig, ax = plt.subplots()

    # first curve
    ax.plot(*zip(*curve_a), '-o', color='blue', label=label_a)
    # second curve
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
    df_scatter = dict(Exact=exact_dists, Greedy=greedy_dists, Curve=names)
    fig = px.scatter(df_scatter, x='Exact', y='Greedy', text='Curve',
                     title=f'Exact vs Greedy Fréchet (ref: {ref_name})')
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis_title='Exact Fréchet', yaxis_title='Greedy Fréchet')
    fig.show()

def runtime_boxplot(exact_times, greedy_times):
    df_time = {
        'Method': ['Exact'] * len(exact_times) + ['Greedy'] * len(greedy_times),
        'Runtime': exact_times + greedy_times
    }
    fig = px.box(df_time, x='Method', y='Runtime', log_y=True,
                 title='Runtime Distribution (log scale)')
    fig.show()

def plot_distance_matrix_with_path(dist_matrix: np.ndarray, path: List[Tuple[int, int]], ref_name: str, other_name: str):
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

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

def plot_jerk(
    jerks: np.ndarray,
    title: str = "Jerk along the curve",
    show_norm: bool = True
):
    """
    Plot the jerk values along a curve.

    Parameters:
        jerks (np.ndarray): Array of shape (N-3, 2) where each row contains the jerk vector (dx, dy) at each point.
        title (str): Plot title.
        show_norm (bool): If True, also plot the jerk norm (magnitude).
    """
    n = jerks.shape[0]
    idx = np.arange(n)

    plt.figure(figsize=(10, 5))

    # Plot X and Y components of jerk
    plt.plot(idx, jerks[:, 0], label='Jerk X', color='blue')
    plt.plot(idx, jerks[:, 1], label='Jerk Y', color='orange')

    # Optionally plot the norm (magnitude) of the jerk vector
    if show_norm:
        jerk_norms = np.linalg.norm(jerks, axis=1)
        plt.plot(idx, jerk_norms, label='Jerk Norm', color='green', linestyle='--', linewidth=2)

    plt.xlabel("Point Index")
    plt.ylabel("Jerk Value")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_curve_with_jerk_coloring(
    curve: List[Tuple[float, float]],
    jerks: np.ndarray,
    title: str = "Curve colored by Jerk norm"
):
    """
    Plot the curve, coloring each segment by the jerk norm.

    Parameters:
        curve (List[Tuple[float, float]]): List of 2D points (x, y) representing the curve.
        jerks (np.ndarray): Array of shape (N-3, 2) with the jerk vectors at each point.
        title (str): Plot title.
    """
    import matplotlib.cm as cm

    # Compute jerk norm (magnitude) for coloring
    jerk_norms = np.linalg.norm(jerks, axis=1)
    # Normalize jerk values to [0, 1] for the colormap
    norm = (jerk_norms - jerk_norms.min()) / (jerk_norms.ptp() + 1e-9)
    cmap = cm.get_cmap('plasma')

    curve_arr = np.array(curve)
    # Draw each segment with color corresponding to jerk norm
    for i in range(2, len(curve)-1):
        plt.plot(
            curve_arr[i:i+2, 0],
            curve_arr[i:i+2, 1],
            color=cmap(norm[i-2]),
            linewidth=3
        )
    # Overlay all points
    plt.scatter(curve_arr[:, 0], curve_arr[:, 1], c='black', s=20, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

