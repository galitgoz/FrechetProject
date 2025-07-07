import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from src.frechet.computations import compute_jerk, filter_outlier_points, rdp_simplify, discrete_frechet_distance, align_curves, velocity_grid_simplify
from src.frechet.visualization import plot_curves, plot_curve_with_special_points
from tqdm import tqdm
from simplification.cutil import simplify_coords_vw, simplify_coords
import io
from PIL import Image

# --- CONFIG ---
CSV_PATH = '../data/taxi.csv'  #  dataset
SIGMA = 3
RDP_EPSILON = 0.0005  # Adjust as needed
ALIGN_PAIR_START = True  # align start for each pair
ALIGN_PAIR_END = True   # align end for each pair

# --- LOAD DATA ---
MAX_TAXIS = 5  # Set to an integer to limit the number of taxis loaded, or None for all
df = pd.read_csv(CSV_PATH)
unique_ids = df['id'].unique()
if MAX_TAXIS is not None:
    unique_ids = unique_ids[:MAX_TAXIS]
df = df[df['id'].isin(unique_ids)]
if 'date' in df.columns and 'hour' in df.columns:
    df = df.sort_values(['id', 'date', 'hour'])

curves = {}
all_jerks = []
for id_val in tqdm(unique_ids, desc='Jerk computation and simplification'):
    subdf = df[df['id'] == id_val].copy()
    curve = list(zip(subdf['lon'], subdf['lat']))
    if len(curve) < 4:
        continue
    # Compute jerk
    jerks = compute_jerk(curve)
    jerk_norms = np.linalg.norm(jerks, axis=1)
    all_jerks.extend(jerk_norms)
    # Filter outliers
    filtered_curve, keep_idx = filter_outlier_points(curve, jerks, sigma=SIGMA)
    # Simplify using velocity_grid_simplify
    simplified_curve = velocity_grid_simplify(filtered_curve)
    curves[id_val] = {
        'original': curve,
        'filtered': filtered_curve,
        'simplified': simplified_curve,
        'jerks': jerks,
        'jerk_norms': jerk_norms
    }

# --- JERK HISTOGRAM FOR FULL DATA ---
if all_jerks:
    plt.figure(figsize=(8, 4))
    plt.hist(all_jerks, bins=50, color='teal', alpha=0.7)
    plt.xlabel('Jerk Norm (degrees/hour³)')
    plt.ylabel('Count')
    plt.title('Histogram of Jerk Norms (All Curves)')
    plt.tight_layout()
    plt.savefig('jerk_histogram_all_curves.pdf')
    plt.show()
    print('Jerk histogram saved to jerk_histogram_all_curves.pdf')

# --- SIMPLIFIED VS FILTERED POINT COUNT BOXPLOTS ---
curve_ids = list(curves.keys())
filtered_lens = [len(curves[id_val]['filtered']) for id_val in curve_ids]
simplified_lens = [len(curves[id_val]['simplified']) for id_val in curve_ids]

if filtered_lens and simplified_lens:
    data = [filtered_lens, simplified_lens]
    labels = ['Filtered', 'Simplified']
    fig, ax = plt.subplots(figsize=(max(8, len(curve_ids)), 6))
    # For each curve, plot two boxplots side by side
    positions = []
    box_data = []
    xtick_labels = []
    for i, id_val in enumerate(curve_ids):
        positions.extend([2*i+1, 2*i+2])
        box_data.append([len(curves[id_val]['filtered'])])
        box_data.append([len(curves[id_val]['simplified'])])
        xtick_labels.extend([f'{id_val}\nFiltered', f'{id_val}\nSimplified'])
    bplot = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True, showmeans=True)
    colors = ['skyblue', 'salmon'] * len(curve_ids)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xticks(positions)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
    ax.set_ylabel('Number of Points')
    ax.set_title('Filtered vs Simplified Point Counts per Curve (Boxplots)')
    plt.tight_layout()
    plt.savefig('filtered_vs_simplified_boxplots.pdf')
    plt.show()

# --- PAIRWISE FRECHET ANALYSIS ---
ids = list(curves.keys())
results = []
t0_all = time.time()
total_pairs = len(ids) * (len(ids) - 1) // 2
pair_count = 0

# Optional: limit the number of pairs processed
MAX_PAIRS = None  # Set to an integer to limit, or None for all pairs

for i in range(len(ids)):
    for j in range(i+1, len(ids)):
        if MAX_PAIRS is not None and pair_count >= MAX_PAIRS:
            break
        pair_count += 1
        print(f"[Pair {pair_count}/{total_pairs}] Comparing id {ids[i]} vs {ids[j]}")
        t_pair = time.time()
        id_a, id_b = ids[i], ids[j]
        ca, cb = curves[id_a], curves[id_b]
        # Align filtered curves before Fréchet calculation
        a_pre = ca['filtered']
        b_pre = cb['filtered']
        a_aligned, b_aligned = align_curves(a_pre, b_pre, ALIGN_PAIR_START, ALIGN_PAIR_END)
        # Visualization: pre-aligned vs aligned using plot_curve_with_special_points
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        # Pre-aligned
        a_pre_arr = np.array(a_pre)
        b_pre_arr = np.array(b_pre)
        plot_curve_with_special_points(
            a_pre,
            special_points=[0, len(a_pre)-1],
            special_labels=[f'{id_a} start', f'{id_a} end'],
            color='blue',
            title=f'{id_a} (pre)',
            show=False
        )
        axs[0].imshow(plt.gcf().canvas.buffer_rgba())
        plt.clf()
        plot_curve_with_special_points(
            b_pre,
            special_points=[0, len(b_pre)-1],
            special_labels=[f'{id_b} start', f'{id_b} end'],
            color='orange',
            title=f'{id_b} (pre)',
            show=False
        )
        axs[0].imshow(plt.gcf().canvas.buffer_rgba())
        axs[0].set_title('Pre-aligned curves')
        axs[0].axis('off')
        plt.clf()
        # Aligned
        a_aligned_arr = np.array(a_aligned)
        b_aligned_arr = np.array(b_aligned)
        plot_curve_with_special_points(
            a_aligned,
            special_points=[0, len(a_aligned)-1],
            special_labels=[f'{id_a} start', f'{id_a} end'],
            color='blue',
            title=f'{id_a} (aligned)',
            show=False
        )
        axs[1].imshow(plt.gcf().canvas.buffer_rgba())
        plt.clf()
        plot_curve_with_special_points(
            b_aligned,
            special_points=[0, len(b_aligned)-1],
            special_labels=[f'{id_b} start', f'{id_b} end'],
            color='orange',
            title=f'{id_b} (aligned)',
            show=False
        )
        axs[1].imshow(plt.gcf().canvas.buffer_rgba())
        axs[1].set_title('Aligned curves')
        axs[1].axis('off')
        fig.suptitle(f'Alignment visualization: {id_a} vs {id_b}')
        plt.tight_layout()
        plt.savefig(f'alignment_vis_{id_a}_vs_{id_b}.pdf')
        plt.close(fig)
        # Filtered (aligned)
        t0 = time.time()
        d_filt = discrete_frechet_distance(a_aligned, b_aligned)
        t1 = time.time()
        # Simplified (align simplified curves as well)
        a_simp_aligned, b_simp_aligned = align_curves(ca['simplified'], cb['simplified'], ALIGN_PAIR_START, ALIGN_PAIR_END)
        d_simp = discrete_frechet_distance(a_simp_aligned, b_simp_aligned)
        t2 = time.time()
        # Aligned (filtered, for compatibility)
        d_aligned = d_filt
        results.append({
            'id_a': id_a, 'id_b': id_b,
            'd_filt': d_filt,
            'd_simp': d_simp,
            'd_aligned': d_aligned,
            't_filt': t1-t0,
            't_simp': t2-t1
        })
        print(f"    Done in {time.time() - t_pair:.2f} seconds.")
    if MAX_PAIRS is not None and pair_count >= MAX_PAIRS:
        break
print(f"All pairs processed in {time.time() - t0_all:.2f} seconds.")

# --- DEVIATION ANALYSIS ---
results_df = pd.DataFrame(results)
results_df['diff_simp_vs_filt'] = results_df['d_simp'] - results_df['d_filt']
results_df['speedup_simp_vs_filt'] = results_df['t_filt'] / results_df['t_simp']

# --- HISTOGRAMS ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(results_df['diff_simp_vs_filt'], bins=30, color='purple', alpha=0.7, label='Simplified - Filtered')
plt.xlabel('Distance Difference (degrees)')
plt.ylabel('Count (# of pairs)')
plt.title('Simplified vs Filtered')
plt.legend()
plt.subplot(1,2,2)
# Filter out non-finite values before plotting
speedup_vals = results_df['speedup_simp_vs_filt']
speedup_vals_finite = speedup_vals[np.isfinite(speedup_vals)]
if len(speedup_vals_finite) < len(speedup_vals):
    print(f"Warning: {len(speedup_vals) - len(speedup_vals_finite)} non-finite values removed from speedup_simp_vs_filt before plotting.")
plt.hist(speedup_vals_finite, bins=30, color='purple', alpha=0.7, label='Simplified/Filtered')
plt.xlabel('Speedup (unitless)')
plt.ylabel('Count')
plt.title('Computation Speedup')
plt.legend()
plt.tight_layout()
# Add metadata to the PDF report
metadata_text = f"Data file: {os.path.basename(CSV_PATH)}\n" \
                f"Number of pairs: {pair_count}\n" \
                f"Simplification: RDP (epsilon={RDP_EPSILON})\n" \
                f"Aligned at start: {ALIGN_PAIR_START}, end: {ALIGN_PAIR_END}"
plt.gcf().text(0.01, 0.98, metadata_text, fontsize=8, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
plt.savefig('report_frechet.pdf')

# --- OPTIONAL: PLOT ORIGINAL CURVES WITH OUTLIER POINTS IN RED ---
PLOT_OUTLIERS = True  # Set to True to include these plots in the report
if PLOT_OUTLIERS:
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages('original_curves_with_outliers.pdf') as pdf:
        for id_val in curves:
            curve = curves[id_val]['original']
            # Use already calculated jerk_norms from curves dict
            jerk_norms = curves[id_val]['jerk_norms']
            threshold = jerk_norms.mean() + SIGMA * jerk_norms.std()
            outlier_idx = np.where(jerk_norms > threshold)[0]
            arr = np.array(curve)
            plt.figure(figsize=(8, 6))
            plt.plot(arr[:, 0], arr[:, 1], '-o', color='blue', label='Original Curve')
            # Mark outlier points in red on the curve
            if len(outlier_idx) > 0:
                outlier_curve_idx = [i+2 for i in outlier_idx if i+2 < len(arr)]
                plt.scatter(arr[outlier_curve_idx, 0], arr[outlier_curve_idx, 1], color='red', s=80, label='Outlier Points')
            plt.title(f"Original Curve with Outlier Points (id={id_val})")
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            pdf.savefig(); plt.close('all')
    print('Original curves with outlier points saved to original_curves_with_outliers.pdf')

# --- RANDOM PAIR CURVE PLOTS ---
if len(results) > 0:
    # Pick a random pair from the results (within MAX_PAIRS)
    random_result = random.choice(results)
    id_a = random_result['id_a']
    id_b = random_result['id_b']
    ca, cb = curves[id_a], curves[id_b]
    # Plot original curves
    plot_curves(ca['original'], cb['original'], label_a=f'Original {id_a}', label_b=f'Original {id_b}', title='Original Curves')
    # Plot filtered curves
    plot_curves(ca['filtered'], cb['filtered'], label_a=f'Filtered {id_a}', label_b=f'Filtered {id_b}', title='Filtered Curves')
    # Plot simplified curves
    plot_curves(ca['simplified'], cb['simplified'], label_a=f'Simplified {id_a}', label_b=f'Simplified {id_b}', title='Simplified Curves')

# --- SUMMARY ---
finite_diff = results_df['diff_simp_vs_filt'][np.isfinite(results_df['diff_simp_vs_filt'])]
finite_speedup = results_df['speedup_simp_vs_filt'][np.isfinite(results_df['speedup_simp_vs_filt'])]
print('Filtered vs Simplified deviation summary:')
print(finite_diff.describe())
print('Speedup summary:')
print(finite_speedup.describe())
print('Analysis complete. Histograms saved to frechet_deviation_histograms.pdf')

def velocity_diff_points(curve, times, threshold_kmh):
    """
    Given a curve (list of (lon, lat)) and times (in hours),
    return indices where the velocity changes by at least threshold_kmh.
    Always includes the first and last point.
    """
    from src.frechet.computations import haversine
    lats = np.array([p[1] for p in curve])
    lons = np.array([p[0] for p in curve])
    speeds = []
    for i in range(1, len(curve)):
        dist = haversine(lats[i-1], lons[i-1], lats[i], lons[i])
        dt = times[i] - times[i-1]
        speed = dist / dt if dt > 0 else 0.0
        speeds.append(speed)
    speeds = np.array(speeds)
    keep = set([0])
    for i in range(1, len(speeds)):
        if abs(speeds[i] - speeds[i-1]) >= threshold_kmh:
            keep.add(i)
    keep.add(len(curve)-1)
    return sorted(keep)

import math

for id_val in tqdm(unique_ids, desc='Jerk+Velocity+Grid simplification'):
    subdf = df[df['id'] == id_val].copy()
    curve = list(zip(subdf['lon'], subdf['lat']))
    n = len(curve)
    if n < 4:
        continue
    # Filter outliers (already done above, but repeat for clarity)
    jerks = compute_jerk(curve)
    filtered_curve, keep_idx = filter_outlier_points(curve, jerks, sigma=SIGMA)
    filtered_arr = np.array(filtered_curve)
    # Compute times for filtered curve
    if 'date' in subdf.columns and 'hour' in subdf.columns:
        subdf_filt = subdf.iloc[keep_idx].reset_index(drop=True)
        dt_str = subdf_filt['date'].astype(str) + subdf_filt['hour'].astype(str).str.zfill(6)
        times = pd.to_datetime(dt_str, format='%Y%m%d%H%M%S', errors='coerce').astype('int64') / 3.6e12  # hours
    else:
        times = np.arange(len(filtered_curve))
    # Add points by velocity diff
    vel_indices = velocity_diff_points(filtered_curve, times, threshold_kmh=10)
    curve_with_vel = [filtered_curve[i] for i in vel_indices]
    arr_vel = np.array(curve_with_vel)
    # Now sample sqrt(n)/log(n) points from this curve
    n0 = len(filtered_curve)
    c = math.log(n0) if n0 > 1 else 1
    num_points = max(2, int(np.sqrt(n0) / c))
    if num_points >= len(arr_vel):
        grid_simplified = arr_vel
    else:
        # Use grid sampling from computations
        from src.frechet.computations import simplify_curve_grid_sampling
        grid_simplified = simplify_curve_grid_sampling(arr_vel, num_points=num_points)
    # Store for later analysis
    curves[id_val]['vel_grid'] = grid_simplified
    # Visualization: plot each step
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].plot(filtered_arr[:,0], filtered_arr[:,1], '-o', label='Filtered (jerk)', color='blue')
    axs[0].set_title('Filtered (jerk)')
    axs[1].plot(filtered_arr[:,0], filtered_arr[:,1], '-o', color='blue', alpha=0.3)
    axs[1].plot(arr_vel[:,0], arr_vel[:,1], '-o', color='orange', label='+vel diff')
    axs[1].set_title('After velocity diff')
    axs[2].plot(arr_vel[:,0], arr_vel[:,1], '-o', color='orange', alpha=0.3)
    axs[2].plot(grid_simplified[:,0], grid_simplified[:,1], '-o', color='green', label='Grid sample')
    axs[2].set_title('Grid sample')
    axs[3].plot(filtered_arr[:,0], filtered_arr[:,1], '-o', color='blue', alpha=0.2, label='Filtered')
    axs[3].plot(grid_simplified[:,0], grid_simplified[:,1], '-o', color='green', label='Final')
    axs[3].set_title('Filtered vs Final')
    for ax in axs:
        ax.legend(); ax.grid(True, linestyle='--', alpha=0.5)
    fig.suptitle(f'Curve simplification steps (id={id_val})')
    plt.tight_layout()
    plt.savefig(f'curve_simplification_steps_id_{id_val}.pdf')
    plt.close(fig)

# --- FRECHET DISTANCE COMPARISON ---
from src.frechet.computations import discrete_frechet_distance, align_curves
frechet_results = []
for id_val in curves:
    if 'vel_grid' not in curves[id_val]:
        continue
    filtered = curves[id_val]['filtered']
    vel_grid = curves[id_val]['vel_grid']
    # Align for fair comparison
    filtered_aligned, vel_grid_aligned = align_curves(filtered, vel_grid, True, True)
    d_filt_vs_grid = discrete_frechet_distance(filtered_aligned, vel_grid_aligned)
    # For reference, also compare filtered vs simplified (RDP)
    simplified = curves[id_val]['simplified']
    filtered_aligned2, simplified_aligned = align_curves(filtered, simplified, True, True)
    d_filt_vs_simp = discrete_frechet_distance(filtered_aligned2, simplified_aligned)
    frechet_results.append({
        'id': id_val,
        'd_filt_vs_grid': d_filt_vs_grid,
        'd_filt_vs_simp': d_filt_vs_simp,
        'n_filtered': len(filtered),
        'n_vel_grid': len(vel_grid),
        'n_simplified': len(simplified)
    })

# --- VISUALIZE FRECHET DISTANCE COMPARISON ---
import pandas as pd
frechet_df = pd.DataFrame(frechet_results)
plt.figure(figsize=(8,6))
plt.scatter(frechet_df['n_filtered'], frechet_df['d_filt_vs_grid'], label='Filtered vs Vel+Grid', color='green')
plt.scatter(frechet_df['n_filtered'], frechet_df['d_filt_vs_simp'], label='Filtered vs RDP', color='purple', alpha=0.7)
plt.xlabel('Number of Points (Filtered)')
plt.ylabel('Fréchet Distance (degrees)')
plt.title('Fréchet Distance: Filtered vs Simplified/Vel+Grid')
plt.legend()
plt.tight_layout()
plt.savefig('frechet_distance_comparison.pdf')
plt.show()
print('Fréchet distance comparison saved to frechet_distance_comparison.pdf')
