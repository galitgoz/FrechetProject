import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from src.frechet.computations import compute_jerk, filter_outlier_points, rdp_simplify, discrete_frechet_distance, align_curves
from src.frechet.visualization import plot_curves
from tqdm import tqdm
from simplification.cutil import simplify_coords_vw, simplify_coords

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
for id_val in tqdm(unique_ids, desc='Jerk computation and simplification'):
    subdf = df[df['id'] == id_val].copy()
    curve = list(zip(subdf['lon'], subdf['lat']))
    if len(curve) < 4:
        continue
    # Filter outliers
    jerks = compute_jerk(curve)
    filtered_curve, keep_idx = filter_outlier_points(curve, jerks, sigma=SIGMA)
    # Simplify
    simplified_curve = rdp_simplify(filtered_curve, epsilon=RDP_EPSILON)
    curves[id_val] = {
        'original': curve,
        'filtered': filtered_curve,
        'simplified': simplified_curve
    }

# --- JERK HISTOGRAM FOR FULL DATA ---
all_jerks = []
for id_val in curves:
    curve = curves[id_val]['original']
    if len(curve) < 4:
        continue
    jerks = compute_jerk(curve)
    jerk_norms = np.linalg.norm(jerks, axis=1)
    all_jerks.extend(jerk_norms)
if all_jerks:
    plt.figure(figsize=(8, 4))
    plt.hist(all_jerks, bins=50, color='teal', alpha=0.7)
    plt.xlabel('Jerk Norm')
    plt.ylabel('Curve Count')
    plt.title('Histogram of Jerk Norms (All Curves)')
    plt.tight_layout()
    plt.savefig('jerk_histogram_all_curves.pdf')
    plt.show()
    print('Jerk histogram saved to jerk_histogram_all_curves.pdf')

# --- SIMPLIFIED VS FILTERED POINT COUNT SCATTERPLOT ---
filtered_lens = []
simplified_lens = []
for id_val in curves:
    filtered_lens.append(len(curves[id_val]['filtered']))
    simplified_lens.append(len(curves[id_val]['simplified']))
if filtered_lens and simplified_lens:
    plt.figure(figsize=(7, 7))
    plt.scatter(filtered_lens, simplified_lens, alpha=0.6, color='darkred')
    plt.plot([min(filtered_lens), max(filtered_lens)], [min(filtered_lens), max(filtered_lens)], 'k--', label='y=x')
    # Add y=sqrt(x) reference line
    x_vals = np.linspace(min(filtered_lens), max(filtered_lens), 200)
    plt.plot(x_vals, np.sqrt(x_vals), 'g-.', label='y=sqrt(x)')
    plt.xlabel('Number of Points (Filtered)')
    plt.ylabel('Number of Points (Simplified)')
    plt.title('Simplified vs Filtered Curve Point Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('simplified_vs_filtered_point_count.pdf')
    plt.show()
    print('Scatterplot saved to simplified_vs_filtered_point_count.pdf')

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
        a_aligned, b_aligned = align_curves(ca['filtered'], cb['filtered'], ALIGN_PAIR_START, ALIGN_PAIR_END)
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
plt.xlabel('Distance Difference')
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
plt.xlabel('Speedup')
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
            jerks = compute_jerk(curve)
            jerk_norms = np.linalg.norm(jerks, axis=1)
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

