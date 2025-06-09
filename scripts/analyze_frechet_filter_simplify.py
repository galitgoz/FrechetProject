import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.frechet.computations import compute_jerk, filter_outlier_points, rdp_simplify, discrete_frechet_distance, align_curves
from tqdm import tqdm
from simplification.cutil import simplify_coords_vw, simplify_coords

# --- CONFIG ---
CSV_PATH = '../data/taxi.csv'  #  dataset
SIGMA = 3
RDP_EPSILON = 0.0005  # Adjust as needed
ALIGN_PAIR_START = True  # align start for each pair
ALIGN_PAIR_END = False   # align end for each pair

# --- LOAD DATA ---
df = pd.read_csv(CSV_PATH)
if 'date' in df.columns and 'hour' in df.columns:
    df = df.sort_values(['id', 'date', 'hour'])

curves = {}
for id_val in df['id'].unique():
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

# --- PAIRWISE FRECHET ANALYSIS ---
ids = list(curves.keys())
results = []
t0_all = time.time()
total_pairs = len(ids) * (len(ids) - 1) // 2
pair_count = 0

# Optional: limit the number of pairs processed
MAX_PAIRS = 500  # Set to an integer to limit, or None for no limit

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
plt.ylabel('Count')
plt.title('Simplified vs Filtered')
plt.legend()
plt.subplot(1,2,2)
plt.hist(results_df['speedup_simp_vs_filt'], bins=30, color='purple', alpha=0.7, label='Simplified/Filtered')
plt.xlabel('Speedup')
plt.ylabel('Count')
plt.title('Computation Speedup')
plt.legend()
plt.tight_layout()
plt.savefig('frechet_deviation_histograms.pdf')

# --- SUMMARY ---
print('Filtered vs Simplified deviation summary:')
print(results_df['diff_simp_vs_filt'].describe())
print('Speedup summary:')
print(results_df['speedup_simp_vs_filt'].describe())
print('Analysis complete. Histograms saved to frechet_deviation_histograms.pdf')

