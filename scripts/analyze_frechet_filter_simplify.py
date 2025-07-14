import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from src.frechet.computations import compute_jerk, filter_outlier_points, rdp_simplify, discrete_frechet_distance, \
    align_curves, velocity_grid_simplify, convert_curve_lonlat_to_xy, convert_curve_xy_to_lonlat
from src.frechet.visualization import plot_curves, plot_curve_with_special_points, plot_alignment, \
    plot_gps_density_and_intervals, plot_jerk_norms_with_outliers
from tqdm import tqdm
from simplification.cutil import simplify_coords_vw, simplify_coords
import io
from PIL import Image
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from src.frechet.visualization import plot_processing_stages

# df = pd.read_csv("../data/taxi.csv")
# df = df.drop_duplicates()
# df.to_csv("../data/taxi_no_duplicates.csv", index=False)

# --- CONFIG ---
#CSV_PATH = '../data/taxi.csv'  #  dataset
CSV_PATH = '../data/taxi_no_duplicates.csv'  # dataset without duplicates
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
    df['hour'] = df['hour'].astype(str).str.zfill(6)
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + df['hour'], format='%Y%m%d%H%M%S', errors='coerce')
    df = df.dropna(subset=['datetime'])
    df = df.sort_values(['id', 'datetime'])

plot_gps_density_and_intervals(df)

curves = {}
all_jerks = []
for id_val in tqdm(unique_ids, desc='Jerk computation and simplification'):
    subdf = df[df['id'] == id_val].copy()
    curve_lonlat = list(zip(subdf['lon'], subdf['lat']))
    if len(curve_lonlat) < 4:
        continue
    # Compute jerk
    #curve_lonlat = np.array(curve)
    curve_xy = convert_curve_lonlat_to_xy(curve_lonlat)
    curve_times = subdf['datetime'].values
    jerks = compute_jerk(curve_xy, times=curve_times) # m/s^3
    jerk_norms = np.linalg.norm(jerks, axis=1) #Euclidean norm
    jerk_norms = np.array(jerk_norms)

    all_jerks.extend(jerk_norms)
    # Filter outliers
    filtered_curve_xy, keep_idx, outlier_idx = filter_outlier_points(curve_xy, jerk_norms, sigma=SIGMA)
    filtered_curve_lonlat = convert_curve_xy_to_lonlat(filtered_curve_xy)


    print("first 10 outlier indices:", outlier_idx[:10])
    # Simplify using velocity_grid_simplify
    filtered_times = curve_times[keep_idx]
    simplified_curve_xy, augmented_curve_xy = velocity_grid_simplify(filtered_curve_xy, filtered_times, c=1)
    curves[id_val] = {
        'original_lonlat': curve_lonlat,
        'original_xy': curve_xy,
        'curve_times': curve_times,
        'filtered_lonLat': filtered_curve_lonlat,
        'filtered_xy': filtered_curve_xy,
        'augmented_xy': augmented_curve_xy,
        'simplified_lonLat': simplified_curve_xy,
        'simplified_xy': simplified_curve_xy,
        'jerks': jerks,
        'jerk_norms': jerk_norms
    }
    # # --- Save jerk trend plot with outliers highlighted ---


    filtered_times = curve_times[keep_idx]
    filtered_jerks = compute_jerk(filtered_curve_xy, times=filtered_times)
    filtered_jerk_norms = np.linalg.norm(filtered_jerks, axis=1)

    plot_jerk_norms_with_outliers(jerk_norms, outlier_idx, id_val, SIGMA,filtered_jerk_norms=filtered_jerk_norms,  original_curve=curve_lonlat)

# --- SIMPLIFIED VS FILTERED POINT COUNT PAIRED PLOT (LOG SCALE) ---
curve_ids = list(curves.keys())
filtered_lens = [len(curves[id_val]['filtered_xy']) for id_val in curve_ids]
simplified_lens = [len(curves[id_val]['simplified_xy']) for id_val in curve_ids]

# --- PAIRWISE FRECHET ANALYSIS ---
results = []
t0_all = time.time()
total_pairs = len(curve_ids) * (len(curve_ids) - 1) // 2
pair_count = 0
# Optional: limit the number of pairs processed
MAX_PAIRS = None  # Set to an integer to limit, or None for all pairs
IS_SHOW_AlIGNMENT = True  # Set to True to visualize alignment

for i in range(len(curve_ids)):
    for j in range(i+1, len(curve_ids)):
        if MAX_PAIRS is not None and pair_count >= MAX_PAIRS:
            break
        pair_count += 1
        print(f"[Pair {pair_count}/{total_pairs}] Comparing id {curve_ids[i]} vs {curve_ids[j]}")
        t_pair = time.time()
        id_a, id_b = curve_ids[i], curve_ids[j]
        ca, cb = curves[id_a], curves[id_b]
        # Align filtered curves before Fréchet calculation
        a_pre = ca['filtered_xy']
        b_pre = cb['filtered_xy']
        a_aligned, b_aligned = align_curves(a_pre, b_pre, ALIGN_PAIR_START, ALIGN_PAIR_END)
        if IS_SHOW_AlIGNMENT:         # Visualization: pre-aligned vs aligned
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            plot_alignment(axs[0], a_pre, b_pre, f'{id_a} (pre)', f'{id_b} (pre)', 'Pre-aligned  (before simplification)')
            axs[0].legend()
            axs[0].axis('equal')
            axs[0].grid(True, linestyle='--', alpha=0.7)
            plot_alignment(axs[1], a_aligned, b_aligned, f'{id_a} (aligned)', f'{id_b} (aligned)', 'Aligned curves (before simplification)')
            axs[1].legend()
            axs[1].axis('equal')
            axs[1].grid(True, linestyle='--', alpha=0.5)
            fig.suptitle(f'Alignment visualization: {id_a} vs {id_b}')
            plt.tight_layout()
            plt.savefig(f'alignment_vis_{id_a}_vs_{id_b}.pdf')
            plt.close(fig)
        # Filtered (aligned)
        t0 = time.time()
        d_filt = discrete_frechet_distance(a_aligned, b_aligned)
        t1 = time.time()
        # Simplified (align simplified curves as well)
        a_simp_aligned, b_simp_aligned = align_curves(ca['simplified_xy'], cb['simplified_xy'], ALIGN_PAIR_START, ALIGN_PAIR_END)
        t2 = time.time()
        d_simp = discrete_frechet_distance(a_simp_aligned, b_simp_aligned)
        t3 = time.time()
        results.append({
            'id_a': id_a, 'id_b': id_b,
            'd_filt': d_filt,
            'd_simp': d_simp,
            't_filt': t1-t0,
            't_simp': t3-t2
        })
        print(f"    Done in {time.time() - t_pair:.2f} seconds.")
    if MAX_PAIRS is not None and pair_count >= MAX_PAIRS:
        break
print(f"All pairs processed in {time.time() - t0_all:.2f} seconds.")

# --- DEVIATION ANALYSIS ---
results_df = pd.DataFrame(results)
results_df['d_simp_vs_filt'] = results_df['d_simp'] / results_df['d_filt']
results_df['speedup_simp_vs_filt'] = results_df['t_simp'] / results_df['t_filt']
ratios = results_df['d_simp_vs_filt']
ratios = ratios[np.isfinite(ratios) & (results_df['d_filt'] > 0)]

plt.figure(figsize=(8, 5))
sorted_ratios = np.sort(ratios)
yvals = np.arange(1, len(sorted_ratios)+1) / len(sorted_ratios)
plt.step(sorted_ratios, yvals * 100, where='post', label='Simplified/Filtered Fréchet Ratio')
plt.axvline(1, color='red', linestyle='--', label='Ratio = 1 (Exact)')
plt.xlabel('Simplified dF(P, Q) / filtered d_F(P, Q)')
plt.ylabel('Cumulative Percentage of Pairs (%)')
plt.title('Simplification Algorithm / True Fréchet Distance Ratios')
plt.legend()
plt.tight_layout()
plt.savefig('simplified_vs_frechet_ratio.pdf')
plt.show()
print('simplification/Fréchet ratios saved to simplified_vs_frechet_ratio.pdf')

# Add metadata to the PDF report
metadata_text = f"Data file: {os.path.basename(CSV_PATH)}\n" \
                f"Number of pairs: {pair_count}\n" \
                f"Simplification: velocity_based_simplify \n" \
                f"Aligned at start: {ALIGN_PAIR_START}, end: {ALIGN_PAIR_END}"
plt.gcf().text(0.01, 0.98, metadata_text, fontsize=8, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
plt.savefig('report_frechet.pdf')

# --- RANDOM PAIR CURVE PLOTS ---
if len(results) > 0:
    # Pick a random pair from the results (within MAX_PAIRS)
    random_result = random.choice(results)
    id_a = random_result['id_a']
    id_b = random_result['id_b']
    ca, cb = curves[id_a], curves[id_b]
    plot_curves(ca['original_lonlat'], cb['original_lonlat'], label_a=f'Original {id_a}', label_b=f'Original {id_b}', title='Original Curves')
    plot_curves(ca['filtered_lonLat'], cb['filtered_lonLat'], label_a=f'Filtered {id_a}', label_b=f'Filtered {id_b}', title='Filtered Curves')
    plot_curves(ca['simplified_lonLat'], cb['simplified_lonLat'], label_a=f'Simplified {id_a}', label_b=f'Simplified {id_b}', title='Simplified Curves')

# --- SUMMARY ---
finite_diff = results_df['d_simp_vs_filt'][np.isfinite(results_df['d_simp_vs_filt'])]
finite_speedup = results_df['speedup_simp_vs_filt'][np.isfinite(results_df['speedup_simp_vs_filt'])]
print('Filtered vs Simplified deviation summary:')
print(finite_diff.describe())
print('Speedup summary:')
print(finite_speedup.describe())
print('Analysis complete. Histograms saved to frechet_deviation_histograms.pdf')


# --- FRECHET DISTANCE COMPARISON simplification from the filtered  ---
frechet_results = []
for id_val in curves:
    if 'simplified_xy' not in curves[id_val] or 'filtered_xy' not in curves[id_val]:
        continue
    filtered = curves[id_val]['filtered_xy']
    simplified = curves[id_val]['simplified_xy']
    d_filt_vs_simp = discrete_frechet_distance(filtered, simplified)
    frechet_results.append({
        'id': id_val,
        'd_filt_vs_simp': d_filt_vs_simp,
        'n_filtered': len(filtered),
        'n_simplified': len(simplified),
        'simplified_percentage': len(simplified) / len(filtered) * 100 if len(filtered) > 0 else np.nan
    })

# --- VISUALIZE FRECHET DISTANCE COMPARISON ---
frechet_df = pd.DataFrame(frechet_results)
plt.figure(figsize=(8,6))
plt.scatter(frechet_df['simplified_percentage'], frechet_df['d_filt_vs_simp'], label='Filtered vs velocity_based_simplify', color='green')
plt.xlabel('simplified_percentage')
plt.ylabel('Fréchet Distance (m)')
plt.title('Fréchet Distance of Filtered vs velocity_based_simplify')
plt.legend()
plt.tight_layout()
plt.savefig('frechet_distance_comparison.pdf')
plt.show()
print('Fréchet distance comparison saved to frechet_distance_comparison.pdf')

# --- PDF REPORT SETUP ---
report_pdf_path = 'analysis_report.pdf'
with PdfPages(report_pdf_path) as pdf:

    # --- JERK HISTOGRAM (SEABORN, CUMULATIVE) ---
    plt.figure(figsize=(8, 4))
    sns.histplot(all_jerks, bins=50, color='teal', kde=True, stat='density', alpha=0.7, label='Jerk Norms')
    sns.ecdfplot(all_jerks, color='black', label='Cumulative')
    plt.xlabel('Jerk Norm (m/s³)')
    plt.ylabel('Density / Cumulative')
    plt.title('Histogram & Cumulative of Jerk Norms (All Curves)')
    plt.legend()
    plt.tight_layout()
    pdf.savefig(); plt.close()

    # --- CUMULATIVE DISTRIBUTIONS: FILTERED VS SIMPLIFIED ---
    plt.figure(figsize=(8, 4))
    sns.ecdfplot(filtered_lens, label='Filtered')
    sns.ecdfplot(simplified_lens, label='Simplified')
    plt.legend()
    plt.title('Cumulative Distribution of Point Counts')
    plt.xlabel('Number of Points')
    plt.tight_layout()
    pdf.savefig(); plt.close()

    # --- FRECHET DEVIATION HISTOGRAMS (OVERLAY) ---
    plt.figure(figsize=(8, 4))
    sns.histplot(results_df['d_simp_vs_filt'], bins=30, color='purple', kde=True, stat='density', alpha=0.7, label='Simplified - Filtered')
    plt.axvline(results_df['d_simp_vs_filt'].mean(), color='red', linestyle='--', label='Mean')
    plt.axvline(results_df['d_simp_vs_filt'].median(), color='blue', linestyle=':', label='Median')
    plt.legend()
    plt.title('Deviation in Frechet Distance (Simplified / Filtered)')
    plt.xlabel('Distance Difference (degrees)')
    plt.tight_layout()
    pdf.savefig(); plt.close()

    # --- CUMULATIVE: FRECHET DEVIATION ---
    plt.figure(figsize=(8, 4))
    sns.ecdfplot(results_df['d_simp_vs_filt'], label='Deviation')
    plt.axvline(results_df['d_simp_vs_filt'].mean(), color='red', linestyle='--', label='Mean')
    plt.axvline(results_df['d_simp_vs_filt'].median(), color='blue', linestyle=':', label='Median')
    plt.legend()
    plt.title('Cumulative Deviation in Frechet Distance')
    plt.xlabel('Distance Difference (m)')
    plt.tight_layout()
    pdf.savefig(); plt.close()

    # --- SUMMARY TABLE: FRECHET DEVIATION ---
    summary_frechet = results_df['d_simp_vs_filt'].describe().to_frame().T
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    tbl = pd.plotting.table(ax, summary_frechet, loc='center', colWidths=[0.1]*len(summary_frechet.columns))
    plt.title('Summary Statistics: Frechet Deviation')
    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # --- REPRESENTATIVE CURVES: STEP-BY-STEP VISUALS ---
    # Pick 3: median, min, max deviation
    rep_ids = results_df.iloc[results_df['d_simp_vs_filt'].abs().sort_values().index[[0, len(results_df)//2, -1]]][['id_a', 'id_b']].values
    for id_a, id_b in rep_ids:
        ca, cb = curves[id_a], curves[id_b]
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].plot(*zip(*ca['original_lonlat']), '-o', label='Original')
        axs[0].set_title('original_lonlat')
        axs[1].plot(*zip(*ca['filtered_lonLat']), '-o', label='Filtered', color='blue')
        axs[1].set_title('Filtered')
        axs[2].plot(*zip(*ca['simplified_lonLat']), '-o', label='Simplified', color='green')
        axs[2].set_title('Simplified_lonLat')
        a_aligned, b_aligned = align_curves(ca['filtered_xy'], cb['filtered_xy'], True, True)
        axs[3].plot(*zip(*a_aligned), '-o', label='Frechet-Aligned', color='purple')
        axs[3].set_title('Frechet-Aligned')
        for ax in axs: ax.legend(); ax.grid(True, linestyle='--', alpha=0.5)
        plt.suptitle(f'Representative Curve: {id_a}')
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    # --- FRECHET DISTANCE COMPARISON SCATTER ---
    plt.figure(figsize=(8,6))
    plt.scatter(frechet_df['n_filtered'], frechet_df['d_filt_vs_simp'], label='Filtered vs velocity_grid_simplify', color='green')
    plt.xlabel('Number of Points (Filtered)')
    plt.ylabel('Fréchet Distance (degrees)')
    plt.title('Fréchet Distance: Filtered vs velocity_grid_simplify')
    plt.legend()
    plt.tight_layout()
    pdf.savefig(); plt.close()

    # --- FINAL COVER PAGE ---
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis('off')
    plt.text(0.5, 0.5, 'Trajectory Simplification & Frechet Analysis Report\nGenerated on: ' + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
             ha='center', va='center', fontsize=14)
    pdf.savefig(fig); plt.close(fig)

print(f'Full analysis report saved to {report_pdf_path}')


# --- PDF REPORT: MAIN CURVE PROCESSING VISUALIZATION ---
with PdfPages('curve_processing_stages.pdf') as pdf:
    for id_val in curves:
        original = np.array(curves[id_val]['original_xy'])
        filtered = np.array(curves[id_val]['filtered_xy'])
        augmented = np.array(curves[id_val]['augmented_xy'])
        simplified = np.array(curves[id_val]['simplified_xy'])

        plot_processing_stages(original, filtered, augmented, simplified, curve_id=id_val, save_path=None)
        plt.close('all')
