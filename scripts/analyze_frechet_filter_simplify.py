import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import random
from src.frechet.computations import compute_jerk, filter_outlier_points, rdp_simplify, discrete_frechet_distance, \
    align_curves, velocity_grid_simplify, convert_curve_lonlat_to_xy, convert_curve_xy_to_lonlat, continuous_frechet_distance
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
#RDP_EPSILON = 0.0005  # Adjust as needed
ALIGN_PAIR_START = True  # align start for each pair
ALIGN_PAIR_END = False   # align end for each pair

# --- LOAD DATA ---
MAX_TAXIS = 10  # Set to an integer to limit the number of taxis loaded, or None for all
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
    curve_xy = convert_curve_lonlat_to_xy(curve_lonlat)
    curve_times = subdf['datetime'].values
    jerks = compute_jerk(curve_xy, times=curve_times) # m/s^3
    jerk_norms = np.linalg.norm(jerks, axis=1) #Euclidean norm
    jerk_norms = np.array(jerk_norms)
    all_jerks.extend(jerk_norms)
    # Filter outliers
    filtered_curve_xy, keep_idx, outlier_idx = filter_outlier_points(curve_xy, jerk_norms, sigma=SIGMA)
    filtered_curve_lonlat = convert_curve_xy_to_lonlat(filtered_curve_xy)

    # Simplify using velocity_grid_simplify
    filtered_times = curve_times[keep_idx]
    simplified_curve_xy, augmented_curve_xy = velocity_grid_simplify(filtered_curve_xy, filtered_times, c=0.25)
    simplified_lonLat=convert_curve_xy_to_lonlat(simplified_curve_xy)

    curves[id_val] = {
        'original_lonlat': curve_lonlat,
        'original_xy': curve_xy,
        'curve_times': curve_times,
        'filtered_lonLat': filtered_curve_lonlat,
        'filtered_xy': filtered_curve_xy,
        'augmented_xy': augmented_curve_xy,
        'augmented_lonLat': convert_curve_xy_to_lonlat(augmented_curve_xy),
        'simplified_lonLat': simplified_lonLat,
        'simplified_xy': simplified_curve_xy,
        'jerks': jerks,
        'jerk_norms': jerk_norms
    }
    # # --- Save jerk trend plot with outliers highlighted ---


    filtered_times = curve_times[keep_idx]
    filtered_jerks = compute_jerk(filtered_curve_xy, times=filtered_times)
    filtered_jerk_norms = np.linalg.norm(filtered_jerks, axis=1)

    plot_jerk_norms_with_outliers(jerk_norms, outlier_idx, id_val, SIGMA,filtered_jerk_norms=filtered_jerk_norms,  original_curve_lonlat=curve_lonlat)

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
        # Align augmented curves before Fréchet calculation
        a_pre = ca['augmented_xy']
        b_pre = cb['augmented_xy']
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
        # Augmented (aligned)
        t0 = time.time()
        d_aug = discrete_frechet_distance(a_aligned, b_aligned)
        t1 = time.time()
        # Simplified (align simplified curves as well)
        a_simp_aligned, b_simp_aligned = align_curves(ca['simplified_xy'], cb['simplified_xy'], ALIGN_PAIR_START, ALIGN_PAIR_END)
        t2 = time.time()
        d_simp = discrete_frechet_distance(a_simp_aligned, b_simp_aligned)
        t3 = time.time()
        results.append({
            'id_a': id_a, 'id_b': id_b,
            'd_aug': d_aug,
            'd_simp': d_simp,
            't_aug': t1-t0,
            't_simp': t3-t2
        })
        print(f"    Done in {time.time() - t_pair:.2f} seconds.")
    if MAX_PAIRS is not None and pair_count >= MAX_PAIRS:
        break
print(f"All pairs processed in {time.time() - t0_all:.2f} seconds.")

# --- DEVIATION ANALYSIS ---
results_df = pd.DataFrame(results)
results_df['d_simp_vs_aug'] = results_df['d_simp'] / results_df['d_aug']
results_df['speedup_simp_vs_aug'] = results_df['t_simp'] / results_df['t_aug']
ratios = results_df['d_simp_vs_aug']
ratios = ratios[np.isfinite(ratios) & (results_df['d_aug'] > 0)]

plt.figure(figsize=(8, 5))
sorted_ratios = np.sort(ratios)
yvals = np.arange(1, len(sorted_ratios)+1) / len(sorted_ratios)
plt.step(sorted_ratios, yvals * 100, where='post', label='Simplified/Augmented Fréchet Ratio')
plt.axvline(1, color='red', linestyle='--', label='Ratio = 1 (Exact)')
plt.xlabel('Simplified dF(P, Q) / Augmented d_F(P, Q)')
plt.ylabel('Cumulative Percentage of Pairs (%)')
plt.title('Simplification Algorithm / Augmented Fréchet Distance Ratios')
plt.legend()
plt.tight_layout()
plt.savefig('simplified_vs_Augmented_ratio.pdf')
plt.close()  # Close the figure instead of showing it
print('Simplified/Augmented distance ratios saved to simplified_vs_Augmented_ratio.pdf')

# --- RANDOM PAIR CURVE PLOTS ---
if len(results) > 0:
    # Pick a random pair from the results (within MAX_PAIRS)
    random_result = random.choice(results)
    id_a = random_result['id_a']
    id_b = random_result['id_b']
    ca, cb = curves[id_a], curves[id_b]
    plot_curves(ca['original_lonlat'], cb['original_lonlat'], label_a=f'Original {id_a}', label_b=f'Original {id_b}', title='Original Curves', xlabel="Longitude", ylabel="Latitude")
    plot_curves(ca['filtered_lonLat'], cb['filtered_lonLat'], label_a=f'Filtered {id_a}', label_b=f'Filtered {id_b}', title='Filtered Curves',xlabel="Longitude", ylabel="Latitude")
    plot_curves(ca['simplified_lonLat'], cb['simplified_lonLat'], label_a=f'Simplified {id_a}', label_b=f'Simplified {id_b}', title='Simplified Curves',xlabel="Longitude", ylabel="Latitude")

# --- SUMMARY ---
finite_diff = results_df['d_simp_vs_aug'][np.isfinite(results_df['d_simp_vs_aug'])]
finite_speedup = results_df['speedup_simp_vs_aug'][np.isfinite(results_df['speedup_simp_vs_aug'])]
print('Augmented vs Simplified deviation summary:')
print(finite_diff.describe())
print('Speedup summary:')
print(finite_speedup.describe())

# --- Simplification distance from the filtered curve  ---
simp_results = []
for id_val in curves:
    if 'simplified_xy' not in curves[id_val] or 'augmented_xy' not in curves[id_val]:
        continue
    filtered = curves[id_val]['filtered_xy']
    augmented_curve_xy= curves[id_val]['augmented_xy']
    simplified = curves[id_val]['simplified_xy']
    d_filt_vs_simp = discrete_frechet_distance(filtered, simplified)
    d_aug_vs_simp = discrete_frechet_distance(augmented_curve_xy, simplified)
    simp_results.append({
        'id': id_val,
        'd_filt_vs_simp': d_filt_vs_simp,
        'd_aug_vs_simp': d_aug_vs_simp,
        'n_filtered': len(filtered),
        'n_augmented': len(augmented_curve_xy),
        'n_simplified': len(simplified),
        'simplified_percentage': len(simplified) / len(filtered) * 100 if len(filtered) > 0 else np.nan
    })



# --- PDF REPORT SETUP ---
report_pdf_path = 'analysis_report.pdf'
with PdfPages(report_pdf_path) as pdf:
    # --- JERK HISTOGRAM (SEABORN, CUMULATIVE) ---
    plt.figure(figsize=(8, 4))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    # Histogram
    xmax= np.percentile(jerk_norms, 99)
    print("max jerk norm:", xmax)
    xmax= min(xmax, 10)  # Limit x-axis to 100 for better visibility
    n, bins, patches = ax1.hist(all_jerks, bins=50, density=True, alpha=0.7, label='Jerk Norms', color='teal')
    ax1.set_xlabel("Jerk Norm (m/s³)")
    ax1.set_ylabel("Density", color='teal')
    ax1.set_xlim([0, xmax])

    # Cumulative
    ax2 = ax1.twinx()
    ax2.plot(np.sort(all_jerks), np.linspace(0, 1, len(all_jerks)), color='black', label='Cumulative')
    ax2.set_ylabel("Cumulative", color='black')

    plt.title('Histogram & Cumulative of Jerk Norms (All Curves)')
    plt.legend()
    plt.tight_layout()
    pdf.savefig(); plt.close()

    # --- VISUALIZE FRECHET DISTANCE COMPARISON ---
    frechet_df = pd.DataFrame(simp_results)
    plt.figure(figsize=(8, 6))
    plt.scatter(frechet_df['simplified_percentage'], frechet_df['d_aug_vs_simp']/1000,
                label='Augmented vs velocity_based_simplify', color='green')
    plt.xlabel('simplified_percentage (% out of filtered curve length)')
    plt.ylabel('Fréchet Distance (km)')
    plt.title('Fréchet Distance of velocity_based_simplify to the Filtered curve ')
    plt.legend()
    plt.tight_layout()
    pdf.savefig();  plt.close()

    # --- CUMULATIVE: FRECHET RATIO ---
    plt.figure(figsize=(8, 4))
    sns.ecdfplot(results_df['d_simp_vs_aug'], label='Deviation')
    plt.axvline(results_df['d_simp_vs_aug'].mean(), color='red', linestyle='--', label='Mean')
    plt.axvline(results_df['d_simp_vs_aug'].median(), color='blue', linestyle=':', label='Median')
    plt.legend()
    plt.title('Cumulative ratio of simplified vs augmented Frechet Distance between 2 curves')
    plt.xlabel('Distance ratio simplified / augmented')
    plt.tight_layout()
    pdf.savefig(); plt.close()

    # --- SUMMARY TABLE: FRECHET DEVIATION ---
    summary_frechet = results_df['d_simp_vs_aug'].describe().to_frame().T
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    tbl = pd.plotting.table(ax, summary_frechet, loc='center', colWidths=[0.1]*len(summary_frechet.columns))
    plt.title('Summary Statistics: Frechet ratio between simplified  and augmented calculation')
    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # --- REPRESENTATIVE CURVES: STEP-BY-STEP VISUALS ---
    # Pick 3: median, min, max deviation
    rep_ids = results_df.iloc[results_df['d_simp_vs_aug'].abs().sort_values().index[[0, len(results_df)//2, -1]]][['id_a', 'id_b']].values
    for id_a, id_b in rep_ids:
        ca, cb = curves[id_a], curves[id_b]
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].plot(*zip(*ca['original_lonlat']), '-o', label='Original')
        axs[0].set_title('original_lonlat')
        axs[0].set_xlabel('Longitude')
        axs[0].set_ylabel('Latitude')
        axs[1].plot(*zip(*ca['filtered_lonLat']), '-o', label='Filtered', color='blue')
        axs[1].set_title('Filtered_lonLat')
        axs[1].set_xlabel('Longitude')
        axs[1].set_ylabel('Latitude')
        axs[2].plot(*zip(*ca['augmented_lonLat']), '-o', label='Simplified', color='green')
        axs[2].set_title('augmented_lonLat')
        axs[2].set_xlabel('Longitude')
        axs[2].set_ylabel('Latitude')
        # Last panel: simplified with original overlay
        axs[3].plot(*zip(*ca['original_lonlat']), '-o', color='gray', alpha=0.4, label='Original (background)', zorder=1)
        axs[3].plot(*zip(*ca['simplified_lonLat']), '-o', label='Simplified', color='green', zorder=2)
        axs[3].set_title('Simplified_lonLat')
        axs[3].set_xlabel('Longitude')
        axs[3].set_ylabel('Latitude')
        for ax in axs: ax.legend(); ax.grid(True, linestyle='--', alpha=0.5)
        plt.suptitle(f'Representative Curve: {id_a}')
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    # --- FINAL COVER PAGE ---
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis('off')
    plt.text(0.5, 0.5, 'Trajectory Simplification & Frechet Analysis Report\nGenerated on: ' + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
             ha='center', va='center', fontsize=14)
    # --- METADATA TEXT ---
    metadata_text = f"Data file: {os.path.basename(CSV_PATH)}\n" \
                    f"Number of pairs: {pair_count}\n" \
                    f"Simplification: velocity_based_simplify \n" \
                    f"Aligned at start: {ALIGN_PAIR_START}, end: {ALIGN_PAIR_END}"
    plt.gcf().text(0.01, 0.98, metadata_text, fontsize=8, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    pdf.savefig(fig); plt.close()

print(f'Full analysis report saved to {report_pdf_path}')
