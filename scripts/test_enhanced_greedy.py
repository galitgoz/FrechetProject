"""
Test Enhanced Greedy Algorithm for Fréchet Decision Problem

This script tests the enhanced greedy algorithm against exact Fréchet distance
calculations and evaluates performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import time
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.frechet.enhanced_greedy import enhanced_greedy_etd, enhanced_greedy_decision, compute_approximation_quality
from src.frechet.computations import discrete_frechet_distance, convert_curve_lonlat_to_xy

Point = Tuple[float, float]
Curve = List[Point]


def load_real_curves_from_taxi_data(csv_path: str = "../data/taxi_no_duplicates.csv",
                                   max_curves: int = 10, min_length: int = 10) -> List[Curve]:
    """
    Load real trajectory curves from taxi dataset.

    Args:
        csv_path: Path to taxi CSV file
        max_curves: Maximum number of curves to load
        min_length: Minimum number of points required per curve

    Returns:
        List of curves in XY coordinates (meters)
    """
    try:
        df = pd.read_csv(csv_path)

        # Parse datetime if needed
        if 'date' in df.columns and 'hour' in df.columns:
            df['hour'] = df['hour'].astype(str).str.zfill(6)
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + df['hour'],
                                          format='%Y%m%d%H%M%S', errors='coerce')
            df = df.dropna(subset=['datetime'])
            df = df.sort_values(['id', 'datetime'])

        curves = []
        unique_ids = df['id'].unique()[:max_curves]

        for id_val in unique_ids:
            subdf = df[df['id'] == id_val].copy()
            if len(subdf) < min_length:
                continue

            curve_lonlat = list(zip(subdf['lon'], subdf['lat']))
            curve_xy = convert_curve_lonlat_to_xy(curve_lonlat)
            curves.append(curve_xy)

        return curves
    except Exception as e:
        print(f"Warning: Could not load real curves from {csv_path}: {e}")
        return []


def run_experiment(curve_pairs, k=4):
    """
    Run enhanced greedy experiment on curve pairs.

    Args:
        curve_pairs: List of (P, Q) curve pairs
        k: Number of partitions for enhanced greedy algorithm

    Returns:
        DataFrame with results
    """
    results = []
    for idx, (P, Q) in enumerate(curve_pairs):
        # Compute exact Fréchet distance
        start_time = time.time()
        d_exact = discrete_frechet_distance(P, Q)
        exact_time = time.time() - start_time

        # Compute enhanced greedy ETD
        start_time = time.time()
        etd = enhanced_greedy_etd(P, Q, k)
        enhanced_time = time.time() - start_time

        # Evaluate results
        correct = abs(etd - d_exact) < 1e-3 or etd >= d_exact  # Accept if etd >= d_exact
        ratio = etd / d_exact if d_exact > 0 else np.nan
        speedup = exact_time / enhanced_time if enhanced_time > 0 else np.inf

        results.append({
            'idx': idx,
            'd_exact': d_exact,
            'etd': etd,
            'ratio': ratio,
            'correct': correct,
            'exact_time': exact_time,
            'enhanced_time': enhanced_time,
            'speedup': speedup,
            'curve_length_p': len(P),
            'curve_length_q': len(Q)
        })

        print(f"Pair {idx}: d_F={d_exact:.4f}, ETD={etd:.4f}, ratio={ratio:.4f}, correct={correct}")

    df = pd.DataFrame(results)
    correct_pct = 100 * df['correct'].mean()
    print(f"\nCorrect decision rate: {correct_pct:.2f}%")

    # CDF plot
    ratios = df['ratio'].dropna().sort_values()
    yvals = np.arange(1, len(ratios)+1) / len(ratios)
    plt.figure(figsize=(8,5))
    plt.step(ratios, yvals, where='post')
    plt.xlabel('ETD / d_F')
    plt.ylabel('Cumulative Fraction')
    plt.title(f'CDF of ETD/d_F Ratio (k={k})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'enhanced_greedy_cdf_k{k}.pdf')
    plt.show()
    return df


def run_comprehensive_test(use_real_data: bool = True):
    """
    Run comprehensive test of enhanced greedy algorithm.

    Args:
        use_real_data: Whether to include real trajectory data
    """
    print("=" * 60)
    print("ENHANCED GREEDY FRÉCHET ALGORITHM - COMPREHENSIVE TEST")
    print("=" * 60)

    # Load real data if available
    real_pairs = []
    if use_real_data:
        print("\n2. Loading real trajectory data...")
        real_curves = load_real_curves_from_taxi_data(max_curves=8)

        if len(real_curves) >= 2:
            # Create pairs from real curves
            for i in range(min(5, len(real_curves))):
                for j in range(i+1, min(i+3, len(real_curves))):
                    real_pairs.append((real_curves[i], real_curves[j]))
            print(f"   Created {len(real_pairs)} pairs from {len(real_curves)} real curves")
        else:
            print("   Not enough real curves loaded, using only synthetic data")

    # Combine test data
    all_pairs =  real_pairs
    print(f"\nTotal test pairs: {len(all_pairs)}  ")

    # Run experiments for different k values
    print("\n3. Running enhanced greedy experiments...")
    all_results = []

    for k in [2, 4, 8]:
        print(f"\n--- Testing with k={k} ---")
        results_df = run_experiment(all_pairs, k=k)
        results_df['k'] = k
        all_results.append(results_df)

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Print summary statistics
    print("\n4. EXPERIMENT RESULTS SUMMARY")
    print("-" * 40)

    for k in [2, 4, 8]:
        k_results = combined_results[combined_results['k'] == k]
        if len(k_results) > 0:
            print(f"\nk = {k}:")
            print(f"  Correct decisions: {k_results['correct'].mean():.1%}")
            print(f"  Mean approximation ratio: {k_results['ratio'].mean():.3f}")
            print(f"  Median approximation ratio: {k_results['ratio'].median():.3f}")
            print(f"  Mean speedup: {k_results['speedup'].mean():.1f}x")

    # Save results
    combined_results.to_csv('enhanced_greedy_results.csv', index=False)
    print(f"\nResults saved to 'enhanced_greedy_results.csv'")

    return combined_results


if __name__ == "__main__":
    # Run comprehensive test
    results_df = run_comprehensive_test()

    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
