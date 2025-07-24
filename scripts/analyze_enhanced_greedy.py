"""
Analysis and Visualization for Enhanced Greedy Fréchet Algorithm

This script generates comprehensive analysis reports, CDF plots, and performance
comparisons for the enhanced greedy algorithm results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.frechet.enhanced_greedy import enhanced_greedy_etd, enhanced_greedy_decision
from src.frechet.computations import discrete_frechet_distance


def plot_approximation_quality_cdf(results_df: pd.DataFrame, save_path: str = None):
    """
    Plot cumulative distribution function of ETD/d_F ratio across all queries.

    Args:
        results_df: DataFrame with experimental results
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))

    # Plot CDF for each k value
    k_values = sorted(results_df['k'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(k_values)))

    for i, k in enumerate(k_values):
        k_data = results_df[results_df['k'] == k]
        ratios = k_data['ratio'].dropna().sort_values()

        if len(ratios) > 0:
            # Calculate CDF
            y_vals = np.arange(1, len(ratios) + 1) / len(ratios)

            plt.step(ratios, y_vals * 100, where='post',
                    label=f'k={k} (n={len(ratios)})', color=colors[i], linewidth=2)

    # Add reference line at ratio = 1
    plt.axvline(1.0, color='red', linestyle='--', alpha=0.7,
                label='Perfect approximation (ratio=1)')

    plt.xlabel('ETD / d_F Ratio', fontsize=12)
    plt.ylabel('Cumulative Percentage (%)', fontsize=12)
    plt.title('Cumulative Distribution of Approximation Quality\n(ETD/d_F Ratio)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CDF plot saved to {save_path}")

    plt.show()


def plot_decision_accuracy_analysis(results_df: pd.DataFrame, save_path: str = None):
    """
    Plot decision accuracy analysis across different k values.

    Args:
        results_df: DataFrame with experimental results
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Correct decision rate by k
    ax1 = axes[0, 0]
    k_accuracy = results_df.groupby('k')['correct'].mean() * 100
    k_accuracy.plot(kind='bar', ax=ax1, color='skyblue', alpha=0.8)
    ax1.set_title('Decision Accuracy by Partition Count (k)')
    ax1.set_ylabel('Correct Decision Rate (%)')
    ax1.set_xlabel('Number of Partitions (k)')
    ax1.tick_params(axis='x', rotation=0)
    ax1.grid(True, alpha=0.3)

    # 2. Approximation ratio distribution
    ax2 = axes[0, 1]
    for k in sorted(results_df['k'].unique()):
        k_data = results_df[results_df['k'] == k]['ratio'].dropna()
        ax2.hist(k_data, bins=20, alpha=0.6, label=f'k={k}', density=True)
    ax2.set_title('Distribution of Approximation Ratios')
    ax2.set_xlabel('ETD / d_F Ratio')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Speedup analysis
    ax3 = axes[1, 0]
    speedup_data = results_df.groupby('k')['speedup'].agg(['mean', 'median', 'std'])
    speedup_data['mean'].plot(kind='bar', ax=ax3, color='lightgreen',
                             yerr=speedup_data['std'], capsize=5)
    ax3.set_title('Computational Speedup by k')
    ax3.set_ylabel('Speedup Factor (×)')
    ax3.set_xlabel('Number of Partitions (k)')
    ax3.tick_params(axis='x', rotation=0)
    ax3.grid(True, alpha=0.3)

    # 4. Accuracy vs curve length
    ax4 = axes[1, 1]
    results_df['avg_length'] = (results_df['curve_length_p'] + results_df['curve_length_q']) / 2
    length_bins = pd.cut(results_df['avg_length'], bins=5)
    accuracy_by_length = results_df.groupby(length_bins)['correct'].mean() * 100
    accuracy_by_length.plot(kind='bar', ax=ax4, color='orange', alpha=0.8)
    ax4.set_title('Accuracy vs Average Curve Length')
    ax4.set_ylabel('Correct Decision Rate (%)')
    ax4.set_xlabel('Average Curve Length (binned)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Decision accuracy analysis saved to {save_path}")

    plt.show()


def generate_performance_report(results_df: pd.DataFrame, save_path: str = None):
    """
    Generate a comprehensive performance report.

    Args:
        results_df: DataFrame with experimental results
        save_path: Optional path to save the report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ENHANCED GREEDY FRÉCHET ALGORITHM - PERFORMANCE REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Overall statistics
    total_pairs = len(results_df['idx'].unique())
    k_values = sorted(results_df['k'].unique())

    report_lines.append(f"Dataset Summary:")
    report_lines.append(f"  Total curve pairs tested: {total_pairs}")
    report_lines.append(f"  Partition values (k) tested: {k_values}")
    report_lines.append("")

    # Results by k value
    for k in k_values:
        k_data = results_df[results_df['k'] == k]

        report_lines.append(f"Results for k = {k}:")
        report_lines.append("-" * 40)

        # Decision accuracy
        correct_rate = k_data['correct'].mean() * 100
        report_lines.append(f"  Decision Accuracy: {correct_rate:.1f}%")

        # Approximation quality
        ratios = k_data['ratio'].dropna()
        mean_ratio = ratios.mean()
        median_ratio = ratios.median()
        std_ratio = ratios.std()
        min_ratio = ratios.min()
        max_ratio = ratios.max()

        report_lines.append(f"  Approximation Quality (ETD/d_F):")
        report_lines.append(f"    Mean ratio: {mean_ratio:.3f}")
        report_lines.append(f"    Median ratio: {median_ratio:.3f}")
        report_lines.append(f"    Std deviation: {std_ratio:.3f}")
        report_lines.append(f"    Range: [{min_ratio:.3f}, {max_ratio:.3f}]")

        # Performance metrics
        speedups = k_data['speedup'].replace([np.inf], np.nan).dropna()
        if len(speedups) > 0:
            mean_speedup = speedups.mean()
            median_speedup = speedups.median()
            report_lines.append(f"  Computational Performance:")
            report_lines.append(f"    Mean speedup: {mean_speedup:.1f}×")
            report_lines.append(f"    Median speedup: {median_speedup:.1f}×")

        # Quality assessment
        excellent_count = len(ratios[ratios <= 1.1])
        good_count = len(ratios[(ratios > 1.1) & (ratios <= 1.5)])
        poor_count = len(ratios[ratios > 1.5])

        report_lines.append(f"  Quality Assessment:")
        report_lines.append(f"    Excellent (≤1.1): {excellent_count}/{len(ratios)} ({excellent_count/len(ratios)*100:.1f}%)")
        report_lines.append(f"    Good (1.1-1.5): {good_count}/{len(ratios)} ({good_count/len(ratios)*100:.1f}%)")
        report_lines.append(f"    Poor (>1.5): {poor_count}/{len(ratios)} ({poor_count/len(ratios)*100:.1f}%)")
        report_lines.append("")

    # Recommendations
    report_lines.append("Recommendations:")
    report_lines.append("-" * 40)

    # Find best k value
    k_performance = results_df.groupby('k').agg({
        'correct': 'mean',
        'ratio': 'mean',
        'speedup': lambda x: x.replace([np.inf], np.nan).mean()
    })

    best_accuracy_k = k_performance['correct'].idxmax()
    best_ratio_k = k_performance['ratio'].idxmin()

    report_lines.append(f"  Best accuracy: k={best_accuracy_k} ({k_performance.loc[best_accuracy_k, 'correct']*100:.1f}%)")
    report_lines.append(f"  Best approximation: k={best_ratio_k} (ratio={k_performance.loc[best_ratio_k, 'ratio']:.3f})")

    if best_accuracy_k == best_ratio_k:
        report_lines.append(f"  Recommended k value: {best_accuracy_k} (optimal for both accuracy and quality)")
    else:
        report_lines.append(f"  Recommended k value: {best_accuracy_k} for accuracy, {best_ratio_k} for quality")

    report_lines.append("")
    report_lines.append("=" * 80)

    # Print report
    report_text = "\n".join(report_lines)
    print(report_text)

    # Save report
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to {save_path}")

    return report_text


def create_comparison_plots(results_df: pd.DataFrame, save_dir: str = "."):
    """
    Create a comprehensive set of comparison plots.

    Args:
        results_df: DataFrame with experimental results
        save_dir: Directory to save plots
    """
    print("Generating comprehensive analysis plots...")

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # 1. CDF plot
    cdf_path = os.path.join(save_dir, "enhanced_greedy_approximation_cdf.pdf")
    plot_approximation_quality_cdf(results_df, cdf_path)

    # 2. Decision accuracy analysis
    accuracy_path = os.path.join(save_dir, "enhanced_greedy_decision_analysis.pdf")
    plot_decision_accuracy_analysis(results_df, accuracy_path)

    # 3. Performance comparison scatter plot
    plt.figure(figsize=(10, 6))
    for k in sorted(results_df['k'].unique()):
        k_data = results_df[results_df['k'] == k]
        plt.scatter(k_data['d_exact'], k_data['etd'], alpha=0.6, label=f'k={k}', s=50)

    # Add perfect correlation line
    min_val = min(results_df['d_exact'].min(), results_df['etd'].min())
    max_val = max(results_df['d_exact'].max(), results_df['etd'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect correlation')

    plt.xlabel('Exact Fréchet Distance')
    plt.ylabel('Enhanced Trajectory Distance (ETD)')
    plt.title('ETD vs Exact Fréchet Distance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    scatter_path = os.path.join(save_dir, "enhanced_greedy_correlation.pdf")
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Correlation plot saved to {scatter_path}")

    # 4. Generate performance report
    report_path = os.path.join(save_dir, "enhanced_greedy_performance_report.txt")
    generate_performance_report(results_df, report_path)


def analyze_results_from_csv(csv_path: str = "enhanced_greedy_results.csv"):
    """
    Load and analyze results from a saved CSV file.

    Args:
        csv_path: Path to the results CSV file
    """
    try:
        results_df = pd.read_csv(csv_path)
        print(f"Loaded results from {csv_path}")
        print(f"Data shape: {results_df.shape}")

        # Create analysis plots and reports
        create_comparison_plots(results_df, save_dir="enhanced_greedy_analysis")

        return results_df

    except FileNotFoundError:
        print(f"Results file {csv_path} not found. Please run test_enhanced_greedy.py first.")
        return None


if __name__ == "__main__":
    # Try to analyze existing results, or suggest running tests first
    results_df = analyze_results_from_csv()

    if results_df is None:
        print("\nTo generate results, run:")
        print("python scripts/test_enhanced_greedy.py")
    else:
        print("\nAnalysis completed successfully!")
        print("Check the 'enhanced_greedy_analysis' directory for plots and reports.")
