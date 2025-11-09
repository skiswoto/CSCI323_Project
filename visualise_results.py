"""
Visualization script for K-Shortest Path benchmark results
Generates publication-quality charts for your report
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_data(filepath='benchmark_results.csv'):
    """Load and validate benchmark results"""
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} benchmark results from {filepath}")
        print(f"  Algorithms: {df['algorithm'].unique()}")
        print(f"  Buckets: {df['bucket'].unique()}")
        print(f"  K values: {sorted(df['K'].unique())}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found!")
        print("   Run 'python run_fast_benchmark.py' first to generate results.")
        return None


def plot_dijkstra_vs_astar_all_distances(df, output_dir='plots'):
    """
    Chart 1: Dijkstra vs A* across all distance categories
    Shows your main contribution
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Filter baseline algorithms
    baseline = df[(df['algorithm'].isin(['Dijkstra', 'A*'])) & 
                  (df['objective'] == 'distance')]
    
    if len(baseline) == 0:
        print("⚠️  No baseline data found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Runtime by distance
    ax = axes[0]
    bucket_order = ['short', 'medium', 'long']
    
    for alg in ['Dijkstra', 'A*']:
        data = baseline[baseline['algorithm'] == alg]
        means = data.groupby('bucket')['runtime_sec'].mean().reindex(bucket_order)
        stds = data.groupby('bucket')['runtime_sec'].std().reindex(bucket_order)
        
        ax.plot(bucket_order, means.values, marker='o', linewidth=2, 
                label=alg, markersize=8)
        ax.fill_between(range(len(bucket_order)), 
                        means - stds, means + stds, alpha=0.2)
    
    ax.set_xlabel('Query Distance Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Dijkstra vs A* Performance Across Distance Categories', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Speedup factor
    ax = axes[1]
    speedups = []
    for bucket in bucket_order:
        dijk_time = baseline[(baseline['algorithm'] == 'Dijkstra') & 
                            (baseline['bucket'] == bucket)]['runtime_sec'].mean()
        astar_time = baseline[(baseline['algorithm'] == 'A*') & 
                             (baseline['bucket'] == bucket)]['runtime_sec'].mean()
        speedup = dijk_time / astar_time if astar_time > 0 else 1
        speedups.append(speedup)
    
    bars = ax.bar(bucket_order, speedups, color=['#3498db', '#2ecc71', '#e74c3c'], 
                  alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='No speedup')
    
    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Query Distance Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor (Dijkstra / A*)', fontsize=12, fontweight='bold')
    ax.set_title('A* Speedup Over Dijkstra', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = f'{output_dir}/1_dijkstra_vs_astar_all_distances.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_nodes_expanded_comparison(df, output_dir='plots'):
    """
    Chart 2: Node expansion comparison (Dijkstra vs A*)
    Shows efficiency of A* heuristic
    """
    baseline = df[(df['algorithm'].isin(['Dijkstra', 'A*'])) & 
                  (df['objective'] == 'distance') &
                  (df['nodes_expanded'] > 0)]
    
    if len(baseline) == 0:
        print("⚠️  No node expansion data found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bucket_order = ['short', 'medium', 'long']
    
    # Plot 1: Absolute nodes expanded
    ax = axes[0]
    for alg in ['Dijkstra', 'A*']:
        data = baseline[baseline['algorithm'] == alg]
        means = data.groupby('bucket')['nodes_expanded'].mean().reindex(bucket_order)
        
        ax.plot(bucket_order, means.values, marker='s', linewidth=2, 
                label=alg, markersize=8)
    
    ax.set_xlabel('Query Distance Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Nodes Expanded', fontsize=12, fontweight='bold')
    ax.set_title('Node Expansion: Dijkstra vs A*', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Reduction percentage
    ax = axes[1]
    reductions = []
    for bucket in bucket_order:
        dijk_nodes = baseline[(baseline['algorithm'] == 'Dijkstra') & 
                             (baseline['bucket'] == bucket)]['nodes_expanded'].mean()
        astar_nodes = baseline[(baseline['algorithm'] == 'A*') & 
                              (baseline['bucket'] == bucket)]['nodes_expanded'].mean()
        reduction = 100 * (1 - astar_nodes / dijk_nodes) if dijk_nodes > 0 else 0
        reductions.append(reduction)
    
    bars = ax.bar(bucket_order, reductions, 
                  color=['#3498db', '#2ecc71', '#e74c3c'], 
                  alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar, reduction in zip(bars, reductions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{reduction:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Query Distance Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Node Expansion Reduction (%)', fontsize=12, fontweight='bold')
    ax.set_title('A* Node Expansion Reduction vs Dijkstra', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = f'{output_dir}/2_nodes_expanded_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_k_shortest_performance(df, output_dir='plots'):
    """
    Chart 3: K-shortest path algorithm performance
    Shows scaling with K and algorithm comparison
    """
    k_shortest = df[(df['algorithm'].isin(['Yens', 'PSB', 'SB'])) & 
                    (df['objective'] == 'distance') &
                    (df['bucket'] == 'short')]
    
    if len(k_shortest) == 0:
        print("⚠️  No K-shortest path data found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Runtime by K value - NOW WITH LARGER K
    ax = axes[0]
    k_values = sorted(k_shortest['K'].unique())
    
    colors = {'Yens': '#3498db', 'PSB': '#2ecc71', 'SB': '#e74c3c'}
    
    for alg in ['Yens', 'PSB', 'SB']:
        data = k_shortest[k_shortest['algorithm'] == alg]
        means = data.groupby('K')['runtime_sec'].mean()
        stds = data.groupby('K')['runtime_sec'].std()
        
        ax.plot(k_values, [means.get(k, 0) for k in k_values], 
                marker='o', linewidth=2, label=alg, 
                markersize=8, color=colors.get(alg))
        
        # Add error bars if we have std data
        if not stds.isna().all():
            ax.fill_between(k_values,
                           [means.get(k, 0) - stds.get(k, 0) for k in k_values],
                           [means.get(k, 0) + stds.get(k, 0) for k in k_values],
                           alpha=0.2, color=colors.get(alg))
    
    ax.set_xlabel('K (Number of Paths)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('K-Shortest Path Scalability (Short Queries)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Algorithm comparison at highest K
    ax = axes[1]
    max_k = max(k_values)
    algs = ['Yens', 'PSB', 'SB']
    means_at_max_k = []
    
    for alg in algs:
        data = k_shortest[(k_shortest['algorithm'] == alg) & 
                         (k_shortest['K'] == max_k)]
        if len(data) > 0:
            means_at_max_k.append(data['runtime_sec'].mean())
        else:
            means_at_max_k.append(0)
    
    bars = ax.bar(algs, means_at_max_k, 
                  color=['#3498db', '#2ecc71', '#e74c3c'], 
                  alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, mean in zip(bars, means_at_max_k):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Average Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_title(f'K-Shortest Algorithm Comparison at K={max_k}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = f'{output_dir}/3_k_shortest_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_single_vs_k_shortest(df, output_dir='plots'):
    """
    Chart 4: Cost of finding alternative routes
    Shows the dramatic increase in runtime for K-shortest
    """
    short_data = df[(df['bucket'] == 'short') & 
                    (df['objective'] == 'distance')]
    
    if len(short_data) == 0:
        print("⚠️  No short query data found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get baseline (K=1) from A*
    baseline_data = short_data[short_data['algorithm'] == 'A*']
    baseline_time = baseline_data['runtime_sec'].mean()
    
    # Get K-shortest data (using Yens as representative)
    k_shortest_data = short_data[short_data['algorithm'] == 'Yens']
    k_values = sorted(k_shortest_data['K'].unique())
    
    k_means = []
    k_factors = []
    for k in k_values:
        data = k_shortest_data[k_shortest_data['K'] == k]
        if len(data) > 0:
            mean_time = data['runtime_sec'].mean()
            k_means.append(mean_time)
            k_factors.append(mean_time / baseline_time if baseline_time > 0 else 1)
    
    # Plot K-shortest performance
    ax.plot(k_values, k_means, marker='o', linewidth=3, 
            markersize=10, color='#e74c3c', label='K-Shortest Paths')
    
    # Add baseline
    ax.axhline(y=baseline_time, linestyle='--', linewidth=2,
               color='#3498db', label='Single Path Baseline (K=1)')
    
    # Add slowdown factor labels
    for k, mean, factor in zip(k_values, k_means, k_factors):
        ax.annotate(f'{factor:.1f}x',
                   xy=(k, mean), xytext=(10, 10),
                   textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('K (Number of Paths)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Cost of Finding Alternative Routes (Short Queries)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    output_path = f'{output_dir}/4_single_vs_k_shortest.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_memory_usage(df, output_dir='plots'):
    """
    Chart 5: Memory usage comparison
    Separated into single-path algorithms and K-shortest algorithms
    """
    memory_data = df[(df['objective'] == 'distance') & 
                     (df['peak_memory_mb'] > 0)]
    
    if len(memory_data) == 0:
        print("⚠️  No memory data found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Single-path algorithms (ALL queries)
    ax = axes[0]
    single_path_data = []
    single_path_labels = ['Dijkstra', 'A*']
    
    for alg in ['Dijkstra', 'A*']:
        alg_data = memory_data[memory_data['algorithm'] == alg]
        if len(alg_data) > 0:
            single_path_data.append(alg_data['peak_memory_mb'].values)
        else:
            single_path_data.append([0])
    
    bp1 = ax.boxplot(single_path_data, labels=single_path_labels, patch_artist=True)
    
    colors_single = ['#3498db', '#2ecc71']
    for patch, color in zip(bp1['boxes'], colors_single):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Peak Memory Usage (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Single-Path Algorithms\n(All Query Distances)', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: K-shortest algorithms (SHORT queries only)
    ax = axes[1]
    k_shortest_data = []
    k_shortest_labels = ['Yens\n(K=3)', 'PSB\n(K=3)', 'SB\n(K=3)']
    
    for alg in ['Yens', 'PSB', 'SB']:
        alg_data = memory_data[(memory_data['algorithm'] == alg) & 
                              (memory_data['K'] == 3) &
                              (memory_data['bucket'] == 'short')]
        if len(alg_data) > 0:
            k_shortest_data.append(alg_data['peak_memory_mb'].values)
        else:
            k_shortest_data.append([0])
    
    bp2 = ax.boxplot(k_shortest_data, labels=k_shortest_labels, patch_artist=True)
    
    colors_k = ['#e74c3c', '#f39c12', '#9b59b6']
    for patch, color in zip(bp2['boxes'], colors_k):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Peak Memory Usage (MB)', fontsize=12, fontweight='bold')
    ax.set_title('K-Shortest Path Algorithms\n(Short Queries Only)', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = f'{output_dir}/5_memory_usage.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_scalability_comparison(df, output_dir='plots'):
    """
    Chart 6: Scalability comparison - LINEAR vs ACTUAL growth
    Shows whether K-shortest scales sub-linearly or super-linearly
    """
    k_shortest = df[(df['algorithm'] == 'Yens') & 
                    (df['objective'] == 'distance') &
                    (df['bucket'] == 'short')]
    
    if len(k_shortest) == 0:
        print("⚠️  No K-shortest data for scalability analysis")
        return
    
    k_values = sorted(k_shortest['K'].unique())
    
    if len(k_values) < 2:
        print("⚠️  Need at least 2 K values for scalability analysis")
        return
    
    # Calculate mean runtimes
    runtimes = [k_shortest[k_shortest['K'] == k]['runtime_sec'].mean() 
                for k in k_values]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Actual vs Linear scaling
    ax = axes[0]
    
    # Actual performance
    ax.plot(k_values, runtimes, marker='o', linewidth=3, 
            markersize=10, color='#e74c3c', label='Actual Performance')
    
    # Linear scaling (from K=3)
    baseline_k = k_values[0]
    baseline_time = runtimes[0]
    linear_times = [baseline_time * k / baseline_k for k in k_values]
    
    ax.plot(k_values, linear_times, linestyle='--', linewidth=2,
            color='#95a5a6', label='Linear Scaling (theoretical)')
    
    ax.set_xlabel('K (Number of Paths)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('K-Shortest Path Scalability: Actual vs Linear', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Scaling efficiency
    ax = axes[1]
    
    # Calculate efficiency: actual_growth / expected_linear_growth
    efficiencies = []
    for i in range(1, len(k_values)):
        actual_ratio = runtimes[i] / runtimes[0]
        expected_ratio = k_values[i] / k_values[0]
        efficiency = actual_ratio / expected_ratio
        efficiencies.append(efficiency)
    
    k_ratios = [f"{k_values[0]}→{k_values[i]}" for i in range(1, len(k_values))]
    
    colors_eff = ['#2ecc71' if e < 1 else '#e74c3c' for e in efficiencies]
    bars = ax.bar(k_ratios, efficiencies, color=colors_eff, 
                  alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, 
               label='Linear scaling baseline')
    
    # Add value labels
    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        label = "Sub-linear\n(efficient)" if eff < 1 else "Super-linear\n(inefficient)"
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.2f}x\n{label}',
                ha='center', va='bottom' if eff > 1 else 'top',
                fontweight='bold', fontsize=9)
    
    ax.set_xlabel('K Transition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Scaling Efficiency\n(<1 = sub-linear, >1 = super-linear)', 
                  fontsize=12, fontweight='bold')
    ax.set_title('Scaling Efficiency Analysis', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = f'{output_dir}/6_scalability_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_runtime_distribution(df, output_dir='plots'):
    """
    Chart 7: Runtime distribution box plots
    Shows variance in performance
    """
    short_data = df[(df['bucket'] == 'short') & 
                    (df['objective'] == 'distance')]
    
    if len(short_data) == 0:
        print("⚠️  No short query data found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get actual K values from data
    k_shortest_data = short_data[short_data['algorithm'] == 'Yens']
    k_values_in_data = sorted(k_shortest_data['K'].unique())
    
    # Prepare data - baseline + selected K values
    selected_k = [k for k in k_values_in_data if k in [3, 5, 10, 15, 20]][:4]  # Max 4 K values
    
    algorithms = ['Dijkstra', 'A*'] + [f'Yen\nK={k}' for k in selected_k]
    data_to_plot = []
    
    # Baseline algorithms
    for alg in ['Dijkstra', 'A*']:
        alg_data = short_data[short_data['algorithm'] == alg]
        if len(alg_data) > 0:
            data_to_plot.append(alg_data['runtime_sec'].values)
        else:
            data_to_plot.append([0])
    
    # K-shortest for selected K values
    for k in selected_k:
        alg_data = short_data[(short_data['algorithm'] == 'Yens') & 
                             (short_data['K'] == k)]
        if len(alg_data) > 0:
            data_to_plot.append(alg_data['runtime_sec'].values)
        else:
            data_to_plot.append([0])
    
    bp = ax.boxplot(data_to_plot, labels=algorithms, patch_artist=True)
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    for patch, color in zip(bp['boxes'], colors[:len(algorithms)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Runtime Distribution (Short Queries)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = f'{output_dir}/7_runtime_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def generate_summary_table(df, output_dir='plots'):
    """
    Generate summary statistics table
    """
    print("\n" + "="*80)
    print(" SUMMARY STATISTICS TABLE")
    print("="*80)
    
    distance_data = df[df['objective'] == 'distance']
    
    # Baseline algorithms
    print("\nBASELINE ALGORITHMS (All Query Distances):")
    print("-"*80)
    print(f"{'Algorithm':<15} {'Bucket':<10} {'Avg Runtime':<15} {'Avg Nodes':<15} {'N':<10}")
    print("-"*80)
    
    for alg in ['Dijkstra', 'A*']:
        for bucket in ['short', 'medium', 'long']:
            data = distance_data[(distance_data['algorithm'] == alg) & 
                                (distance_data['bucket'] == bucket)]
            if len(data) > 0:
                print(f"{alg:<15} {bucket:<10} "
                      f"{data['runtime_sec'].mean():<15.4f} "
                      f"{data['nodes_expanded'].mean():<15.0f} "
                      f"{len(data):<10}")
    
    # K-shortest algorithms
    print("\nK-SHORTEST PATH ALGORITHMS (Short Queries Only):")
    print("-"*80)
    print(f"{'Algorithm':<15} {'K':<5} {'Avg Runtime':<15} {'Avg Memory':<15} {'N':<10}")
    print("-"*80)
    
    short_data = distance_data[distance_data['bucket'] == 'short']
    
    # Get all K values in the data
    k_values_in_data = sorted(short_data[short_data['algorithm'].isin(['Yens', 'PSB', 'SB'])]['K'].unique())
    
    for alg in ['Yens', 'PSB', 'SB']:
        for k in k_values_in_data:
            data = short_data[(short_data['algorithm'] == alg) & 
                             (short_data['K'] == k)]
            if len(data) > 0:
                print(f"{alg:<15} {k:<5} "
                      f"{data['runtime_sec'].mean():<15.4f} "
                      f"{data['peak_memory_mb'].mean():<15.2f} "
                      f"{len(data):<10}")
    
    # Speedup summary
    print("\nSPEEDUP SUMMARY:")
    print("-"*80)
    
    for bucket in ['short', 'medium', 'long']:
        dijk = distance_data[(distance_data['algorithm'] == 'Dijkstra') & 
                            (distance_data['bucket'] == bucket)]
        astar = distance_data[(distance_data['algorithm'] == 'A*') & 
                             (distance_data['bucket'] == bucket)]
        
        if len(dijk) > 0 and len(astar) > 0:
            speedup = dijk['runtime_sec'].mean() / astar['runtime_sec'].mean()
            reduction = 100 * (1 - astar['nodes_expanded'].mean() / 
                              dijk['nodes_expanded'].mean())
            print(f"{bucket.capitalize():<10}: "
                  f"A* is {speedup:.2f}x faster, "
                  f"{reduction:.1f}% fewer nodes expanded")


def main():
    """Generate all visualizations"""
    print("\n" + "="*80)
    print(" BENCHMARK RESULTS VISUALIZATION")
    print("="*80)
    
    # Load data
    df = load_data('benchmark_results.csv')
    if df is None:
        return
    
    print(f"\n✓ Dataset summary:")
    print(f"  Total records: {len(df)}")
    print(f"  Algorithms: {', '.join(df['algorithm'].unique())}")
    print(f"  Query buckets: {', '.join(df['bucket'].unique())}")
    
    # Create output directory
    output_dir = 'plots'
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate all plots
    print(f"\nGenerating visualizations...")
    print("-"*80)
    
    plot_dijkstra_vs_astar_all_distances(df, output_dir)
    plot_nodes_expanded_comparison(df, output_dir)
    plot_k_shortest_performance(df, output_dir)
    plot_single_vs_k_shortest(df, output_dir)
    plot_memory_usage(df, output_dir)
    plot_scalability_comparison(df, output_dir)
    plot_runtime_distribution(df, output_dir)
    
    # Generate summary table
    generate_summary_table(df, output_dir)
    
    print("\n" + "="*80)
    print(" VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\n✓ All plots saved to '{output_dir}/' directory")
    print("\nGenerated plots:")
    print("  1. 1_dijkstra_vs_astar_all_distances.png - Main contribution")
    print("  2. 2_nodes_expanded_comparison.png - A* efficiency")
    print("  3. 3_k_shortest_performance.png - K-shortest scaling")
    print("  4. 4_single_vs_k_shortest.png - Cost of alternatives")
    print("  5. 5_memory_usage.png - Memory comparison (separated)")
    print("  6. 6_scalability_analysis.png - Scaling efficiency")
    print("  7. 7_runtime_distribution.png - Performance variance")
    print("\nUse these charts in your report!")


if __name__ == "__main__":
    main()