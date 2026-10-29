#!/usr/bin/env python3
"""
Plot results from Assignment 4 benchmark campaign.

Generates visualizations comparing:
- Sequential baseline vs distributed (local mode)
- Speedup and efficiency curves
- Timing breakdown by thread count
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
FIGURES_DIR = Path(__file__).parent.parent / "report" / "figures"

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

def load_campaign_data():
    """Load all campaign results from CSV files."""
    csv_files = list(RESULTS_DIR.glob("local_*.csv"))
    if not csv_files:
        print("No CSV files found in", RESULTS_DIR)
        return None
    
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Extract matrix_size from filename
            parts = csv_file.stem.split('_')
            if len(parts) >= 2:
                df['matrix_size'] = int(parts[1])
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not all_data:
        return None
    
    return pd.concat(all_data, ignore_index=True)

def plot_timing_comparison(df):
    """Plot timing comparison: sequential vs distributed."""
    print("\nGenerating timing comparison plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by matrix_size and threads, calculate mean time
    summary = df.groupby(['matrix_size', 'threads'])['total_ms'].mean().reset_index()
    
    # Plot for each matrix size
    for size in sorted(df['matrix_size'].unique()):
        data = summary[summary['matrix_size'] == size]
        ax.plot(data['threads'], data['total_ms'], marker='o', linewidth=2, label=f'n={size}')
    
    ax.set_xlabel('Thread Count', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Distributed Matrix Multiplication: Timing vs Thread Count', fontsize=14)
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "timing_vs_threads.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_speedup_efficiency(df):
    """Plot speedup and efficiency curves."""
    print("\nGenerating speedup and efficiency plots...")
    
    # Calculate speedup and efficiency
    # Speedup = T_seq / T_parallel (where T_seq is threads=1)
    # Efficiency = Speedup / threads
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for size in sorted(df['matrix_size'].unique()):
        data = df[df['matrix_size'] == size]
        
        # Get sequential time (threads=1)
        seq_times = data[data['threads'] == 1]['total_ms'].mean()
        
        if seq_times == 0:
            continue
        
        # Calculate speedup and efficiency
        summary = data.groupby('threads')['total_ms'].mean().reset_index()
        summary['speedup'] = seq_times / summary['total_ms']
        summary['efficiency'] = summary['speedup'] / summary['threads']
        
        # Plot speedup
        ax1.plot(summary['threads'], summary['speedup'], marker='o', 
                linewidth=2, label=f'n={size}')
        
        # Plot efficiency
        ax2.plot(summary['threads'], summary['efficiency'], marker='s', 
                linewidth=2, label=f'n={size}')
    
    # Ideal speedup line
    threads_range = df['threads'].unique()
    ax1.plot(threads_range, threads_range, 'k--', label='Ideal', alpha=0.5)
    
    ax1.set_xlabel('Thread Count', fontsize=12)
    ax1.set_ylabel('Speedup', fontsize=12)
    ax1.set_title('Speedup: T(1) / T(p)', fontsize=14)
    ax1.legend()
    ax1.set_xscale('log', base=2)
    
    ax2.set_xlabel('Thread Count', fontsize=12)
    ax2.set_ylabel('Efficiency', fontsize=12)
    ax2.set_title('Efficiency: Speedup / p', fontsize=14)
    ax2.legend()
    ax2.set_xscale('log', base=2)
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "speedup_efficiency.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_overhead_analysis(df):
    """Analyze overhead of distributed execution."""
    print("\nGenerating overhead analysis plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate overhead percentage
    for size in sorted(df['matrix_size'].unique()):
        data = df[df['matrix_size'] == size]
        seq_time = data[data['threads'] == 1]['total_ms'].mean()
        
        summary = data.groupby('threads')['total_ms'].mean().reset_index()
        summary['overhead_pct'] = ((summary['total_ms'] / (seq_time / summary['threads'])) - 1) * 100
        
        ax.plot(summary['threads'], summary['overhead_pct'], marker='o', 
               linewidth=2, label=f'n={size}')
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Zero overhead')
    ax.set_xlabel('Thread Count', fontsize=12)
    ax.set_ylabel('Overhead (%)', fontsize=12)
    ax.set_title('Distributed Execution Overhead vs Ideal Scaling', fontsize=14)
    ax.legend()
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "overhead_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def main():
    """Main plotting function."""
    print("="*60)
    print("Assignment 4 - Results Visualization")
    print("="*60)
    
    # Ensure figures directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading campaign data...")
    df = load_campaign_data()
    
    if df is None or df.empty:
        print("ERROR: No data found. Run experiments first with run_experiments.py")
        return
    
    print(f"✓ Loaded {len(df)} records")
    print(f"✓ Matrix sizes: {sorted(df['matrix_size'].unique())}")
    print(f"✓ Thread counts: {sorted(df['threads'].unique())}")
    
    # Generate plots
    plot_timing_comparison(df)
    plot_speedup_efficiency(df)
    plot_overhead_analysis(df)
    
    print("\n" + "="*60)
    print("Plotting complete!")
    print("="*60)

if __name__ == "__main__":
    main()
