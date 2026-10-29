#!/usr/bin/env python3
"""
Consolidate all CSV results into a single analysis file.
"""

import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

# Read all CSV files
results_dir = Path(__file__).parent.parent / "data" / "results"
all_data = defaultdict(list)

for csv_file in sorted(results_dir.glob("local_*.csv")):
    try:
        df = pd.read_csv(csv_file)
        # Extract matrix_size from filename
        parts = csv_file.stem.split('_')
        if len(parts) >= 2:
            matrix_size = int(parts[1])
            df['matrix_size'] = matrix_size
        all_data[csv_file.name] = df
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

print(f"Found {len(all_data)} CSV files")

# Consolidate all data
all_dfs = list(all_data.values())
if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal records: {len(combined)}")
    print(f"Columns: {list(combined.columns)}")
    
    # Group by configuration
    summary = combined.groupby(['matrix_size', 'threads']).agg({
        'total_ms': ['mean', 'std', 'count'],
        'compute_ms': ['mean', 'std'],
        'overhead_ms': ['mean', 'std']
    }).round(2)
    
    print("\n=== Results Summary ===")
    print(summary)
    
    # Save consolidated data
    output_file = results_dir / "consolidated_results.csv"
    combined.to_csv(output_file, index=False)
    print(f"\nSaved consolidated results to: {output_file}")
    
    # Save summary
    summary_file = results_dir / "summary_stats.csv"
    summary.to_csv(summary_file)
    print(f"Saved summary statistics to: {summary_file}")
else:
    print("No data found!")
