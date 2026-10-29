#!/usr/bin/env python3
"""
Campaign orchestrator for Assignment 4 - Distributed Matrix Multiplication.

Executes benchmarks with different thread counts and matrix sizes,
collects results, and generates CSV outputs for analysis.
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Configuration
JAVA_CMD = "java"
JAR_PATH = Path(__file__).parent.parent / "target" / "matrixmul-distributed.jar"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
INPUTS_DIR = Path(__file__).parent.parent / "data" / "inputs"

# JVM arguments required for Spark
JVM_ARGS = [
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
    "--add-opens=java.base/java.io=ALL-UNNAMED",
    "--add-opens=java.base/java.net=ALL-UNNAMED",
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/java.util=ALL-UNNAMED",
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
    "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
]

# Experiment configurations
MATRIX_SIZES = [500, 1000, 1500, 2000]
THREAD_COUNTS = [1, 2, 4, 8, 12]

def run_benchmark(matrix_size: int, thread_count: int, timestamp: str) -> dict:
    """Run a single benchmark configuration."""
    print(f"\n{'='*60}")
    print(f"Running: n={matrix_size}, threads={thread_count}")
    print(f"{'='*60}")
    
    output_file = RESULTS_DIR / f"local_{matrix_size}_{timestamp}.csv"
    
    cmd = [JAVA_CMD] + JVM_ARGS + [
        "-cp", str(JAR_PATH),
        "es.ulpgc.bigdata.matrixmul.distributed.LocalRunner",
        str(matrix_size),
        str(RESULTS_DIR),
        str(thread_count)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode != 0:
            print(f"ERROR: Benchmark failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return None
        
        # Parse output to extract metrics
        output_lines = result.stdout.split('\n')
        metrics = {
            'matrix_size': matrix_size,
            'threads': thread_count,
            'timestamp': timestamp,
            'output_file': str(output_file),
        }
        
        # Extract time and checksum from output
        for line in output_lines:
            if 'Total time:' in line:
                time_ms = int(line.split(':')[1].strip().replace('ms', '').strip())
                metrics['time_ms'] = time_ms
            if 'Checksum match:' in line:
                checksum_match = 'true' in line.lower()
                metrics['checksum_match'] = checksum_match
        
        print(f"✓ Completed in {metrics.get('time_ms', 'N/A')} ms")
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Benchmark timed out after 300 seconds")
        return None
    except Exception as e:
        print(f"ERROR: Exception during benchmark: {e}")
        return None

def run_campaign():
    """Run the complete benchmark campaign."""
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    print("="*60)
    print("Assignment 4 - Distributed Matrix Multiplication Campaign")
    print("="*60)
    print(f"Timestamp: {timestamp}")
    print(f"Matrix sizes: {MATRIX_SIZES}")
    print(f"Thread counts: {THREAD_COUNTS}")
    print(f"Results directory: {RESULTS_DIR}")
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check that JAR exists
    if not JAR_PATH.exists():
        print(f"\nERROR: JAR file not found at {JAR_PATH}")
        print("Please run 'mvn package' first.")
        sys.exit(1)
    
    # Run all configurations
    all_results = []
    total_configs = len(MATRIX_SIZES) * len(THREAD_COUNTS)
    current = 0
    
    for size in MATRIX_SIZES:
        for threads in THREAD_COUNTS:
            current += 1
            print(f"\n[{current}/{total_configs}] Configuration {current} of {total_configs}")
            
            result = run_benchmark(size, threads, timestamp)
            if result:
                all_results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("Campaign Summary")
    print("="*60)
    print(f"Total configurations: {total_configs}")
    print(f"Successful: {len(all_results)}")
    print(f"Failed: {total_configs - len(all_results)}")
    
    if all_results:
        # Save summary to JSON
        summary_file = RESULTS_DIR / f"campaign_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'configurations': all_results,
                'matrix_sizes': MATRIX_SIZES,
                'thread_counts': THREAD_COUNTS,
            }, f, indent=2)
        print(f"\nSummary saved to: {summary_file}")
    
    return all_results

if __name__ == "__main__":
    results = run_campaign()
    if not results:
        sys.exit(1)
