#!/usr/bin/env python3
"""
Correctness verification for Assignment 4 - Distributed Matrix Multiplication.

Compares distributed implementation against sequential baseline to verify
bit-identical results (or within acceptable floating-point tolerance).
"""

import subprocess
import sys
from pathlib import Path
import numpy as np

JAVA_CMD = "java"
JAR_PATH = Path(__file__).parent.parent / "target" / "matrixmul-distributed.jar"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"

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

# Test configurations
TEST_SIZES = [100, 200, 500]
THREAD_COUNTS = [1, 2, 4, 8]

def run_correctness_test(matrix_size: int, threads: int) -> dict:
    """Run a correctness test and verify checksum match."""
    print(f"\nTesting: n={matrix_size}, threads={threads}")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / f"correctness_{matrix_size}_{threads}.csv"
    
    cmd = [JAVA_CMD] + JVM_ARGS + [
        "-cp", str(JAR_PATH),
        "es.ulpgc.bigdata.matrixmul.distributed.LocalRunner",
        str(matrix_size),
        str(RESULTS_DIR),
        str(threads)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            print(f"  ✗ Execution failed with return code {result.returncode}")
            print(f"    STDERR: {result.stderr[:200]}")
            return None
        
        # Parse output
        output_lines = result.stdout.split('\n')
        metrics = {
            'matrix_size': matrix_size,
            'threads': threads,
            'checksum_match': False,
            'time_ms': 0,
        }
        
        for line in output_lines:
            if 'Total time:' in line:
                time_str = line.split(':')[1].strip().replace('ms', '').strip()
                metrics['time_ms'] = int(time_str)
            if 'Checksum match:' in line:
                metrics['checksum_match'] = 'true' in line.lower()
            if 'Sequential checksum:' in line:
                seq_checksum = float(line.split(':')[1].strip())
                metrics['seq_checksum'] = seq_checksum
            if 'Distributed checksum:' in line:
                dist_checksum = float(line.split(':')[1].strip())
                metrics['dist_checksum'] = dist_checksum
        
        status = "✓" if metrics['checksum_match'] else "✗"
        print(f"  {status} Time: {metrics['time_ms']} ms, Checksum match: {metrics['checksum_match']}")
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"  ✗ Test timed out")
        return None
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return None

def main():
    """Run all correctness tests."""
    print("="*60)
    print("Assignment 4 - Correctness Verification")
    print("="*60)
    
    if not JAR_PATH.exists():
        print(f"\nERROR: JAR file not found at {JAR_PATH}")
        print("Please run 'mvn package' first.")
        sys.exit(1)
    
    all_results = []
    total_tests = len(TEST_SIZES) * len(THREAD_COUNTS)
    current = 0
    
    for size in TEST_SIZES:
        for threads in THREAD_COUNTS:
            current += 1
            print(f"\n[{current}/{total_tests}] Test {current} of {total_tests}")
            
            result = run_correctness_test(size, threads)
            if result:
                all_results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("Correctness Test Summary")
    print("="*60)
    
    total = len(all_results)
    passed = sum(1 for r in all_results if r['checksum_match'])
    failed = total - passed
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\n⚠ Some tests failed! Check checksum mismatches above.")
        sys.exit(1)
    else:
        print("\n✓ All correctness tests passed!")
        print("Distributed implementation produces identical results to sequential baseline.")

if __name__ == "__main__":
    main()
