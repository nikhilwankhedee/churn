#!/usr/bin/env python3
"""
Example 03: Batch Execution

Demonstrates how to run the pipeline on multiple datasets
and compare results across ecosystems.

Usage:
    cd project_root
    python examples/03_batch_execution.py
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import ChurnFramework


def main():
    fw = ChurnFramework()

    # List available datasets
    datasets = fw.list_datasets()
    print(f"Available datasets: {datasets}")

    # Run batch on a subset (use all for full benchmark)
    selected = datasets[:3]  # First 3 for quick demo
    print(f"\nRunning batch on: {selected}")

    batch = fw.run_batch(datasets=selected, sensitivity=False)

    # Print results
    print(f"\n{'='*60}")
    print("Batch Execution Summary")
    print(f"{'='*60}")
    print(f"Successful: {batch.successful}")
    print(f"Failed:     {batch.failed}")
    print(f"Duration:   {batch.total_duration:.1f}s")

    for ds in selected:
        result = batch.results.get(ds, {})
        if "error" in result:
            print(f"\n  {ds}: FAILED - {result['error']}")
        else:
            churn_rate = result.get("churn_rate", 0)
            best_model = result.get("best_model", "N/A")
            duration = result.get("duration_seconds", 0)
            print(f"\n  {ds}:")
            print(f"    Best Model:  {best_model}")
            print(f"    Churn Rate:  {churn_rate:.1%}")
            print(f"    Duration:    {duration:.1f}s")

    # Benchmark table
    if batch.benchmark_table:
        print(f"\n{'='*60}")
        print("Benchmark Table")
        print(f"{'='*60}")
        for model, metrics in batch.benchmark_table.items():
            print(f"\n  {model}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")


if __name__ == "__main__":
    main()
