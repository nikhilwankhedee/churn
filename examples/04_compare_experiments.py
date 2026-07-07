#!/usr/bin/env python3
"""
Example 04: Compare Experiments

Demonstrates how to compare experiment results across datasets
and analyze feature importance patterns.

Usage:
    cd project_root
    python examples/04_compare_experiments.py
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import ChurnFramework


def main():
    fw = ChurnFramework()

    # List experiment history
    print("Experiment History:")
    experiments = fw.list_experiments(limit=10)
    if experiments:
        for exp in experiments:
            print(f"  {exp.get('dataset', 'N/A')} - {exp.get('best_model', 'N/A')} "
                  f"({exp.get('churn_rate', 0):.1%} churn)")
    else:
        print("  No experiments found. Run some datasets first.")
        print("  Example: python examples/01_run_existing_dataset.py")

    # Compare across datasets (if we have multiple runs)
    available = fw.list_datasets()
    if len(available) >= 2:
        print(f"\nComparing experiments across: {available[:3]}")
        comparison = fw.compare(available[:3])

        if comparison is not None and not comparison.empty:
            print("\nComparison Results:")
            print(comparison.to_string(index=False))
        else:
            print("  No comparison data available.")

        # Feature group comparison
        print("\nFeature Group Comparison:")
        feature_comparison = fw.get_feature_comparison()
        if feature_comparison is not None and not feature_comparison.empty:
            print(feature_comparison.to_string(index=False))
        else:
            print("  No feature comparison data available.")
    else:
        print("\nNeed at least 2 datasets for comparison.")
        print("Run more datasets: python examples/01_run_existing_dataset.py")

    # Generate reports from a pipeline result
    print("\n--- Generating Reports ---")
    if experiments:
        # Re-run one dataset to get a fresh result for report generation
        result = fw.run("olist")
        reports = fw.generate_reports(result)

        for name, markdown in reports.items():
            print(f"\n  Report: {name}")
            # Print first 200 chars of each report
            preview = markdown[:200].replace("\n", "\n    ")
            print(f"    {preview}...")
    else:
        print("  Run a pipeline first to generate reports.")


if __name__ == "__main__":
    main()
