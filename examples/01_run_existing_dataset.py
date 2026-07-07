#!/usr/bin/env python3
"""
Example 01: Run an Existing Dataset

Demonstrates how to run the full churn prediction pipeline
on a registered dataset using the Python API.

Usage:
    cd project_root
    python examples/01_run_existing_dataset.py
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import ChurnFramework


def main():
    fw = ChurnFramework()

    # List available datasets
    print("Available datasets:", fw.list_datasets())

    # Run the full pipeline on Olist
    print("\nRunning pipeline on 'olist'...")
    result = fw.run("olist")

    # Print summary
    print(f"\nDataset:     {result.get('dataset', 'N/A')}")
    print(f"Ecosystem:   {result.get('ecosystem_type', 'N/A')}")
    print(f"Churn Rate:  {result.get('churn_rate', 0):.1%}")
    print(f"Best Model:  {result.get('best_model', 'N/A')}")
    print(f"Duration:    {result.get('duration_seconds', 0):.1f}s")

    # The result dict contains:
    # - models: trained model objects
    # - metrics: evaluation metrics per model
    # - shap_values: feature importance
    # - calibration: calibration data
    # - risk_scores: customer risk scores
    # - segments: customer segments
    # - Many more...
    print(f"\nResult keys: {list(result.keys())}")


if __name__ == "__main__":
    main()
