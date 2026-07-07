#!/usr/bin/env python3
"""
Example 02: Register a New Dataset

Demonstrates how to register a new dataset from a CSV file,
inspect its columns, and generate a reusable YAML configuration.

Usage:
    cd project_root
    python examples/02_register_new_dataset.py
"""
import sys
import os
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import ChurnFramework


def create_sample_csv():
    """Create a sample CSV for demonstration."""
    os.makedirs("/tmp/churn_example", exist_ok=True)
    path = "/tmp/churn_example/sample_orders.csv"

    df = pd.DataFrame({
        "customer_id": range(500),
        "order_date": pd.date_range("2022-01-01", periods=500, freq="D"),
        "order_value": [25.0, 50.0, 75.0, 100.0, 30.0] * 100,
        "product_category": ["Electronics", "Clothing", "Food", "Home", "Sports"] * 100,
        "payment_method": ["credit_card", "debit_card", "paypal", "credit_card", "cash"] * 100,
    })
    df.to_csv(path, index=False)
    return path


def main():
    # Create a sample CSV
    csv_path = create_sample_csv()
    print(f"Created sample CSV: {csv_path}")

    fw = ChurnFramework()

    # Method 1: Inspect the CSV without registering
    print("\n--- Inspecting CSV ---")
    profile = fw.profile_csv(csv_path)
    print(f"Rows: {profile['n_rows']}, Columns: {profile['n_columns']}")
    print(f"Customer ID: {profile.get('inferred_customer_id', 'N/A')}")
    print(f"Timestamp: {profile.get('inferred_event_time', 'N/A')}")
    print(f"Transaction Value: {profile.get('inferred_transaction_value', 'N/A')}")

    # Method 2: Register the dataset (generates YAML config)
    print("\n--- Registering Dataset ---")
    config_path = fw.register_dataset(
        csv_path=csv_path,
        name="sample_orders",
        ecosystem="transactional_marketplace",
        source_url="https://example.com/sample-data",
        citation="Example Dataset (2026)",
        output="/tmp/churn_example/sample_orders.yaml",
    )
    print(f"Config written to: {config_path}")

    # Show the generated config
    print("\n--- Generated Config ---")
    with open(config_path) as f:
        print(f.read())

    # Method 3: Check readiness
    print("\n--- Readiness Check ---")
    from src.wizard import inspect_csv, generate_readiness_report
    inspection = inspect_csv(csv_path)
    report = generate_readiness_report(inspection)
    print(report.summary())


if __name__ == "__main__":
    main()
