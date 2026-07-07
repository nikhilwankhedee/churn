#!/usr/bin/env python3
"""
Example 06: Custom Churn Strategy

Demonstrates how to create a custom churn labeling strategy
that integrates with the framework's registry.

Usage:
    cd project_root
    python examples/06_custom_strategy.py
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.churn.base import ChurnStrategy, ChurnResult
from src.churn.registry import register_churn_strategy_class
import pandas as pd
import numpy as np


class FrequencyBasedStrategy(ChurnStrategy):
    """Custom strategy: churn = customer's purchase frequency drops below median.

    This demonstrates how to create a custom churn labeling strategy
    that goes beyond simple inactivity windows.

    Logic:
    1. Compute each customer's inter-purchase intervals
    2. Calculate the median frequency across all customers
    3. Mark as churned if the most recent interval > 2x median
    """

    @property
    def name(self) -> str:
        return "frequency_based"

    @property
    def description(self) -> str:
        return (
            "Frequency-based churn: marks customers whose most recent "
            "inter-purchase interval exceeds 2x the median frequency."
        )

    @property
    def required_columns(self) -> list:
        return ["customer_id", "event_time"]

    def label(
        self,
        df: pd.DataFrame,
        cutoff_date: pd.Timestamp,
        customer_id_col: str = "customer_id",
        **kwargs,
    ) -> ChurnResult:
        """Create churn labels based on purchase frequency.

        Parameters
        ----------
        df : pd.DataFrame
            Standardized data with customer_id and event_time.
        cutoff_date : pd.Timestamp
            The reference date for computing labels.

        Returns
        -------
        ChurnResult with labels and metadata.
        """
        # Filter to events before cutoff
        events = df[df["event_time"] < cutoff_date].copy()

        if events.empty:
            customers = df[customer_id_col].unique()
            labels = pd.DataFrame({
                "customer_id": customers,
                "churn": np.ones(len(customers), dtype=int),
            })
            return ChurnResult(
                labels=labels,
                strategy_name=self.name,
                metadata={"cutoff_date": str(cutoff_date), "n_churned": int(labels["churn"].sum())},
            )

        # Compute inter-purchase intervals per customer
        events = events.sort_values([customer_id_col, "event_time"])
        events["prev_date"] = events.groupby(customer_id_col)["event_time"].shift(1)
        events["interval_days"] = (
            events["event_time"] - events["prev_date"]
        ).dt.total_seconds() / 86400

        # Get median frequency across all customers
        median_interval = events["interval_days"].median()

        # Get each customer's most recent interval
        last_intervals = (
            events.groupby(customer_id_col)["interval_days"]
            .last()
            .reset_index()
        )

        # Get all customers
        all_customers = df[customer_id_col].unique()
        result_df = pd.DataFrame({customer_id_col: all_customers})

        # Merge with last intervals
        result_df = result_df.merge(last_intervals, on=customer_id_col, how="left")

        # Churn if last interval > 2x median or no previous purchase
        threshold = 2 * median_interval if pd.notna(median_interval) else 365
        result_df["churn"] = (
            (result_df["interval_days"].isna()) |
            (result_df["interval_days"] > threshold)
        ).astype(int)

        return ChurnResult(
            labels=result_df[[customer_id_col, "churn"]],
            strategy_name=self.name,
            metadata={
                "cutoff_date": str(cutoff_date),
                "median_interval_days": float(median_interval) if pd.notna(median_interval) else None,
                "threshold_days": float(threshold),
                "n_churned": int(result_df["churn"].sum()),
            },
        )


def main():
    print("Custom Churn Strategy Example")
    print("=" * 60)

    # Create the strategy
    strategy = FrequencyBasedStrategy()

    print(f"Strategy name: {strategy.name}")
    print(f"Description: {strategy.description}")
    print(f"Required columns: {strategy.required_columns}")

    # Demonstrate with sample data
    print("\n--- Sample Usage ---")

    # Create sample transaction data
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    customers = [f"cust_{i}" for i in range(20)]
    np.random.seed(42)

    data = []
    for cust in customers:
        n_purchases = np.random.randint(2, 9)
        purchase_dates = np.random.choice(dates, n_purchases, replace=False)
        for d in sorted(purchase_dates):
            data.append({"customer_id": cust, "event_time": d})

    df = pd.DataFrame(data)

    # Create labels using the custom strategy
    cutoff = pd.Timestamp("2023-04-01")
    result = strategy.label(df, cutoff_date=cutoff)

    print(f"\nCustomers: {len(result.labels)}")
    print(f"Churned:   {result.labels['churn'].sum()} ({result.labels['churn'].mean():.1%})")
    print(f"Retained:  {(1 - result.labels['churn']).sum()} ({1 - result.labels['churn'].mean():.1%})")
    print(f"Metadata:  {result.metadata}")

    # Register with the framework
    print("\n--- Registering with Framework ---")
    register_churn_strategy_class("frequency_based", FrequencyBasedStrategy)
    print("Registered 'frequency_based' with churn registry")

    print("\nTo use this strategy in the pipeline:")
    print("1. Add it to src/churn/ directory")
    print("2. Register it in src/churn/__init__.py")
    print("3. Set 'churn.strategy: frequency_based' in your config")


if __name__ == "__main__":
    main()
