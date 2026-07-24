"""
Subscription (contractual) churn strategy.

For datasets that provide native churn labels (e.g., Telco), this strategy
bypasses inactivity-based labeling and uses the dataset-provided label directly.

The adapter's `create_churn_labels(df, cutoff_date)` method is called
to produce labels, preserving the existing dataset-specific logic.
"""
from typing import Any, List

import pandas as pd

from src.churn.base import ChurnStrategy, ChurnResult


class SubscriptionStrategy(ChurnStrategy):
    """Use dataset-provided native churn labels (contractual churn)."""

    @property
    def name(self) -> str:
        return "subscription"

    @property
    def required_columns(self) -> List[str]:
        return ["customer_id"]

    @property
    def description(self) -> str:
        return (
            "Contractual churn: uses the native churn label provided by the "
            "dataset.适用于 subscription-based datasets (e.g., Telco)."
        )

    def label(
        self,
        df: pd.DataFrame,
        cutoff_date: pd.Timestamp,
        customer_id_col: str = "customer_id",
        **kwargs: Any,
    ) -> ChurnResult:
        # The dataset adapter is responsible for producing native labels.
        # This strategy delegates to the adapter's create_churn_labels method.
        adapter = kwargs.get("adapter")
        if adapter is None:
            raise ValueError(
                "SubscriptionStrategy requires an 'adapter' keyword argument "
                "with a BaseDatasetAdapter instance that provides "
                "create_churn_labels()."
            )

        if not hasattr(adapter, "create_churn_labels"):
            raise ValueError(
                f"Adapter '{adapter.dataset_name}' does not implement "
                "create_churn_labels(). Use InactivityStrategy instead."
            )

        labels = adapter.create_churn_labels(df, cutoff_date)

        return ChurnResult(
            labels=labels,
            strategy_name=self.name,
            metadata={
                "cutoff_date": cutoff_date.isoformat(),
                "adapter": adapter.dataset_name,
                "n_customers": len(labels),
                "churn_rate": float(labels["churn"].mean()),
            },
        )
