"""
Inactivity-based churn strategy.

A customer is labeled churned if they placed NO event in the prediction
window following the cutoff date. This is the canonical behavioral churn
definition used in the published research.

Wraps the existing `src.churn_labeling.create_churn_labels()` function
to preserve 100% backward compatibility.
"""
from typing import Any, Dict, List, Optional

import pandas as pd

from src.churn.base import ChurnStrategy, ChurnResult
from src.config import PREDICTION_WINDOW_DAYS


class InactivityStrategy(ChurnStrategy):
    """Label churn based on inactivity over a temporal window."""

    @property
    def name(self) -> str:
        return "inactivity"

    @property
    def required_columns(self) -> List[str]:
        return ["customer_id", "event_time"]

    @property
    def description(self) -> str:
        return (
            "Behavioral churn: a customer is labeled churned if they have "
            "no event in the prediction window following the cutoff date."
        )

    def label(
        self,
        df: pd.DataFrame,
        cutoff_date: pd.Timestamp,
        customer_id_col: str = "customer_id",
        event_time_col: str = "event_time",
        prediction_window_days: int = PREDICTION_WINDOW_DAYS,
        **kwargs: Any,
    ) -> ChurnResult:
        from src.churn_labeling import create_churn_labels

        labels = create_churn_labels(
            df=df,
            cutoff_date=cutoff_date,
            prediction_window_days=prediction_window_days,
            customer_id_col=customer_id_col,
            event_time_col=event_time_col,
        )

        return ChurnResult(
            labels=labels,
            strategy_name=self.name,
            metadata={
                "prediction_window_days": prediction_window_days,
                "cutoff_date": cutoff_date.isoformat(),
                "n_customers": len(labels),
                "churn_rate": float(labels["churn"].mean()),
            },
        )
