"""
Cadence degradation churn strategy.

Labels a customer as churned if their inter-purchase gap in the observation
period exceeds a multiple of their historical median cadence.

This captures "the customer has slowed down" rather than "the customer
has stopped entirely" — useful for early churn detection.

The strategy computes each customer's median inter-purchase interval from
their historical events, then checks whether their most recent gap before
the cutoff exceeds a configurable threshold multiplier.
"""
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.churn.base import ChurnStrategy, ChurnResult


class CadenceStrategy(ChurnStrategy):
    """Label churn based on degraded purchase cadence."""

    @property
    def name(self) -> str:
        return "cadence"

    @property
    def required_columns(self) -> List[str]:
        return ["customer_id", "event_time"]

    @property
    def description(self) -> str:
        return (
            "A customer is labeled churned if their most recent inter-purchase "
            "gap exceeds a multiple of their historical median cadence. "
            "Enables early churn detection before full inactivity."
        )

    def label(
        self,
        df: pd.DataFrame,
        cutoff_date: pd.Timestamp,
        customer_id_col: str = "customer_id",
        event_time_col: str = "event_time",
        threshold_multiplier: float = 2.0,
        min_events: int = 3,
        **kwargs: Any,
    ) -> ChurnResult:
        self.validate_columns(df)

        events_before = df[df[event_time_col] < cutoff_date].copy()
        events_before = events_before.sort_values([customer_id_col, event_time_col])

        # Compute per-customer statistics
        results = []
        for cid, group in events_before.groupby(customer_id_col):
            times = group[event_time_col].dropna().sort_values()

            if len(times) < min_events:
                # Not enough history — classify as churned (conservative)
                results.append({customer_id_col: cid, "churn": 1})
                continue

            # Median inter-purchase gap
            gaps = times.diff().dt.total_seconds().dropna()
            if len(gaps) == 0:
                results.append({customer_id_col: cid, "churn": 1})
                continue

            median_gap = gaps.median()

            # Time from last event before cutoff to cutoff itself
            last_event = times.iloc[-1]
            gap_to_cutoff = (cutoff_date - last_event).total_seconds()

            # Churned if gap to cutoff exceeds threshold
            churned = int(gap_to_cutoff > threshold_multiplier * median_gap)
            results.append({customer_id_col: cid, "churn": churned})

        labels = pd.DataFrame(results)

        if labels.empty:
            raise ValueError(
                f"No customers found with events before cutoff {cutoff_date.date()}"
            )

        churn_rate = labels["churn"].mean()

        return ChurnResult(
            labels=labels,
            strategy_name=self.name,
            metadata={
                "cutoff_date": cutoff_date.isoformat(),
                "threshold_multiplier": threshold_multiplier,
                "min_events": min_events,
                "n_customers": len(labels),
                "churn_rate": float(churn_rate),
            },
        )
