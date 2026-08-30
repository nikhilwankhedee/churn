"""
Temporal churn labeling with strict leakage prevention and dataset-aware
churn definition support.

Behavioural churn (default): a customer is labelled *churned* if they placed
NO event in the PREDICTION_WINDOW_DAYS following the cutoff date.

Contractual churn: datasets like Telco provide native churn labels —
inactivity-based labeling is bypassed.

Labels are always computed from data strictly after the cutoff —
never before it — ensuring no temporal leakage into features.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from src.config import PREDICTION_WINDOW_DAYS, TRAIN_SPLIT_QUANTILE
from src.utils import get_logger

logger = get_logger(__name__)


def create_churn_labels(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    prediction_window_days: int = PREDICTION_WINDOW_DAYS,
    customer_id_col: str = 'customer_id',
    event_time_col: str = 'event_time',
) -> pd.DataFrame:
    """Create inactivity-based churn labels for a given cutoff.

    A customer is labelled churned (1) if they have NO event in the
    prediction window following the cutoff.
    """
    if customer_id_col not in df.columns or event_time_col not in df.columns:
        raise ValueError(
            f"Required columns '{customer_id_col}' and '{event_time_col}' "
            f"not found in DataFrame"
        )

    active_before = df[df[event_time_col] < cutoff_date]
    customer_ids = active_before[customer_id_col].dropna().unique()

    if len(customer_ids) == 0:
        raise ValueError(
            f"No customers with events before cutoff {cutoff_date.date()}"
        )

    window_end = cutoff_date + pd.Timedelta(days=prediction_window_days)
    future_events = df[
        (df[event_time_col] > cutoff_date)
        & (df[event_time_col] <= window_end)
    ]
    future_customers = set(future_events[customer_id_col].dropna().unique())

    labels = pd.DataFrame({customer_id_col: customer_ids})
    labels['churn'] = labels[customer_id_col].apply(
        lambda cid: 0 if cid in future_customers else 1
    )

    churn_rate = labels['churn'].mean()
    logger.info(
        "Churn labels at %s — window: %d days, rate: %.2f%% (%d / %d)",
        cutoff_date.date(), prediction_window_days,
        churn_rate * 100, int(labels['churn'].sum()), len(labels),
    )

    if churn_rate < 0.01 or churn_rate > 0.99:
        logger.warning(
            "Extreme churn rate (%.1f%%) — verify window / data period",
            churn_rate * 100,
        )

    return labels


def compute_imbalance_ratio(y: pd.Series) -> float:
    """Return imbalance ratio (neg/pos) for a binary label series."""
    pos = int(y.sum())
    neg = int((1 - y).sum())
    if pos == 0:
        return float('inf')
    return neg / pos


def get_train_test_cutoffs(
    df: pd.DataFrame,
    train_quantile: float = TRAIN_SPLIT_QUANTILE,
    prediction_window_days: int = PREDICTION_WINDOW_DAYS,
    event_time_col: str = 'event_time',
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Determine train/test temporal cutoffs.

    The test cutoff is placed PREDICTION_WINDOW_DAYS before the max date
    (to leave room for the label window).  The train cutoff is at the
    specified quantile of the event timeline.
    """
    max_date = df[event_time_col].max()
    test_cutoff = max_date - pd.Timedelta(days=prediction_window_days)
    train_cutoff = df[event_time_col].quantile(train_quantile)

    if train_cutoff >= test_cutoff:
        train_cutoff = test_cutoff - pd.Timedelta(days=prediction_window_days)
        logger.warning(
            "Train cutoff >= test cutoff; adjusted train cutoff to %s",
            train_cutoff.date(),
        )

    logger.info(
        "Cutoffs — train: %s, test: %s",
        train_cutoff.date(), test_cutoff.date(),
    )
    return train_cutoff, test_cutoff
