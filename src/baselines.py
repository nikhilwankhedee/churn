"""
Baseline classifiers for behavioural churn research.

All metrics must be compared against these to establish whether
sophisticated models provide meaningful lift over trivial strategies.

1. Majority-class baseline: always predict the majority class.
   This sets the floor for accuracy and reflects class imbalance.

2. Random baseline: predict churn with probability equal to the
   empirical churn rate in the training set.  This provides a
   reference ROC-AUC of ~0.5 and helps contextualise model performance.
"""
import numpy as np
import pandas as pd
from typing import Tuple

from src.utils import get_logger

logger = get_logger(__name__)


def majority_class_baseline(
    y_train: pd.Series,
    y_test: pd.Series,
) -> np.ndarray:
    """Predict the majority class from training set for all test rows.

    Returns
    -------
    np.ndarray of shape (n_test,) with predictions (0 for retained,
    1 for churned).
    """
    majority_class = y_train.mode().iloc[0]
    predictions = np.full(len(y_test), majority_class, dtype=int)
    logger.info(
        "Majority-class baseline — train majority: %d (%.1f%%), test preds: %d",
        majority_class,
        (y_train == majority_class).mean() * 100,
        majority_class,
    )
    return predictions


def random_baseline(
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random classifier calibrated to the empirical churn rate.

    Returns
    -------
    (predictions, probabilities) where predictions are binary and
    probabilities equal the training churn rate for every row.
    """
    churn_rate = y_train.mean()
    rng = np.random.RandomState(random_state)
    predictions = (rng.rand(len(y_test)) < churn_rate).astype(int)
    probabilities = np.full(len(y_test), churn_rate)
    logger.info(
        "Random baseline — train churn rate: %.4f", churn_rate,
    )
    return predictions, probabilities
