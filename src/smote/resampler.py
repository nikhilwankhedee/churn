"""
SMOTE resampling with temporal safety assertions.

This module provides controlled SMOTE application that:
1. Only resamples training data (never test data)
2. Validates no test data leakage via temporal assertions
3. Logs resampling statistics for reproducibility
4. Is gated by configuration (disabled by default)

Critical safety rule: SMOTE must be applied AFTER the temporal train/test
split and ONLY on training data. Applying SMOTE before splitting or on
test data would cause temporal data leakage.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class SmoteResult:
    """Container for SMOTE resampling results.

    Attributes
    ----------
    X_resampled : pd.DataFrame
        Resampled feature matrix.
    y_resampled : pd.Series
        Resampled labels.
    n_original : int
        Number of samples before resampling.
    n_resampled : int
        Number of samples after resampling.
    n_synthetic : int
        Number of synthetic samples generated.
    class_distribution_before : dict
        Class counts before resampling.
    class_distribution_after : dict
        Class counts after resampling.
    """
    X_resampled: pd.DataFrame
    y_resampled: pd.Series
    n_original: int
    n_resampled: int
    n_synthetic: int
    class_distribution_before: dict
    class_distribution_after: dict


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    random_state: int = 42,
    k_neighbors: int = 5,
    sampling_strategy: str = "auto",
) -> SmoteResult:
    """Apply SMOTE to training data with temporal safety checks.

    Parameters
    ----------
    X_train, y_train : training data to resample
    X_test, y_test : test data (used for validation only — never resampled)
    random_state : random seed for reproducibility
    k_neighbors : number of nearest neighbors for SMOTE
    sampling_strategy : 'auto' balances classes, or float for target ratio

    Returns
    -------
    SmoteResult with resampled data and statistics.

    Raises
    ------
    ImportError if imbalanced-learn is not installed.
    ValueError if test data is provided and detected in training set.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError(
            "imbalanced-learn is required for SMOTE. "
            "Install with: pip install imbalanced-learn"
        )

    # ── Temporal safety assertion ──────────────────────────────────
    _validate_no_test_leakage(X_train, y_train, X_test, y_test)

    # ── Record pre-resampling state ────────────────────────────────
    n_original = len(X_train)
    dist_before = {
        "n_neg": int((y_train == 0).sum()),
        "n_pos": int((y_train == 1).sum()),
    }

    # ── Single-class guard ─────────────────────────────────────────
    # SMOTE interpolates between minority-class neighbours, which requires
    # at least 2 minority samples AND both classes present.  With fewer
    # than 2 samples in either class there is nothing to synthesise from,
    # so resampling is skipped and the data is returned unchanged —
    # k_neighbors must never be clamped to <= 0 (that previously produced
    # `SMOTE k_neighbors = -1`).
    n_pos = dist_before["n_pos"]
    n_neg = dist_before["n_neg"]
    if n_pos < 2 or n_neg < 2:
        logger.warning(
            "SMOTE skipped — fewer than 2 samples in a class "
            "(neg: %d, pos: %d); returning data unchanged",
            n_neg, n_pos,
        )
        return SmoteResult(
            X_resampled=X_train.copy(),
            y_resampled=y_train.copy(),
            n_original=n_original,
            n_resampled=n_original,
            n_synthetic=0,
            class_distribution_before=dist_before,
            class_distribution_after=dict(dist_before),
        )

    # ── Apply SMOTE ────────────────────────────────────────────────
    k = min(k_neighbors, min(n_pos, n_neg) - 1)
    smote = SMOTE(
        random_state=random_state,
        k_neighbors=k,
        sampling_strategy=sampling_strategy,
    )

    X_res, y_res = smote.fit_resample(X_train, y_train)

    # ── Record post-resampling state ───────────────────────────────
    n_resampled = len(X_res)
    n_synthetic = n_resampled - n_original
    dist_after = {
        "n_neg": int((y_res == 0).sum()),
        "n_pos": int((y_res == 1).sum()),
    }

    logger.info(
        "SMOTE applied — original: %d, resampled: %d, synthetic: %d, "
        "class dist before: {neg: %d, pos: %d}, after: {neg: %d, pos: %d}",
        n_original, n_resampled, n_synthetic,
        dist_before["n_neg"], dist_before["n_pos"],
        dist_after["n_neg"], dist_after["n_pos"],
    )

    return SmoteResult(
        X_resampled=X_res,
        y_resampled=y_res,
        n_original=n_original,
        n_resampled=n_resampled,
        n_synthetic=n_synthetic,
        class_distribution_before=dist_before,
        class_distribution_after=dist_after,
    )


def _validate_no_test_leakage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame],
    y_test: Optional[pd.Series],
) -> None:
    """Validate that test data is not present in training data.

    This is a safety check to prevent temporal leakage when SMOTE
    is applied. Checks that no test index appears in training index.
    """
    if X_test is None or y_test is None:
        return

    # Check for index overlap
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    overlap = train_indices & test_indices

    if overlap:
        raise ValueError(
            f"Temporal leakage detected: {len(overlap)} indices appear in "
            f"both training and test sets. SMOTE cannot be applied safely. "
            f"Ensure the temporal train/test split is performed BEFORE "
            f"SMOTE resampling."
        )

    # Check for row-level duplication (same feature values in both sets)
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
    X_train_array = X_train.values if hasattr(X_train, 'values') else X_train

    # Quick check: compare shapes first
    if X_test_array.shape[1] == X_train_array.shape[1]:
        # Check if any test row is an exact duplicate of a training row
        # Use a sample for efficiency
        n_check = min(100, len(X_test))
        test_sample_idx = np.random.RandomState(42).choice(
            len(X_test), n_check, replace=False
        )
        for idx in test_sample_idx:
            test_row = X_test_array[idx]
            matches = np.all(X_train_array == test_row, axis=1)
            if matches.any():
                logger.warning(
                    "Test row %d has exact match in training data — "
                    "verify temporal split is correct", idx,
                )
                break
