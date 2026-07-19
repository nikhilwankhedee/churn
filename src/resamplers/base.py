"""
Abstract base class for resampling strategies.

All resamplers implement a common interface for applying resampling
to training data. This replaces the SMOTE-specific implementation
with a generic pluggable system.

Critical safety rule: Resamplers must only be applied AFTER the temporal
train/test split and ONLY on training data. Applying resampling before
splitting or on test data would cause temporal data leakage.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ResampleResult:
    """Container for resampling results.

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
        Number of synthetic samples generated (0 for over-sampling of existing).
    class_distribution_before : dict
        Class counts before resampling.
    class_distribution_after : dict
        Class counts after resampling.
    resampler_name : str
        Name of the resampler that produced this result.
    metadata : dict
        Resampler-specific metadata.
    """
    X_resampled: pd.DataFrame
    y_resampled: pd.Series
    n_original: int
    n_resampled: int
    n_synthetic: int
    class_distribution_before: dict
    class_distribution_after: dict
    resampler_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class Resampler(ABC):
    """Abstract base for resampling strategies.

    Subclasses must implement:
        - name: resampler identifier
        - resample(X_train, y_train, **kwargs) -> ResampleResult
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this resampler (e.g. 'smote', 'adasyn')."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of the resampling strategy."""
        return self.__class__.__doc__ or ""

    @property
    def requires_imbalanced_learn(self) -> bool:
        """Whether this resampler requires imbalanced-learn."""
        return False

    @abstractmethod
    def resample(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs: Any,
    ) -> ResampleResult:
        """Apply resampling to training data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels.
        **kwargs : Any
            Resampler-specific parameters.

        Returns
        -------
        ResampleResult with resampled data and statistics.
        """
        ...

    def validate_inputs(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> None:
        """Validate that test data is not present in training data.

        This is a safety check to prevent temporal leakage.
        """
        if X_test is None or y_test is None:
            return

        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        overlap = train_indices & test_indices

        if overlap:
            raise ValueError(
                f"Temporal leakage detected: {len(overlap)} indices appear in "
                f"both training and test sets. Resampling cannot be applied safely. "
                f"Ensure the temporal train/test split is performed BEFORE "
                f"resampling."
            )

    def __repr__(self) -> str:
        return f"<Resampler: {self.name}>"
