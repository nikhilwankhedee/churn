"""
ADASYN resampler — Adaptive Synthetic Sampling.

Similar to SMOTE but generates more samples for harder-to-learn
minority examples. Requires imbalanced-learn.
"""
from typing import Any, Optional

import pandas as pd

from src.resamplers.base import Resampler, ResampleResult


class AdasynResampler(Resampler):
    """ADASYN (Adaptive Synthetic Sampling) resampler."""

    @property
    def name(self) -> str:
        return "adasyn"

    @property
    def description(self) -> str:
        return (
            "Adaptive synthetic sampling — generates more samples for "
            "harder-to-learn minority examples. Requires imbalanced-learn."
        )

    @property
    def requires_imbalanced_learn(self) -> bool:
        return True

    def resample(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int = 42,
        n_neighbors: int = 5,
        sampling_strategy: str = "auto",
        **kwargs: Any,
    ) -> ResampleResult:
        try:
            from imblearn.over_sampling import ADASYN
        except ImportError:
            raise ImportError(
                "imbalanced-learn is required for ADASYN. "
                "Install with: pip install imbalanced-learn"
            )

        self.validate_inputs(X_train, y_train)

        n_original = len(X_train)
        dist_before = {
            "n_neg": int((y_train == 0).sum()),
            "n_pos": int((y_train == 1).sum()),
        }

        adasyn = ADASYN(
            random_state=random_state,
            n_neighbors=min(n_neighbors, dist_before["n_pos"] - 1),
            sampling_strategy=sampling_strategy,
        )

        X_res, y_res = adasyn.fit_resample(X_train, y_train)

        n_resampled = len(X_res)
        n_synthetic = n_resampled - n_original
        dist_after = {
            "n_neg": int((y_res == 0).sum()),
            "n_pos": int((y_res == 1).sum()),
        }

        return ResampleResult(
            X_resampled=X_res,
            y_resampled=y_res,
            n_original=n_original,
            n_resampled=n_resampled,
            n_synthetic=n_synthetic,
            class_distribution_before=dist_before,
            class_distribution_after=dist_after,
            resampler_name=self.name,
            metadata={
                "random_state": random_state,
                "n_neighbors": n_neighbors,
                "sampling_strategy": sampling_strategy,
            },
        )
