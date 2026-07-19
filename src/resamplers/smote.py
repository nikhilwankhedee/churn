"""
SMOTE resampler — first registered implementation.

Wraps the existing src.smote.resampler.apply_smote() function
to maintain 100% backward compatibility while fitting into the
generic resampler registry.
"""
from typing import Any, Dict, Optional

import pandas as pd

from src.resamplers.base import Resampler, ResampleResult


class SmoteResampler(Resampler):
    """SMOTE (Synthetic Minority Over-sampling Technique) resampler."""

    @property
    def name(self) -> str:
        return "smote"

    @property
    def description(self) -> str:
        return (
            "Generates synthetic minority samples using k-nearest neighbors. "
            "Requires imbalanced-learn."
        )

    @property
    def requires_imbalanced_learn(self) -> bool:
        return True

    def resample(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int = 42,
        k_neighbors: int = 5,
        sampling_strategy: str = "auto",
        **kwargs: Any,
    ) -> ResampleResult:
        from src.smote import apply_smote

        result = apply_smote(
            X_train=X_train,
            y_train=y_train,
            random_state=random_state,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
        )

        return ResampleResult(
            X_resampled=result.X_resampled,
            y_resampled=result.y_resampled,
            n_original=result.n_original,
            n_resampled=result.n_resampled,
            n_synthetic=result.n_synthetic,
            class_distribution_before=result.class_distribution_before,
            class_distribution_after=result.class_distribution_after,
            resampler_name=self.name,
            metadata={
                "random_state": random_state,
                "k_neighbors": k_neighbors,
                "sampling_strategy": sampling_strategy,
            },
        )
