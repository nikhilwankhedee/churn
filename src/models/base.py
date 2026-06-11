"""
Abstract base class for model wrappers.

All model wrappers implement a common interface for training and prediction,
enabling the framework to treat all models uniformly while allowing
dataset-specific or experiment-specific customization.

The wrapper delegates to an underlying scikit-learn-compatible estimator,
exposing fit/predict/predict_proba with consistent signatures.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ModelResult:
    """Container for model training/prediction outputs.

    Attributes
    ----------
    model_name : str
        Identifier for the model type (e.g. 'logistic_regression').
    fitted_model : object
        The trained scikit-learn-compatible estimator.
    train_metrics : dict
        Metrics computed on training data (optional).
    metadata : dict
        Model-specific metadata (hyperparameters, class weights, etc.).
    """
    model_name: str
    fitted_model: Any
    train_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelWrapper(ABC):
    """Abstract base for model wrappers.

    Subclasses must implement:
        - name: model identifier
        - default_hyperparameters: dict of default params
        - _create_estimator(params) -> estimator instance
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this model (e.g. 'logistic_regression')."""
        ...

    @property
    @abstractmethod
    def default_hyperparameters(self) -> Dict[str, Any]:
        """Default hyperparameters for this model type."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of the model."""
        return self.__class__.__doc__ or ""

    @property
    def supported_params(self) -> List[str]:
        """List of hyperparameter names this model accepts."""
        return list(self.default_hyperparameters.keys())

    @abstractmethod
    def _create_estimator(self, params: Dict[str, Any]) -> Any:
        """Create an unfitted estimator instance with the given params.

        Returns a scikit-learn-compatible estimator with fit/predict
        and (optionally) predict_proba methods.
        """
        ...

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> ModelResult:
        """Train the model and return a ModelResult.

        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val : optional validation data (for early stopping)
        **kwargs : override default hyperparameters

        Returns
        -------
        ModelResult with fitted model and metadata.
        """
        params = {**self.default_hyperparameters, **kwargs}
        estimator = self._create_estimator(params)

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            if hasattr(estimator, "fit"):
                # Check if model supports eval_set (XGBoost-style)
                try:
                    estimator.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False,
                    )
                except TypeError:
                    # Fall back to standard fit
                    estimator.fit(X_train, y_train)
            else:
                estimator.fit(X_train, y_train)
        else:
            estimator.fit(X_train, y_train)

        return ModelResult(
            model_name=self.name,
            fitted_model=estimator,
            metadata={
                "hyperparameters": params,
                "n_train": len(X_train),
                "n_val": len(X_val) if X_val is not None else 0,
            },
        )

    def predict(self, model_result: ModelResult, X: pd.DataFrame) -> np.ndarray:
        """Return class predictions."""
        return model_result.fitted_model.predict(X)

    def predict_proba(
        self, model_result: ModelResult, X: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Return probability estimates for the positive class, or None."""
        if hasattr(model_result.fitted_model, "predict_proba"):
            try:
                return model_result.fitted_model.predict_proba(X)[:, 1]
            except Exception:
                return None
        return None
