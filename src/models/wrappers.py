"""
Built-in model wrappers for the three default classifiers.

Each wrapper delegates to the existing training logic in `src.modeling`,
preserving 100% backward compatibility while enabling the registry-based
extensibility of the framework.
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.models.base import ModelWrapper, ModelResult
from src.config import (
    RANDOM_SEED,
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
)


class LogisticRegressionWrapper(ModelWrapper):
    """Logistic Regression with LBFGS solver and balanced class weights."""

    @property
    def name(self) -> str:
        return "logistic_regression"

    @property
    def default_hyperparameters(self) -> Dict[str, Any]:
        return dict(LOGISTIC_REGRESSION_PARAMS)

    def _create_estimator(self, params: Dict[str, Any]) -> LogisticRegression:
        return LogisticRegression(**params)


class RandomForestWrapper(ModelWrapper):
    """Random Forest with balanced subsample class weighting."""

    @property
    def name(self) -> str:
        return "random_forest"

    @property
    def default_hyperparameters(self) -> Dict[str, Any]:
        return dict(RANDOM_FOREST_PARAMS)

    def _create_estimator(self, params: Dict[str, Any]) -> RandomForestClassifier:
        return RandomForestClassifier(**params)


class XGBoostWrapper(ModelWrapper):
    """XGBoost with scale_pos_weight and early stopping support."""

    @property
    def name(self) -> str:
        return "xgboost"

    @property
    def default_hyperparameters(self) -> Dict[str, Any]:
        params = dict(XGBOOST_PARAMS)
        # Remove None-valued keys that XGBoost doesn't accept
        params.pop("scale_pos_weight", None)
        params.pop("use_label_encoder", None)
        return params

    def _create_estimator(self, params: Dict[str, Any]) -> XGBClassifier:
        return XGBClassifier(**params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> ModelResult:
        # Compute scale_pos_weight from training data
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos = n_neg / n_pos if n_pos > 0 and n_neg > 0 else 1.0

        params = {**self.default_hyperparameters, **kwargs}
        params["scale_pos_weight"] = scale_pos

        estimator = self._create_estimator(params)

        if X_val is not None and y_val is not None:
            estimator.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False,
            )
        else:
            estimator.fit(X_train, y_train)

        return ModelResult(
            model_name=self.name,
            fitted_model=estimator,
            metadata={
                "hyperparameters": params,
                "scale_pos_weight": scale_pos,
                "n_train": len(X_train),
                "n_val": len(X_val) if X_val is not None else 0,
            },
        )
