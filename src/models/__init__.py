"""
Model wrappers and registry.

This package provides a pluggable model system with:
- Abstract base class (ModelWrapper)
- Built-in wrappers (LogisticRegression, RandomForest, XGBoost)
- Plugin registry for custom models

Usage:
    from src.models import get_model, list_models

    wrapper = get_model("xgboost")
    result = wrapper.fit(X_train, y_train, X_val, y_val)
    preds = wrapper.predict(result, X_test)
"""
from src.models.base import ModelWrapper, ModelResult
from src.models.registry import (
    get_model,
    list_models,
    register_model,
    register_model_class,
)

__all__ = [
    "ModelWrapper",
    "ModelResult",
    "get_model",
    "list_models",
    "register_model",
    "register_model_class",
]
