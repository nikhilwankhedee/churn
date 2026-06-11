"""
Model registry.

Provides a centralized registry for model wrappers, building on the
core PluginRegistry. Includes convenience functions for registering
built-in models and retrieving them by name.

Usage:
    from src.models.registry import get_model, list_models

    model_wrapper = get_model("logistic_regression")
    result = model_wrapper.fit(X_train, y_train)
    predictions = model_wrapper.predict(result, X_test)
"""
from typing import List, Optional

from src.models.base import ModelWrapper, ModelResult
from src.core.registry import registry


_CATEGORY = "models"


def register_model(
    name: str,
    dotted_path: str,
    metadata: Optional[dict] = None,
) -> None:
    """Register a model wrapper by lazy dotted path."""
    registry.register(name, _CATEGORY, dotted_path, metadata)


def register_model_class(
    name: str,
    cls: type,
    metadata: Optional[dict] = None,
) -> None:
    """Register an already-imported model wrapper class."""
    registry.register_class(name, _CATEGORY, cls, metadata)


def get_model(name: str) -> ModelWrapper:
    """Retrieve a registered model wrapper instance."""
    return registry.get_instance(name, _CATEGORY)


def list_models() -> List[str]:
    """Return all registered model names."""
    return registry.list_registered(_CATEGORY)


def _register_builtins() -> None:
    """Register the three built-in model wrappers (lazy-loaded)."""
    _base = "src.models"
    builtins = {
        "logistic_regression": f"{_base}.wrappers.LogisticRegressionWrapper",
        "random_forest": f"{_base}.wrappers.RandomForestWrapper",
        "xgboost": f"{_base}.wrappers.XGBoostWrapper",
    }
    for name, path in builtins.items():
        if not registry.is_registered(name, _CATEGORY):
            registry.register(
                name, _CATEGORY, path,
                metadata={"builtin": True},
            )


# Auto-register builtins on import
_register_builtins()
