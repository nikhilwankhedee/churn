"""
Metric registry.

Provides a centralized registry for evaluation metrics, building on
the core PluginRegistry. Includes convenience functions for registering
built-in metrics and retrieving them by name.

Usage:
    from src.metrics.registry import get_metric, list_metrics

    metric = get_metric("roc_auc")
    result = metric.evaluate(y_true, y_proba=y_proba)
    print(result.name, result.value)
"""
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from src.metrics.base import EvaluationMetric, MetricResult
from src.core.registry import registry


_CATEGORY = "metrics"


def register_metric(
    name: str,
    dotted_path: str,
    metadata: Optional[dict] = None,
) -> None:
    """Register a metric by lazy dotted path."""
    registry.register(name, _CATEGORY, dotted_path, metadata)


def register_metric_class(
    name: str,
    cls: type,
    metadata: Optional[dict] = None,
) -> None:
    """Register an already-imported metric class."""
    registry.register_class(name, _CATEGORY, cls, metadata)


def get_metric(name: str) -> EvaluationMetric:
    """Retrieve a registered metric instance."""
    return registry.get_instance(name, _CATEGORY)


def list_metrics() -> List[str]:
    """Return all registered metric names."""
    return registry.list_registered(_CATEGORY)


def evaluate_with_all(
    y_true: pd.Series,
    y_pred: Optional[np.ndarray] = None,
    y_proba: Optional[np.ndarray] = None,
    metric_names: Optional[List[str]] = None,
) -> Dict[str, MetricResult]:
    """Evaluate all (or specified) metrics on the given predictions.

    Parameters
    ----------
    y_true : ground truth labels
    y_pred : hard predictions
    y_proba : probability estimates
    metric_names : specific metrics to evaluate (default: all registered)

    Returns
    -------
    Dict mapping metric name -> MetricResult.
    """
    names = metric_names or list_metrics()
    results = {}
    for name in names:
        try:
            metric = get_metric(name)
            result = metric.evaluate(y_true, y_pred=y_pred, y_proba=y_proba)
            results[name] = result
        except Exception:
            continue
    return results


def _register_builtins() -> None:
    """Register the eight built-in metrics (lazy-loaded)."""
    _base = "src.metrics"
    builtins = {
        "accuracy": f"{_base}.builtins.AccuracyMetric",
        "precision": f"{_base}.builtins.PrecisionMetric",
        "recall": f"{_base}.builtins.RecallMetric",
        "f1": f"{_base}.builtins.F1Metric",
        "roc_auc": f"{_base}.builtins.RocAucMetric",
        "avg_precision": f"{_base}.builtins.AveragePrecisionMetric",
        "brier_score": f"{_base}.builtins.BrierScoreMetric",
        "calibration_error": f"{_base}.builtins.CalibrationErrorMetric",
    }
    for name, path in builtins.items():
        if not registry.is_registered(name, _CATEGORY):
            registry.register(
                name, _CATEGORY, path,
                metadata={"builtin": True},
            )


# Auto-register builtins on import
_register_builtins()
