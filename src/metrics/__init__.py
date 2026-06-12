"""
Evaluation metrics and registry.

This package provides a pluggable metric system with:
- Abstract base class (EvaluationMetric)
- Built-in metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier, ECE)
- Plugin registry for custom metrics

Usage:
    from src.metrics import get_metric, list_metrics, evaluate_with_all

    metric = get_metric("roc_auc")
    result = metric.evaluate(y_true, y_proba=y_proba)

    # Or evaluate all metrics at once
    results = evaluate_with_all(y_true, y_pred, y_proba)
"""
from src.metrics.base import EvaluationMetric, MetricResult
from src.metrics.registry import (
    get_metric,
    list_metrics,
    register_metric,
    register_metric_class,
    evaluate_with_all,
)

__all__ = [
    "EvaluationMetric",
    "MetricResult",
    "get_metric",
    "list_metrics",
    "register_metric",
    "register_metric_class",
    "evaluate_with_all",
]
