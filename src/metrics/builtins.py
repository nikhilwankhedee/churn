"""
Built-in metric wrappers for the standard classification metrics.

Each wrapper delegates to the corresponding scikit-learn function,
preserving 100% backward compatibility with the published evaluation code.
"""
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
)

from src.metrics.base import EvaluationMetric, MetricResult


class AccuracyMetric(EvaluationMetric):
    """Fraction of correctly classified samples."""

    @property
    def name(self) -> str:
        return "accuracy"

    @property
    def higher_is_better(self) -> bool:
        return True

    def evaluate(self, y_true, y_pred=None, y_proba=None, **kwargs):
        if y_pred is None:
            raise ValueError("AccuracyMetric requires y_pred")
        return MetricResult(
            name=self.name,
            value=float(accuracy_score(y_true, y_pred)),
        )


class PrecisionMetric(EvaluationMetric):
    """Precision: TP / (TP + FP)."""

    @property
    def name(self) -> str:
        return "precision"

    @property
    def higher_is_better(self) -> bool:
        return True

    def evaluate(self, y_true, y_pred=None, y_proba=None, **kwargs):
        if y_pred is None:
            raise ValueError("PrecisionMetric requires y_pred")
        return MetricResult(
            name=self.name,
            value=float(precision_score(y_true, y_pred, zero_division=0.0)),
        )


class RecallMetric(EvaluationMetric):
    """Recall (sensitivity): TP / (TP + FN)."""

    @property
    def name(self) -> str:
        return "recall"

    @property
    def higher_is_better(self) -> bool:
        return True

    def evaluate(self, y_true, y_pred=None, y_proba=None, **kwargs):
        if y_pred is None:
            raise ValueError("RecallMetric requires y_pred")
        return MetricResult(
            name=self.name,
            value=float(recall_score(y_true, y_pred, zero_division=0.0)),
        )


class F1Metric(EvaluationMetric):
    """Harmonic mean of precision and recall."""

    @property
    def name(self) -> str:
        return "f1"

    @property
    def higher_is_better(self) -> bool:
        return True

    def evaluate(self, y_true, y_pred=None, y_proba=None, **kwargs):
        if y_pred is None:
            raise ValueError("F1Metric requires y_pred")
        return MetricResult(
            name=self.name,
            value=float(f1_score(y_true, y_pred, zero_division=0.0)),
        )


class RocAucMetric(EvaluationMetric):
    """Area under the ROC curve."""

    @property
    def name(self) -> str:
        return "roc_auc"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def requires_proba(self) -> bool:
        return True

    @property
    def requires_pred(self) -> bool:
        return False

    def evaluate(self, y_true, y_pred=None, y_proba=None, **kwargs):
        if y_proba is None:
            raise ValueError("RocAucMetric requires y_proba")
        try:
            value = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            value = float("nan")
        return MetricResult(name=self.name, value=value)


class AveragePrecisionMetric(EvaluationMetric):
    """Area under the precision-recall curve (PR-AUC)."""

    @property
    def name(self) -> str:
        return "avg_precision"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def requires_proba(self) -> bool:
        return True

    @property
    def requires_pred(self) -> bool:
        return False

    def evaluate(self, y_true, y_pred=None, y_proba=None, **kwargs):
        if y_proba is None:
            raise ValueError("AveragePrecisionMetric requires y_proba")
        try:
            value = float(average_precision_score(y_true, y_proba))
        except ValueError:
            value = float("nan")
        return MetricResult(name=self.name, value=value)


class BrierScoreMetric(EvaluationMetric):
    """Brier score: mean squared difference between predicted proba and true label."""

    @property
    def name(self) -> str:
        return "brier_score"

    @property
    def higher_is_better(self) -> bool:
        return False  # Lower is better

    @property
    def requires_proba(self) -> bool:
        return True

    @property
    def requires_pred(self) -> bool:
        return False

    def evaluate(self, y_true, y_pred=None, y_proba=None, **kwargs):
        if y_proba is None:
            raise ValueError("BrierScoreMetric requires y_proba")
        try:
            value = float(brier_score_loss(y_true, y_proba))
        except ValueError:
            value = float("nan")
        return MetricResult(name=self.name, value=value)


class CalibrationErrorMetric(EvaluationMetric):
    """Expected Calibration Error (ECE)."""

    @property
    def name(self) -> str:
        return "calibration_error"

    @property
    def higher_is_better(self) -> bool:
        return False  # Lower is better

    @property
    def requires_proba(self) -> bool:
        return True

    @property
    def requires_pred(self) -> bool:
        return False

    def evaluate(self, y_true, y_pred=None, y_proba=None, **kwargs):
        if y_proba is None:
            raise ValueError("CalibrationErrorMetric requires y_proba")
        n_bins = kwargs.get("n_bins", 10)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins) - 1
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = y_proba[mask].mean()
            ece += np.abs(bin_acc - bin_conf) * mask.sum()
        value = ece / len(y_true) if len(y_true) > 0 else 0.0
        return MetricResult(
            name=self.name,
            value=float(value),
            metadata={"n_bins": n_bins},
        )
