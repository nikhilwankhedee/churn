"""
Model evaluation — full classification metrics, threshold analysis,
calibration data, expected calibration error (ECE), and imbalance analysis.

Handles extreme imbalance gracefully (zero-division avoidance, NaN-safe scoring).

Metrics produced per model (Section 29 of the experiment spec):
    accuracy, precision, recall, f1, roc_auc, avg_precision (PR-AUC),
    balanced_accuracy, mcc, brier_score, calibration_error,
    training_time (sec), inference_time (sec).
"""
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, brier_score_loss,
    precision_recall_curve, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef,
    roc_curve,
)
from sklearn.calibration import calibration_curve

from src.utils import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = 'model',
) -> Tuple[Dict[str, float], np.ndarray, Optional[np.ndarray]]:
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - t0

    has_proba = hasattr(model, 'predict_proba')
    y_proba: Optional[np.ndarray] = None
    if has_proba:
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

    n_pos = int(y_test.sum())
    n_neg = int((1 - y_test).sum())

    metrics: Dict[str, float] = {'model': model_name, 'n_test': len(y_test),
                                  'n_pos': n_pos, 'n_neg': n_neg}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred, zero_division=0.0)
    metrics['recall'] = recall_score(y_test, y_pred, zero_division=0.0)
    metrics['f1'] = f1_score(y_test, y_pred, zero_division=0.0)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_test, y_pred)
    metrics['training_time'] = float(
        getattr(model, '_train_time_sec', np.nan)
    )
    metrics['inference_time'] = float(inference_time)

    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        except Exception:
            metrics['roc_auc'] = np.nan
        try:
            metrics['avg_precision'] = average_precision_score(y_test, y_proba)
        except Exception:
            metrics['avg_precision'] = np.nan
        try:
            metrics['brier_score'] = brier_score_loss(y_test, y_proba)
        except Exception:
            metrics['brier_score'] = np.nan
        try:
            metrics['calibration_error'] = _expected_calibration_error(
                y_test, y_proba
            )
        except Exception:
            metrics['calibration_error'] = np.nan

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    metrics['tp'] = int(tp)

    return metrics, cm, y_proba


def _expected_calibration_error(
    y_true: pd.Series, y_proba: np.ndarray, n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
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
    return ece / len(y_true)


def compute_imbalance_metrics(y: pd.Series) -> Dict[str, float]:
    """Return churn rate and imbalance ratio for a binary label series."""
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    churn_rate = y.mean()
    imbalance_ratio = n_neg / n_pos if n_pos > 0 else float('inf')
    return {
        'churn_rate': churn_rate,
        'imbalance_ratio': imbalance_ratio,
        'n_pos': n_pos,
        'n_neg': n_neg,
        'n_total': len(y),
    }


def threshold_analysis(
    y_true: pd.Series, y_proba: np.ndarray,
    n_thresholds: int = 50,
) -> pd.DataFrame:
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append({
            'threshold': t,
            'precision': precision_score(y_true, y_pred, zero_division=0.0),
            'recall': recall_score(y_true, y_pred, zero_division=0.0),
            'f1': f1_score(y_true, y_pred, zero_division=0.0),
        })
    return pd.DataFrame(rows)


def get_roc_data(
    y_true: pd.Series, y_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    return fpr, tpr, auc


def get_pr_data(
    y_true: pd.Series, y_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    return prec, rec, ap


def get_calibration_data(
    y_true: pd.Series, y_proba: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        prob_true, prob_pred = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy='uniform',
        )
        return prob_true, prob_pred
    except Exception:
        return np.array([]), np.array([])
