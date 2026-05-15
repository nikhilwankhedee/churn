"""
Failure case analysis: compare false-positive / false-negative customer profiles.
"""
import pandas as pd
import numpy as np
from src.utils import get_logger

logger = get_logger(__name__)


def analyze_errors(
    X_test: pd.DataFrame, y_test: pd.Series,
    y_pred: np.ndarray, y_proba: np.ndarray = None,
) -> dict:
    tp = (y_test == 1) & (y_pred == 1)
    tn = (y_test == 0) & (y_pred == 0)
    fp = (y_test == 0) & (y_pred == 1)
    fn = (y_test == 1) & (y_pred == 0)

    results = {}
    for label, mask in [('true_positive', tp), ('true_negative', tn),
                         ('false_positive', fp), ('false_negative', fn)]:
        n = int(mask.sum())
        entry = {'count': n}
        if n > 0:
            sub = X_test[mask]
            entry['feature_means'] = sub.mean().to_dict()
            if y_proba is not None:
                entry['avg_churn_prob'] = float(np.mean(y_proba[mask]))
        results[label] = entry
    return results


def behavioral_comparison(
    fp_features: pd.DataFrame = None,
    fn_features: pd.DataFrame = None,
) -> pd.DataFrame:
    if fp_features is None or fn_features is None:
        return pd.DataFrame()
    if len(fp_features) == 0 or len(fn_features) == 0:
        return pd.DataFrame()
    fpm = fp_features.mean(numeric_only=True)
    fnm = fn_features.mean(numeric_only=True)
    comp = pd.DataFrame({
        'false_positive_mean': fpm,
        'false_negative_mean': fnm,
    })
    comp['difference'] = comp['false_positive_mean'] - comp['false_negative_mean']
    return comp.sort_values('difference', key=lambda x: x.abs(), ascending=False)
