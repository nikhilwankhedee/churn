"""
Statistical comparison of churners vs non-churners.
Mann-Whitney U tests with Benjamini-Hochberg FDR correction
and Cliff's delta effect sizes.
"""
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from src.utils import get_logger

logger = get_logger(__name__)

try:
    from scipy.stats import false_discovery_control as _fdc
except ImportError:
    def _fdc(pvals):
        """Benjamini-Hochberg procedure (fallback for scipy < 1.11)."""
        p = np.asarray(pvals)
        n = len(p)
        sorted_idx = np.argsort(p)
        sorted_p = p[sorted_idx]
        thresholds = (np.arange(1, n + 1) / n) * 0.05
        max_k = 0
        for k in range(n):
            if sorted_p[k] <= thresholds[k]:
                max_k = k + 1
        reject = np.zeros(n, dtype=bool)
        reject[sorted_idx[:max_k]] = True
        return reject


def _cliffs_delta(x, y):
    """Cliff's delta in O((nx + ny) log(ny)) — vectorised, no n x n matrix.

    Equivalent to P(x > y) - P(x < y); ties contribute 0.  The naive
    pairwise sum is O(nx*ny) and hangs (or OOMs) on large cohorts.
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0
    x_arr = np.asarray(x)
    y_sorted = np.sort(np.asarray(y))
    n_less = np.searchsorted(y_sorted, x_arr, side='left')
    n_greater = ny - np.searchsorted(y_sorted, x_arr, side='right')
    return float((n_less.sum() - n_greater.sum()) / (nx * ny))


def feature_distribution_tests(
    features: pd.DataFrame, labels: pd.Series, alpha: float = 0.05,
) -> pd.DataFrame:
    results = []
    numeric_cols = features.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        g0 = features.loc[labels == 0, col].dropna()
        g1 = features.loc[labels == 1, col].dropna()
        if len(g0) < 5 or len(g1) < 5:
            continue
        try:
            _, p_val = mannwhitneyu(g0, g1, alternative='two-sided')
        except Exception:
            continue
        eff = _cliffs_delta(g1.values, g0.values)

        results.append({
            'feature': col,
            'median_retained': float(g0.median()),
            'median_churned': float(g1.median()),
            'p_value': float(p_val),
            'cliffs_delta': float(eff),
            'significant_uncorrected': bool(p_val < alpha),
        })

    res_df = pd.DataFrame(results)
    if len(res_df) > 1:
        try:
            reject = _fdc(res_df['p_value'].values)
            res_df['significant_bh'] = list(reject)
        except Exception:
            res_df['significant_bh'] = res_df['p_value'] < alpha
    else:
        res_df['significant_bh'] = res_df['p_value'] < alpha

    def interpret_effect(d):
        ad = abs(d)
        if ad < 0.147:
            return 'negligible'
        if ad < 0.33:
            return 'small'
        if ad < 0.474:
            return 'medium'
        return 'large'

    res_df['effect_interpretation'] = res_df['cliffs_delta'].apply(interpret_effect)
    return res_df.sort_values('p_value')
