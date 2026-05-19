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

try:
    import pingouin as pg
except ImportError:
    pg = None
    logger.info("pingouin not available — using pure-numpy effect size")


def _cliffs_delta(x, y):
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0
    more = 0
    for xi in x:
        more += int((y < xi).sum()) - int((y > xi).sum())
    return more / (nx * ny)


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
        if pg is not None:
            try:
                eff = pg.compute_effsize(g1, g0, eftype='cliffs')
            except Exception:
                eff = _cliffs_delta(g1.values, g0.values)
        else:
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
    if res_df.empty:
        return res_df
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
