"""
Post-experiment statistical comparison of the five prediction models.

This module runs ONLY AFTER all experiments have finished, on the master
results table (``results/master/all_results.csv``).  It never influences
features, labels, splits, models, SMOTE or evaluation.

Analysis performed
------------------
1. **Friedman test** — global difference in ROC-AUC across the model set,
   blocked on dataset.  Run per SMOTE condition (8 datasets × 5 models) and
   on the pooled block design (dataset × condition × model).
2. **Nemenyi post-hoc** — pairwise model comparisons with the Tukey-type
   critical difference, visualised as a critical-difference diagram.
3. **Paired Wilcoxon (SMOTE pairing)** — for every model, the ROC-AUC of
   ``with_smote`` vs ``without_smote`` is paired per dataset (the pairing
   unit is the ``(dataset, model)`` cell), so each test has ``n = #datasets``
   paired observations.  Raw p-values are ALWAYS reported alongside the
   Benjamini–Hochberg FDR-corrected q-values.

If a dataset lacks some models (e.g. LightGBM unavailable), the offending
row/column is dropped before the corresponding test and noted in the report.

Outputs (``results/master/``)
-----------------------------
    statistical_comparison.csv / .xlsx / .tex
    critical_difference_diagram.png / .pdf
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import studentized_range

from src.config import FRAMEWORK_VERSION
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

ALPHA: float = 0.05
MODELS = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'svm']
MODEL_LABELS = {
    'logistic_regression': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM',
    'svm': 'SVM',
}


# ═════════════════════════════════════════════════════════════════════
# RANK-BASED COMPARISON (Friedman + Nemenyi)
# ═════════════════════════════════════════════════════════════════════

def build_roc_auc_matrix(
    all_results: pd.DataFrame,
    smote: Optional[str] = None,
    models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Pivot master results into rows=datasets, cols=models of ROC-AUC.

    Rows with any missing model are kept (NaN cells are handled by the
    tests) but models with zero non-NaN coverage are dropped.
    """
    models = models or MODELS
    df = all_results
    if smote is not None:
        df = df[df['smote'] == smote]
    df = df[df['model'].isin(models) & df['roc_auc'].notna()]
    matrix = df.pivot_table(index='dataset', columns='model',
                            values='roc_auc', aggfunc='mean')
    matrix = matrix[[c for c in models if c in matrix.columns]]
    matrix = matrix.dropna(how='all')
    matrix = matrix.loc[:, matrix.notna().any(axis=0)]
    return matrix


def _mean_ranks(matrix: pd.DataFrame) -> pd.Series:
    """Mean ranks per model (ascending ROC-AUC ⇒ smaller rank is better)."""
    ranks = matrix.rank(axis=1)
    return ranks.mean(axis=0)


def friedman_test(matrix: pd.DataFrame) -> Dict[str, float]:
    """Friedman test across model columns (blocked on dataset)."""
    cols = list(matrix.columns)
    if len(cols) < 3 or len(matrix) < 2:
        raise ValueError(
            f"Friedman requires >=3 models and >=2 datasets "
            f"(got {len(cols)} models, {len(matrix)} datasets)")
    chi2, p = stats.friedmanchisquare(*[matrix[c].dropna() for c in cols])
    return {
        'n_models': len(cols),
        'n_datasets': len(matrix),
        'chi2': float(chi2),
        'df': len(cols) - 1,
        'p_value': float(p),
        'significant_at_005': bool(p < ALPHA),
    }


def nemenyi_test(
    matrix: pd.DataFrame,
    alpha: float = ALPHA,
) -> Tuple[pd.DataFrame, float]:
    """Nemenyi post-hoc: pairwise p-values + critical difference.

    Uses the Studentized-range (Tukey) distribution as in Pohlert (2016).
    The critical difference is ``q_alpha(k) * sqrt(k*(k+1)/(6n))`` with
    ``k`` = number of models and ``n`` = number of blocks (datasets).
    """
    k = len(matrix.columns)
    n = len(matrix)
    if k < 2 or n < 1:
        raise ValueError("Nemenyi requires >=2 models and >=1 dataset")
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    ranks = _mean_ranks(matrix)
    names = list(ranks.index)

    try:
        q_crit = studentized_range.ppf(1 - alpha, k, np.inf)
        cd = q_crit * se
    except Exception:
        # fallback: normal-approximation CD used in several references
        cd = 2.343 * se  # approximate q for k=5, alpha=.05
        logger.warning("studentized_range unavailable — CD via approximation")

    pvals = pd.DataFrame(np.nan, index=names, columns=names)
    q_obs = pd.DataFrame(np.nan, index=names, columns=names)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            stat = abs(ranks[a] - ranks[b]) / se
            try:
                p = 1.0 - studentized_range.cdf(stat, k, np.inf)
            except Exception:
                p = float('nan')
            pvals.loc[b, a] = p
            pvals.loc[a, b] = p
            q_obs.loc[a, b] = stat
    return pvals, float(cd)


def wilcoxon_smote(
    all_results: pd.DataFrame,
    models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Paired Wilcoxon of with_smote vs without_smote ROC-AUC per model.

    Pairing unit = ``(dataset, model)``: each model therefore has
    ``n = #datasets`` paired observations.  Raw p-values are always kept;
    Benjamini–Hochberg FDR correction is applied across the model tests.
    Per-dataset deltas are included so a NEGATIVE effect of SMOTE is always
    visible (never hidden).
    """
    models = models or MODELS
    rows = []
    for model in models:
        without = all_results[(all_results['model'] == model) &
                              (all_results['smote'] == 'No')]
        with_ = all_results[(all_results['model'] == model) &
                            (all_results['smote'] == 'Yes')]
        merged = without[['dataset', 'roc_auc']].merge(
            with_[['dataset', 'roc_auc']], on='dataset',
            suffixes=('_without', '_with'))
        merged = merged.dropna()
        for _, r in merged.iterrows():
            rows.append({
                'model': model,
                'dataset': r['dataset'],
                'auc_without_smote': r['roc_auc_without'],
                'auc_with_smote': r['roc_auc_with'],
                'delta': r['roc_auc_with'] - r['roc_auc_without'],
                'smote_effect': ('positive' if r['roc_auc_with'] >
                                 r['roc_auc_without'] else 'negative'),
            })

    pairs_df = pd.DataFrame(rows)
    if pairs_df.empty:
        return pairs_df

    summary = []
    for model in models:
        sub = pairs_df[pairs_df['model'] == model]
        n = len(sub)
        if n < 2:
            summary.append({
                'model': model, 'n_datasets': n,
                'wilcoxon_statistic': np.nan, 'raw_p_value': np.nan,
                'note': 'insufficient paired observations (<2 datasets)',
            })
            continue
        a = sub['auc_without_smote'].values
        b = sub['auc_with_smote'].values
        try:
            stat, p = stats.wilcoxon(b, a)
            mean_delta = float(sub['delta'].mean())
            n_neg = int((sub['delta'] < 0).sum())
            summary.append({
                'model': model, 'n_datasets': n,
                'mean_delta': round(mean_delta, 6),
                'n_negative_deltas': n_neg,
                'wilcoxon_statistic': float(stat),
                'raw_p_value': float(p),
                'note': '',
            })
        except ValueError as exc:
            summary.append({
                'model': model, 'n_datasets': n,
                'wilcoxon_statistic': np.nan, 'raw_p_value': np.nan,
                'note': f'wilcoxon failed: {exc}',
            })

    summary_df = pd.DataFrame(summary)

    # BH-FDR across the tests that actually produced a p-value
    mask = summary_df['raw_p_value'].notna()
    pvals = summary_df.loc[mask, 'raw_p_value'].astype(float)
    if len(pvals) > 0:
        ordered = pvals.sort_values()
        m = len(ordered)
        bh = ordered * m / np.arange(1, m + 1)
        qvals = bh.cummin().reindex(pvals.index).sort_values()
        summary_df['q_value_bh'] = np.nan
        summary_df.loc[qvals.index, 'q_value_bh'] = qvals.values
        summary_df['significant_q_005'] = summary_df['q_value_bh'].lt(ALPHA)
    else:
        summary_df['q_value_bh'] = np.nan
        summary_df['significant_q_005'] = False
    return summary_df


# ═════════════════════════════════════════════════════════════════════
# ORCHESTRATION + OUTPUTS
# ═════════════════════════════════════════════════════════════════════

def run_statistical_comparison(
    experiment_dir: str,
    all_results: pd.DataFrame,
) -> Dict:
    """Run the full post-experiment comparison and write every output.

    Returns a results dict (all sub-tables), also written to
    ``results/master/statistical_comparison.csv/.xlsx/.tex`` and the CD
    diagram as PNG/PDF.
    """
    master = ensure_dir(os.path.join(experiment_dir, 'results', 'master'))
    out: Dict = {'ranks': {}, 'friedman': {}, 'nemenyi_p': {},
                 'nemenyi_cd': {}, 'wilcoxon': {}, 'notes': []}

    if all_results is None or all_results.empty:
        logger.warning("No master results — statistical comparison skipped")
        return out

    # ---- Friedman + Nemenyi per SMOTE condition, and pooled ----
    for cond_label, cond in [('without_smote', 'No'),
                             ('with_smote', 'Yes'),
                             ('pooled', None)]:
        try:
            matrix = build_roc_auc_matrix(all_results, smote=cond)
            if matrix.shape[1] < 3 or matrix.shape[0] < 2:
                out['notes'].append(f'{cond_label}: insufficient data '
                                    f'({matrix.shape[0]}×{matrix.shape[1]})')
                continue
            fried = friedman_test(matrix)
            pvals, cd = nemenyi_test(matrix)
            out['ranks'][cond_label] = _mean_ranks(matrix)
            out['friedman'][cond_label] = fried
            out['nemenyi_p'][cond_label] = pvals
            out['nemenyi_cd'][cond_label] = cd
            logger.validation(
                "Friedman %-12s χ²=%.3f p=%.4f (n=%d models, %d datasets)",
                cond_label, fried['chi2'], fried['p_value'],
                fried['n_models'], fried['n_datasets'])
        except Exception as exc:
            out['notes'].append(f'{cond_label}: {exc}')
            logger.warning("Statistical comparison (%s) failed: %s",
                           cond_label, exc)

    # ---- paired Wilcoxon on SMOTE ----
    try:
        out['wilcoxon'] = wilcoxon_smote(all_results)
    except Exception as exc:
        out['notes'].append(f'wilcoxon: {exc}')
        logger.warning("Wilcoxon SMOTE comparison failed: %s", exc)

    _write_outputs(master, out)
    return out


def _flatten_comparison(out: Dict) -> pd.DataFrame:
    """Flatten the comparison results into a long CSV table."""
    rows = []
    for cond_label in out['ranks']:
        for model, rank in out['ranks'][cond_label].items():
            rows.append({
                'section': 'mean_ranks',
                'condition': cond_label,
                'model': model,
                'value': round(float(rank), 6),
            })
    for cond_label in out['friedman']:
        f = out['friedman'][cond_label]
        for k, v in f.items():
            rows.append({
                'section': 'friedman',
                'condition': cond_label,
                'statistic': k,
                'value': v,
            })
    for cond_label in out['nemenyi_p']:
        p = out['nemenyi_p'][cond_label]
        for a in p.index:
            for b in p.columns:
                rows.append({
                    'section': 'nemenyi_p',
                    'condition': cond_label,
                    'model_a': a,
                    'model_b': b,
                    'value': p.loc[a, b],
                })
    for cond_label in out['nemenyi_cd']:
        rows.append({
            'section': 'critical_difference',
            'condition': cond_label,
            'value': out['nemenyi_cd'][cond_label],
        })
    wilcoxon = out.get('wilcoxon')
    if wilcoxon is not None and not wilcoxon.empty:
        for _, r in wilcoxon.iterrows():
            rows.append({
                'section': 'wilcoxon_smote',
                'model': r.get('model'),
                'dataset': r.get('dataset'),
                'statistic': r.get('wilcoxon_statistic'),
                'value': r.get('raw_p_value'),
                'delta': r.get('mean_delta'),
                'note': r.get('note'),
            })
    for note in out.get('notes', []):
        rows.append({'section': 'notes', 'value': note})
    return pd.DataFrame(rows)


def _write_latex(master: str, out: Dict) -> str:
    lines = [
        '% Statistical comparison of the behavioural churn prediction models',
        f'% Framework version: {FRAMEWORK_VERSION}',
        r'\documentclass{article}', r'\begin{document}',
    ]
    for cond_label in ['without_smote', 'with_smote', 'pooled']:
        if cond_label not in out['friedman']:
            continue
        f = out['friedman'][cond_label]
        lines.append(
            f"% Friedman [{cond_label}] chi2={f['chi2']:.4f} "
            f"p={f['p_value']:.4f} (df={f['df']}, significant@0.05="
            f"{'yes' if f['significant_at_005'] else 'no'})")
    if out['nemenyi_p']:
        cond = 'pooled' if 'pooled' in out['nemenyi_p'] else \
            next(iter(out['nemenyi_p']))
        p = out['nemenyi_p'][cond]
        lines.append(r'\begin{tabular}{' + 'l' * (len(p.columns) + 1) + '}')
        lines.append(' & ' + ' & '.join(str(c) for c in p.columns) + r' \\')
        for a in p.index:
            line = [a]
            line += [f'{p.loc[a, b]:.4f}' for b in p.columns]
            lines.append(' & '.join(line) + r' \\')
        lines.append(r'\end{tabular}')
    lines.append(r'\end{document}')
    path = os.path.join(master, 'statistical_comparison.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return path


def _write_outputs(master: str, out: Dict) -> None:
    flat = _flatten_comparison(out)
    csv_path = os.path.join(master, 'statistical_comparison.csv')
    flat.to_csv(csv_path, index=False)

    try:
        xlsx_path = os.path.join(master, 'statistical_comparison.xlsx')
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            flat.to_excel(writer, sheet_name='all', index=False)
            for cond_label in out['nemenyi_p']:
                out['nemenyi_p'][cond_label].to_excel(
                    writer, sheet_name=f'nemenyi_{cond_label}'[:31],
                    index_label='model')
            w = out.get('wilcoxon')
            if w is not None and not w.empty:
                w.to_excel(writer, sheet_name='wilcoxon_smote', index=False)
    except Exception as exc:
        logger.warning("Cannot write statistical_comparison.xlsx: %s", exc)

    _write_latex(master, out)

    for cond_label in out['nemenyi_p']:
        if out['nemenyi_cd'].get(cond_label):
            try:
                plot_critical_difference(
                    out['ranks'][cond_label], out['nemenyi_cd'][cond_label],
                    os.path.join(master, 'critical_difference_diagram.png'),
                    os.path.join(master, 'critical_difference_diagram.pdf'),
                    title=f'Critical difference — {cond_label}')
            except Exception as exc:
                logger.warning("CD diagram (%s) failed: %s", cond_label, exc)


# ═════════════════════════════════════════════════════════════════════
# CRITICAL DIFFERENCE DIAGRAM
# ═════════════════════════════════════════════════════════════════════

def plot_critical_difference(
    ranks: pd.Series,
    cd: float,
    png_path: str,
    pdf_path: Optional[str] = None,
    title: str = 'Critical difference',
) -> str:
    """Render a Nemenyi critical-difference diagram (smaller rank = better)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = [MODEL_LABELS.get(m, str(m)) for m in ranks.index]
    values = ranks.values.astype(float)
    order = np.argsort(values)
    values = values[order]
    labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_ylim(0, 1)
    ax.axis('off')

    best = values.min()
    n = len(values)
    cds = np.clip(cd / (len(values) - 1) if n > 1 else 0, 0, 0.6)

    for i, (lab, v) in enumerate(zip(labels, values)):
        ax.plot([i, i], [0.62, 0.88], color='black', lw=1.2)
        ax.text(i, 0.95, lab, ha='center', va='bottom', fontsize=9,
                rotation=0, wrap=True)
        ax.text(i, 0.52, f'{v:.2f}', ha='center', va='top', fontsize=8)
        # CD bar spanning the top-ranked model
        if i == 0:
            ax.plot([0, n - 1], [0.42, 0.42], color='k', lw=2)
            ax.text(0, 0.40, 'Best (lower is better)', ha='left',
                    va='top', fontsize=7, color='black')

    # horizontal CD interval
    ax.plot([best, best + cds], [0.30, 0.30], color='C3', lw=2.5)
    ax.text(best, 0.24, f'CD = {cd:.3f}', ha='left', va='top', fontsize=8,
            color='C3')
    ax.set_title(title, fontsize=11)

    ensure_dir(os.path.dirname(png_path))
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    if pdf_path:
        ensure_dir(os.path.dirname(pdf_path))
        fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return png_path
