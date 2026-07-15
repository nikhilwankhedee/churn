"""
Behavioural framework quadrant analysis.

A SEPARATE, post-experiment analysis that places every dataset in the
framework quadrant defined by two pre-registered axes:

    Behavioural Continuity    — how sustained and regular a customer's
                                engagement is (repeat purchases, event
                                volume, history span)
    Behavioural Observability — how much fine-grained behavioural signal the
                                data exposes (event-type richness, feature
                                coverage, missingness)

Scoring uses ONLY the raw-data characteristics recorded in
``dataset_characteristics.csv`` (repeat_customer_ratio, event volume, time
span, event types, feature counts, missingness).  It is deliberately fixed
BEFORE any performance overlay and is NEVER tuned to predictive performance.
The quadrant therefore must not and does not influence features, labels,
splits, models, SMOTE or evaluation; model results are only overlaid
afterwards as an outcome.

Outputs (``results/framework/``)
-------------------------------
    framework_methodology.json    — pre-registered axis & scoring definitions
    dataset_positions.csv         — per-dataset axis scores + quadrant
    framework_quadrant_plot.png/.pdf
    quadrant_performance.csv      — quadrant × mean ROC-AUC overlay
"""
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import FRAMEWORK_VERSION
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

# ── Pre-registered axis definitions ─────────────────────────────────────
CONTINUITY_COMPONENTS = {
    'repeat_customer_ratio': {'weight': 0.4, 'log': False},
    'avg_events_per_customer': {'weight': 0.3, 'log': True},
    'time_span_days': {'weight': 0.3, 'log': True},
}
OBSERVABILITY_COMPONENTS = {
    'n_event_types': {'weight': 0.5, 'log': True},
    'n_numerical_features': {'weight': 0.3, 'log': True},
    'missing_value_pct': {'weight': 0.2, 'log': False, 'invert': True},
}

QUADRANT_THRESHOLD = 0.5
QUADRANTS = {
    (True, True): 'Repeated & Observable',
    (True, False): 'Repeated & Opaque',
    (False, True): 'Sporadic & Observable',
    (False, False): 'Sporadic & Opaque',
}

# Columns in dataset_characteristics.csv that encode PREDICTIVE PERFORMANCE.
# These are never allowed to influence axis scores (independence guard).
PERFORMANCE_COLUMNS = [c for c in [
    'avg_accuracy', 'avg_precision', 'avg_recall', 'avg_f1', 'avg_roc_auc',
    'avg_pr_auc', 'avg_brier_score']]


def _transform(value: float, spec: Dict) -> float:
    v = value
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if spec.get('log'):
        v = float(np.log1p(v))
    if spec.get('invert'):
        v = 1.0 - v
    return v


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0,1]; constant input maps to the 0.5 midpoint."""
    values = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


def score_axis(
    characteristics: pd.DataFrame,
    components: Dict,
    axis_name: str,
) -> pd.Series:
    """Score one axis for every dataset using only raw-data characteristics."""
    comp_scores = pd.DataFrame(index=characteristics.index)
    for col, spec in components.items():
        if col not in characteristics.columns:
            raise ValueError(
                f"Axis '{axis_name}' requires characteristic '{col}', "
                f"which is missing from dataset_characteristics.csv")
        comp_scores[col] = characteristics[col].apply(
            lambda v: _transform(v, spec))
    normalized = {}
    for col in components:
        normalized[col] = _minmax_normalize(comp_scores[col].values)
        # missing values are left NaN after normalisation
        normalized[col][pd.isna(comp_scores[col].values)] = np.nan
    weights = np.array([components[c]['weight'] for c in components])
    norm = np.vstack([normalized[c] for c in components])
    axis = np.nansum(norm * weights[:, None], axis=0)
    axis[~np.isfinite(axis)] = np.nan
    return pd.Series(axis, index=characteristics.index)


def assign_quadrant(continuity: float, observability: float) -> str:
    key = (continuity >= QUADRANT_THRESHOLD,
           observability >= QUADRANT_THRESHOLD)
    return QUADRANTS[key]


def score_datasets(
    characteristics: pd.DataFrame,
) -> pd.DataFrame:
    """Compute axis scores + quadrant assignments for all datasets.

    ``characteristics`` must come from ``dataset_characteristics.csv``; any
    ``avg_*`` performance columns are explicitly excluded from the scoring.
    """
    if characteristics is None or characteristics.empty:
        raise ValueError('No dataset characteristics available')
    if 'dataset' not in characteristics.columns:
        raise ValueError('dataset_characteristics.csv has no "dataset" column')

    work = characteristics.copy()
    for col in PERFORMANCE_COLUMNS:
        work = work.drop(columns=[col], errors='ignore')

    continuity = score_axis(work, CONTINUITY_COMPONENTS, 'Behavioural Continuity')
    observability = score_axis(work, OBSERVABILITY_COMPONENTS, 'Behavioural Observability')

    positions = pd.DataFrame({
        'dataset': work['dataset'],
        'continuity_score': continuity,
        'observability_score': observability,
    })
    positions['quadrant'] = [
        assign_quadrant(c, o)
        if pd.notna(c) and pd.notna(o) else 'Unscored'
        for c, o in zip(continuity, observability)]
    return positions


def overlay_performance(
    positions: pd.DataFrame,
    all_results: pd.DataFrame,
) -> pd.DataFrame:
    """Overlay mean ROC-AUC per quadrant (quadrant assignment fixed first).

    Performance is computed from ALL non-baseline results of every dataset
    in each quadrant and is purely descriptive — it never feeds back into
    the axis scoring.
    """
    rows = []
    if all_results is not None and not all_results.empty:
        perf = all_results[~all_results['model'].astype(str)
                           .str.contains('baseline', na=False)]
        perf = perf[perf['roc_auc'].notna()]
    else:
        perf = pd.DataFrame()

    for quadrant in sorted(positions['quadrant'].unique()):
        ds = positions.loc[positions['quadrant'] == quadrant, 'dataset']
        sub = perf[perf['dataset'].isin(ds)] if not perf.empty else pd.DataFrame()
        rows.append({
            'quadrant': quadrant,
            'n_datasets': int(len(ds)),
            'n_experiments': int(len(sub)),
            'mean_roc_auc': float(sub['roc_auc'].mean()) if len(sub) else np.nan,
            'median_roc_auc': float(sub['roc_auc'].median()) if len(sub) else np.nan,
            'mean_roc_auc_with_smote': float(
                sub.loc[sub['smote'] == 'Yes', 'roc_auc'].mean()) if len(sub) else np.nan,
            'mean_roc_auc_without_smote': float(
                sub.loc[sub['smote'] == 'No', 'roc_auc'].mean()) if len(sub) else np.nan,
        })
    return pd.DataFrame(rows)


def run_framework_analysis(
    experiment_dir: str,
    all_results: Optional[pd.DataFrame] = None,
) -> Dict:
    """Run the framework quadrant analysis and write all framework outputs."""
    master = os.path.join(experiment_dir, 'results', 'master')
    chars_path = os.path.join(master, 'dataset_characteristics.csv')
    if not os.path.isfile(chars_path):
        logger.warning('dataset_characteristics.csv not found — '
                       'framework analysis skipped')
        return {}

    characteristics = pd.read_csv(chars_path)
    positions = score_datasets(characteristics)
    quadrant_perf = overlay_performance(positions, all_results)

    framework_dir = ensure_dir(os.path.join(experiment_dir, 'results',
                                            'framework'))
    positions.to_csv(os.path.join(framework_dir, 'dataset_positions.csv'),
                     index=False)
    quadrant_perf.to_csv(os.path.join(framework_dir,
                                      'quadrant_performance.csv'), index=False)

    methodology = {
        'framework_version': FRAMEWORK_VERSION,
        'analysis': 'Post-experiment behavioural framework quadrant',
        'status': 'Independent of model results (performance overlaid after '
                  'quadrant assignment only)',
        'axes': {
            'Behavioural Continuity': {
                'definition': 'How sustained and regular a customer\u2019s '
                              'engagement is.',
                'components': CONTINUITY_COMPONENTS,
            },
            'Behavioural Observability': {
                'definition': 'How much fine-grained behavioural signal the '
                              'data exposes.',
                'components': OBSERVABILITY_COMPONENTS,
            },
        },
        'normalization': 'min-max across datasets per component; constant '
                         'components mapped to the 0.5 midpoint',
        'quadrant_threshold': QUADRANT_THRESHOLD,
        'quadrants': {
            'Repeated & Observable': 'continuity >= 0.5 and observability >= 0.5',
            'Repeated & Opaque': 'continuity >= 0.5 and observability < 0.5',
            'Sporadic & Observable': 'continuity < 0.5 and observability >= 0.5',
            'Sporadic & Opaque': 'continuity < 0.5 and observability < 0.5',
        },
        'excluded_columns': PERFORMANCE_COLUMNS,
        'independence_guarantee': 'Axis scores use only raw-data '
                                  'characteristics; predictive performance '
                                  'is overlaid afterwards and never '
                                  'feeds back.',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(framework_dir, 'framework_methodology.json'),
              'w') as f:
        json.dump(methodology, f, indent=2)

    plot_quadrant(
        positions,
        os.path.join(framework_dir, 'framework_quadrant_plot.png'),
        os.path.join(framework_dir, 'framework_quadrant_plot.pdf'),
    )

    for _, r in positions.iterrows():
        logger.validation("Framework quadrant | %-16s continuity=%.3f "
                          "observability=%.3f → %s",
                          r['dataset'], r['continuity_score'],
                          r['observability_score'], r['quadrant'])

    return {'positions': positions,
            'quadrant_performance': quadrant_perf,
            'methodology': methodology}


def plot_quadrant(
    positions: pd.DataFrame,
    png_path: str,
    pdf_path: Optional[str] = None,
) -> str:
    """Scatter of dataset positions with quadrant grid and labels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.axhline(QUADRANT_THRESHOLD, color='grey', lw=1, ls='--')
    ax.axvline(QUADRANT_THRESHOLD, color='grey', lw=1, ls='--')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Behavioural Continuity score', fontsize=11)
    ax.set_ylabel('Behavioural Observability score', fontsize=11)
    ax.set_title('Behavioural framework quadrant', fontsize=12)

    for _, r in positions.iterrows():
        if r['quadrant'] == 'Unscored':
            continue
        ax.scatter(r['continuity_score'], r['observability_score'],
                   s=160, edgecolors='k', linewidths=1, zorder=3)
        ax.annotate(r['dataset'], (r['continuity_score'],
                                   r['observability_score']),
                    textcoords='offset points', xytext=(8, 8), fontsize=9)

    ax.text(0.25, 0.5, 'Sporadic &\nOpaque', ha='center', va='center',
            fontsize=12, color='0.75')
    ax.text(0.75, 0.5, 'Repeated &\nOpaque', ha='center', va='center',
            fontsize=12, color='0.75')
    ax.text(0.25, 0.95, 'Sporadic &\nObservable', ha='center', va='center',
            fontsize=12, color='0.75')
    ax.text(0.75, 0.95, 'Repeated &\nObservable', ha='center', va='center',
            fontsize=12, color='0.75')

    legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor='C0',
                     markersize=9, label='Dataset')]
    ax.legend(handles=legend, loc='lower right')

    ensure_dir(os.path.dirname(png_path))
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    if pdf_path:
        ensure_dir(os.path.dirname(pdf_path))
        fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return png_path
