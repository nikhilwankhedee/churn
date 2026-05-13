"""
Experiment tracking: captures configuration, validation results, metrics,
environment info, and timestamps for full reproducibility and auditability.

Every experiment run produces a complete metadata record stored in:
    results/experiments/experiment_log.csv

This enables:
  - Full run traceability
  - Cross-dataset comparison validation
  - Debugging reproducibility issues
  - Paper methods section authoring
"""
import datetime
import platform
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from src.config import RANDOM_SEED, TRAIN_SPLIT_QUANTILE, PREDICTION_WINDOW_DAYS
from src.utils import get_logger

logger = get_logger(__name__)


def log_experiment(
    metrics_summary: pd.DataFrame,
    train_cutoff,
    test_cutoff,
    model_names: List[str],
    best_model: str,
    extra_info: Optional[Dict[str, Any]] = None,
    validation_reports: Optional[Dict[str, Any]] = None,
    behavioral_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Capture full experiment metadata for reproducibility.

    Parameters
    ----------
    metrics_summary : pd.DataFrame
        Evaluation metrics for all models.
    train_cutoff, test_cutoff : pd.Timestamp
        Temporal split dates.
    model_names : list of str
        Names of trained models.
    best_model : str
        Name of the best-performing model.
    extra_info : dict, optional
        Additional pipeline metadata.
    validation_reports : dict, optional
        Output from schema/behavioral/output validators.
    behavioral_stats : dict, optional
        Computed behavioral statistics.

    Returns
    -------
    dict of all captured metadata.
    """
    meta: Dict[str, Any] = {
        'timestamp': datetime.datetime.now().isoformat(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'random_seed': RANDOM_SEED,
        'train_cutoff': str(
            train_cutoff.date() if hasattr(train_cutoff, 'date')
            else train_cutoff
        ),
        'test_cutoff': str(
            test_cutoff.date() if hasattr(test_cutoff, 'date')
            else test_cutoff
        ),
        'prediction_window_days': PREDICTION_WINDOW_DAYS,
        'train_split_quantile': TRAIN_SPLIT_QUANTILE,
        'models': ','.join(model_names),
        'best_model': best_model,
    }

    # ── Per-model metrics ────────────────────────────────────────────
    metric_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc',
                   'brier_score', 'avg_precision', 'calibration_error']
    for _, row in metrics_summary.iterrows():
        mname = row.get('model', 'unknown')
        for col in metric_cols:
            if col in row and not (isinstance(row[col], float)
                                    and np.isnan(row[col])):
                meta[f'{mname}_{col}'] = row[col]

    # ── Validation metadata ──────────────────────────────────────────
    if validation_reports:
        schema = validation_reports.get('schema', {})
        meta['schema_n_rows'] = schema.get('n_rows', '')
        meta['schema_n_columns'] = schema.get('n_columns', '')
        meta['schema_detected_columns'] = ','.join(
            schema.get('detected_columns', [])
        )
        meta['schema_missing_optional'] = ','.join(
            schema.get('missing_optional_columns', [])
        )
        meta['schema_enabled_groups'] = ','.join(
            schema.get('enabled_feature_groups', [])
        )
        meta['schema_disabled_groups'] = ','.join(
            schema.get('disabled_feature_groups', [])
        )
        meta['schema_errors'] = len(schema.get('errors', []))
        meta['schema_warnings'] = len(schema.get('warnings', []))

        output = validation_reports.get('outputs', {})
        meta['output_files_missing'] = len(output.get('files_missing', []))
        meta['output_metric_issues'] = len(output.get('metric_issues', []))
        meta['output_prob_issues'] = len(output.get('probability_issues', []))

    # ── Behavioral statistics ────────────────────────────────────────
    if behavioral_stats:
        stats = behavioral_stats.get('statistics', {})
        for k, v in stats.items():
            meta[f'behavioral_{k}'] = v
        meta['behavioral_warnings'] = len(behavioral_stats.get('warnings', []))
        meta['behavioral_critical'] = len(
            behavioral_stats.get('critical_warnings', [])
        )

    # ── Extra info (dataset, churn window, etc.) ─────────────────────
    if extra_info:
        meta.update(extra_info)

    logger.validation(
        "Experiment | Metadata captured (%d keys) for reproducibility",
        len(meta),
    )
    return meta
