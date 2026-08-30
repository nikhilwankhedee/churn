"""
Export pipeline artefacts: models, processed data, metrics, SHAP values,
risk scores, quality reports, experiment metadata, and cross-dataset
master results table.

All paths are derived from the central config and created automatically.
Overwrite-safe — existing files are silently replaced.
"""
import os
import joblib
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, List

from src.run_context import (
    models_dir, processed_dir, results_dir, master_results_path,
)
from src.utils import get_logger

logger = get_logger(__name__)


def save_models(models: Dict[str, Any], suffix: str = '') -> None:
    d = models_dir()
    for name, model in models.items():
        path = os.path.join(d, f'{name}{suffix}.joblib')
        joblib.dump(model, path)
        logger.info("Model saved: %s", path)


def save_processed_data(
    train_X: pd.DataFrame, test_X: pd.DataFrame,
    train_y: pd.DataFrame, test_y: pd.DataFrame,
    suffix: str = '',
) -> None:
    d = processed_dir()
    train_X.to_csv(os.path.join(d, f'train_features{suffix}.csv'), index=True)
    test_X.to_csv(os.path.join(d, f'test_features{suffix}.csv'), index=True)
    train_y.to_csv(os.path.join(d, f'train_labels{suffix}.csv'), index=True)
    test_y.to_csv(os.path.join(d, f'test_labels{suffix}.csv'), index=True)
    logger.info("Processed data saved to %s", d)


def save_evaluation_table(metrics_df: pd.DataFrame,
                          filename: str = 'model_metrics.csv') -> None:
    path = os.path.join(results_dir('model_metrics'), filename)
    metrics_df.to_csv(path, index=False)
    logger.info("Evaluation table saved: %s", path)


def save_shap_values(
    shap_values: Any, feature_names: list, model_name: str, suffix: str = '',
) -> None:
    d = results_dir('shap_values')
    path = os.path.join(d, f'{model_name}_shap_values{suffix}.csv')
    if shap_values is None:
        logger.warning("No SHAP values to save for %s", model_name)
        return
    try:
        sv = np.asarray(shap_values)
        if sv.ndim == 1:
            sv = sv.reshape(-1, 1)
        n_cols = min(sv.shape[1], len(feature_names))
        cols = feature_names[:n_cols]
        df = pd.DataFrame(sv[:, :n_cols], columns=cols)
        df.to_csv(path, index=False)
        logger.info("SHAP values saved: %s", path)
    except Exception as exc:
        logger.warning("Failed to save SHAP values for %s: %s", model_name, exc)


def save_risk_scores(risk_df: pd.DataFrame, model_name: str, suffix: str = '') -> None:
    d = results_dir('risk_scoring')
    path = os.path.join(d, f'{model_name}_risk_scores{suffix}.csv')
    risk_df.to_csv(path, index=False)
    logger.info("Risk scores saved: %s", path)


def save_data_quality_report(report_dict: dict, suffix: str = '') -> None:
    d = results_dir('data_quality')
    txt_path = os.path.join(d, f'data_quality_report{suffix}.txt')
    with open(txt_path, 'w') as f:
        for k, v in report_dict.items():
            f.write(f"{k}: {v}\n")
    pd.DataFrame([report_dict]).to_csv(
        os.path.join(d, f'data_quality_summary{suffix}.csv'), index=False)
    logger.info("Data quality report saved: %s", txt_path)


def save_experiment_metadata(metadata: dict, suffix: str = '') -> None:
    d = results_dir('experiments')
    log_file = os.path.join(d, f'experiment_log{suffix}.csv')
    df = pd.DataFrame([metadata])
    if os.path.exists(log_file):
        df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df.to_csv(log_file, index=False)
    logger.info("Experiment metadata appended to %s", log_file)


def append_to_master_results(
    dataset_name: str,
    ecosystem_type: str,
    churn_rate: float,
    imbalance_ratio: float,
    dominant_feature_group: str,
    metrics: pd.DataFrame,
) -> None:
    """Append a dataset's results to the cross-dataset master table.

    The master table is stored at:
        results/cross_dataset/master_results_<mode>.csv
    (mode is "original" or "smote", so the two sweeps never collide).

    This enables direct cross-dataset comparison for the research paper.

    Entries are validated before appending — master results schema is
    enforced automatically.
    """
    from src.validators import validate_master_results_entry

    path = master_results_path()

    rows = []
    for _, row in metrics.iterrows():
        rows.append({
            'dataset': dataset_name,
            'ecosystem_type': ecosystem_type,
            'model': row.get('model', 'unknown'),
            'roc_auc': row.get('roc_auc', np.nan),
            'pr_auc': row.get('avg_precision', np.nan),
            'f1': row.get('f1', np.nan),
            'precision': row.get('precision', np.nan),
            'recall': row.get('recall', np.nan),
            'brier_score': row.get('brier_score', np.nan),
            'calibration_error': row.get('calibration_error', np.nan),
            'churn_rate': churn_rate,
            'imbalance_ratio': imbalance_ratio,
            'dominant_feature_group': dominant_feature_group,
        })

    # ── Validate before writing ──────────────────────────────────────
    validation = validate_master_results_entry(rows, existing_path=path)
    if not validation['valid']:
        logger.error(
            "MasterResults | Validation failed — not appending: %s",
            validation['errors'],
        )
        return

    new_df = pd.DataFrame(rows)

    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing = existing[existing['dataset'] != dataset_name]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(path, index=False)
    logger.info(
        "MasterResults | Updated — dataset '%s' (%s) appended to %s",
        dataset_name, ecosystem_type, path,
    )
