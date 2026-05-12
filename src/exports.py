"""
Export pipeline artefacts: models, processed data, metrics, SHAP values,
risk scores, quality reports, experiment metadata, and cross-dataset
master results table.

All paths are derived from the central config and created automatically.
Overwrite-safe — existing files are silently replaced.

Isolated per-experiment outputs (Section 30 of the experiment spec) are
written by save_experiment_artifacts() into::

    {results_dir}/without_smote/{dataset}/
    {results_dir}/with_smote/{dataset}/
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, List

from src.config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def save_models(models: Dict[str, Any], base_dir: Optional[str] = None) -> None:
    d = ensure_dir(base_dir or MODELS_DIR)
    for name, model in models.items():
        path = os.path.join(d, f'{name}.joblib')
        joblib.dump(model, path)
        logger.info("Model saved: %s", path)


def save_processed_data(
    train_X: pd.DataFrame, test_X: pd.DataFrame,
    train_y: pd.DataFrame, test_y: pd.DataFrame,
    base_dir: Optional[str] = None,
) -> None:
    d = ensure_dir(base_dir or PROCESSED_DIR)
    train_X.to_csv(os.path.join(d, 'train_features.csv'), index=True)
    test_X.to_csv(os.path.join(d, 'test_features.csv'), index=True)
    train_y.to_csv(os.path.join(d, 'train_labels.csv'), index=True)
    test_y.to_csv(os.path.join(d, 'test_labels.csv'), index=True)
    logger.info("Processed data saved to %s", d)


def save_evaluation_table(metrics_df: pd.DataFrame,
                          filename: str = 'model_metrics.csv',
                          base_dir: Optional[str] = None) -> None:
    path = os.path.join(
        ensure_dir(os.path.join(base_dir or RESULTS_DIR, 'model_metrics')),
        filename)
    metrics_df.to_csv(path, index=False)
    logger.info("Evaluation table saved: %s", path)


def save_shap_values(
    shap_values: Any, feature_names: list, model_name: str,
    base_dir: Optional[str] = None,
) -> None:
    d = ensure_dir(os.path.join(base_dir or RESULTS_DIR, 'shap_values'))
    path = os.path.join(d, f'{model_name}_shap_values.csv')
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


def save_risk_scores(risk_df: pd.DataFrame, model_name: str,
                     base_dir: Optional[str] = None) -> None:
    d = ensure_dir(os.path.join(base_dir or RESULTS_DIR, 'risk_scoring'))
    path = os.path.join(d, f'{model_name}_risk_scores.csv')
    risk_df.to_csv(path, index=False)
    logger.info("Risk scores saved: %s", path)


def save_data_quality_report(report_dict: dict,
                             base_dir: Optional[str] = None) -> None:
    d = ensure_dir(os.path.join(base_dir or RESULTS_DIR, 'data_quality'))
    txt_path = os.path.join(d, 'data_quality_report.txt')
    with open(txt_path, 'w') as f:
        for k, v in report_dict.items():
            f.write(f"{k}: {v}\n")
    pd.DataFrame([report_dict]).to_csv(
        os.path.join(d, 'data_quality_summary.csv'), index=False)
    logger.info("Data quality report saved: %s", txt_path)


def save_experiment_metadata(metadata: dict,
                             base_dir: Optional[str] = None) -> None:
    d = ensure_dir(os.path.join(base_dir or RESULTS_DIR, 'experiments'))
    log_file = os.path.join(d, 'experiment_log.csv')
    df = pd.DataFrame([metadata])
    if os.path.exists(log_file):
        df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df.to_csv(log_file, index=False)
    logger.info("Experiment metadata appended to %s", log_file)


def save_experiment_artifacts(
    dataset: str,
    use_smote: bool,
    metrics_df: pd.DataFrame,
    y_test: pd.Series,
    prob_dict: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    results_dir: Optional[str] = None,
) -> str:
    """Write the isolated per-experiment output bundle (Section 30).

    Writes into ``{results_dir}/without_smote/{dataset}/`` or
    ``{results_dir}/with_smote/{dataset}/``:
        metrics.csv, predictions.csv, model_comparison.csv,
        classification_report.txt, experiment_metadata.json

    Returns the created directory path.
    """
    from sklearn.metrics import classification_report

    condition = 'with_smote' if use_smote else 'without_smote'
    out_dir = ensure_dir(os.path.join(results_dir or RESULTS_DIR,
                                      condition, dataset))

    # metrics.csv — full per-model metric table
    metrics_df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)

    # predictions.csv — true labels + per-model probabilities
    pred_rows = {}
    pred_rows['customer_id'] = list(y_test.index)
    pred_rows['y_test'] = y_test.values
    for name, probs in (prob_dict or {}).items():
        if probs is not None:
            pred_rows[f'{name}_proba'] = probs
    pd.DataFrame(pred_rows).to_csv(
        os.path.join(out_dir, 'predictions.csv'), index=False)

    # model_comparison.csv — compact per-model comparison
    compare_cols = [c for c in [
        'model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
        'avg_precision', 'balanced_accuracy', 'mcc', 'brier_score',
        'calibration_error', 'training_time', 'inference_time',
    ] if c in metrics_df.columns]
    metrics_df[compare_cols].to_csv(
        os.path.join(out_dir, 'model_comparison.csv'), index=False)

    # classification_report.txt — per-model classification reports
    report_lines = []
    for name, probs in (prob_dict or {}).items():
        if probs is None:
            continue
        y_pred = (probs >= 0.5).astype(int)
        report_lines.append(f"=== {name} ===")
        report_lines.append(
            classification_report(y_test, y_pred, zero_division=0.0)
        )
    with open(os.path.join(out_dir, 'classification_report.txt'), 'w') as f:
        f.write("\n".join(report_lines))

    # experiment_metadata.json
    with open(os.path.join(out_dir, 'experiment_metadata.json'), 'w') as f:
        json.dump({k: (v if isinstance(v, (str, int, float, bool, type(None)))
                       else str(v))
                   for k, v in meta.items()},
                  f, indent=2, default=str)

    logger.info("Per-experiment artifacts saved to %s", out_dir)
    return out_dir


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
        results/cross_dataset/master_results.csv

    This enables direct cross-dataset comparison for the research paper.

    Entries are validated before appending — master results schema is
    enforced automatically.
    """
    from src.validators import validate_master_results_entry

    d = ensure_dir(os.path.join(RESULTS_DIR, 'cross_dataset'))
    path = os.path.join(d, 'master_results.csv')

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
