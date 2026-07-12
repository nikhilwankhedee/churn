"""
Best-model persistence and reload verification.

For every dataset × SMOTE condition the framework identifies the best model by
ROC-AUC (never baselines), persists it as ``best_model.pkl`` together with a
machine-readable metadata file, then RELOADS it and re-verifies that the
re-computed predictions reproduce the stored predictions and the master
ROC-AUC exactly (deterministic models, same data ⇒ identical output).

Artifacts
---------
    {results_dir}/master/best_models/{dataset}/{condition}/best_model.pkl
    {results_dir}/master/best_models/{dataset}/{condition}/best_model_metadata.json
    {results_dir}/master/model_persistence_report.csv

The reload verification result is written into the report table; if any model
fails the identity check the overall status is ``FAIL``.
"""
import json
import os
import time
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import FRAMEWORK_VERSION
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

AUC_TOLERANCE: float = 1e-6
STATUS_PASS = 'PASS'
STATUS_FAIL = 'FAIL'
STATUS_SKIPPED = 'SKIPPED'


def _condition_dir(experiment_dir: str, dataset: str, use_smote: bool) -> str:
    cond = 'with_smote' if use_smote else 'without_smote'
    return os.path.join(experiment_dir, 'results', cond, dataset)


def _best_model_row(metrics_df: pd.DataFrame) -> Optional[pd.Series]:
    if metrics_df is None or metrics_df.empty or 'model' not in metrics_df.columns:
        return None
    non_baseline = metrics_df[
        ~metrics_df['model'].astype(str).str.contains('baseline', na=False)
    ]
    if non_baseline.empty or 'roc_auc' not in non_baseline.columns:
        return None
    valid = non_baseline[non_baseline['roc_auc'].notna()]
    if valid.empty:
        return None
    return valid.sort_values('roc_auc', ascending=False).iloc[0]


def _read_metrics(base_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(base_dir, 'metrics.csv')
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return None


def persist_best_models(
    experiment_dir: str,
    all_results: pd.DataFrame,
) -> pd.DataFrame:
    """Persist one best model per dataset × SMOTE condition.

    The best model is selected by ROC-AUC from the per-condition metrics
    table (baselines excluded) and copied from the per-condition model file
    (``{model}.joblib``) into ``results/master/best_models/...``.

    Returns a report DataFrame with one row per condition.
    """
    root = ensure_dir(os.path.join(experiment_dir, 'results', 'master',
                                   'best_models'))
    records: List[Dict[str, Any]] = []

    if all_results is None or all_results.empty:
        logger.warning("No master results — skipping model persistence")
        return pd.DataFrame(records)

    pairs = set()
    for _, row in all_results.iterrows():
        pairs.add((str(row.get('dataset')), bool(row.get('smote') in ('Yes', True, 'True'))))

    for dataset, use_smote in sorted(pairs):
        base_dir = _condition_dir(experiment_dir, dataset, use_smote)
        cond = 'with_smote' if use_smote else 'without_smote'
        record = {'dataset': dataset, 'condition': cond, 'status': STATUS_PASS}

        metrics_df = _read_metrics(base_dir)
        best = _best_model_row(metrics_df)
        if best is None:
            record.update(status=STATUS_SKIPPED,
                          note='No evaluable model (metrics.csv missing/empty)')
            records.append(record)
            continue

        model_name = str(best['model'])
        src = os.path.join(base_dir, f'{model_name}.joblib')
        if not os.path.isfile(src):
            record.update(status=STATUS_SKIPPED,
                          note=f'Model artifact {model_name}.joblib not found')
            records.append(record)
            continue

        out_dir = ensure_dir(os.path.join(root, dataset, cond))
        pkl_path = os.path.join(out_dir, 'best_model.pkl')
        meta_path = os.path.join(out_dir, 'best_model_metadata.json')

        try:
            model = joblib.load(src)
            joblib.dump(model, pkl_path)
        except Exception as exc:
            record.update(status=STATUS_FAIL, note=f'persist failed: {exc}')
            records.append(record)
            continue

        feature_names: List[str] = []
        feat_path = os.path.join(base_dir, 'test_features.csv')
        if os.path.isfile(feat_path):
            try:
                feature_names = list(pd.read_csv(feat_path, nrows=1).columns[1:])
            except Exception:
                pass

        metadata = {
            'dataset': dataset,
            'condition': cond,
            'model': model_name,
            'roc_auc': float(best.get('roc_auc')) if pd.notna(best.get('roc_auc')) else None,
            'avg_precision': float(best.get('avg_precision')) if pd.notna(best.get('avg_precision')) else None,
            'f1': float(best.get('f1')) if pd.notna(best.get('f1')) else None,
            'n_test': int(best.get('n_test')) if pd.notna(best.get('n_test')) else None,
            'n_pos': int(best.get('n_pos')) if pd.notna(best.get('n_pos')) else None,
            'n_neg': int(best.get('n_neg')) if pd.notna(best.get('n_neg')) else None,
            'feature_names': feature_names,
            'source_model_file': os.path.basename(src),
            'source_test_features': os.path.basename(feat_path) if os.path.isfile(feat_path) else None,
            'selection_criterion': 'roc_auc',
            'framework_version': FRAMEWORK_VERSION,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        record.update(
            model=model_name,
            roc_auc=metadata['roc_auc'],
            best_model_file=os.path.relpath(pkl_path, experiment_dir),
            metadata_file=os.path.relpath(meta_path, experiment_dir),
            note='persisted',
        )
        logger.validation("Best model persisted | %s/%s → %s (ROC-AUC %.4f)",
                          dataset, cond, model_name, metadata['roc_auc'])
        records.append(record)

    report = pd.DataFrame(records)
    _save_report(experiment_dir, report)
    return report


def verify_best_models(experiment_dir: str) -> pd.DataFrame:
    """Reload every persisted best model and verify predictions + ROC-AUC.

    For each persisted model:
        1. reload ``best_model.pkl`` and the per-condition test features;
        2. recompute ``predict_proba[:, 1]``;
        3. compare with the stored ``{model}_proba`` column in
           ``predictions.csv`` (must match exactly);
        4. recompute ROC-AUC and compare with ``metrics.csv`` (tol 1e-6).

    Returns a report DataFrame; overall status is FAIL if any row fails.
    """
    root = os.path.join(experiment_dir, 'results', 'master', 'best_models')
    records: List[Dict[str, Any]] = []

    if not os.path.isdir(root):
        logger.warning("No best_models directory — verification skipped")
        return pd.DataFrame(records)

    for dataset in sorted(os.listdir(root)):
        ds_dir = os.path.join(root, dataset)
        if not os.path.isdir(ds_dir):
            continue
        for cond in sorted(os.listdir(ds_dir)):
            out_dir = os.path.join(ds_dir, cond)
            pkl_path = os.path.join(out_dir, 'best_model.pkl')
            meta_path = os.path.join(out_dir, 'best_model_metadata.json')
            record = {'dataset': dataset, 'condition': cond}
            if not os.path.isfile(pkl_path) or not os.path.isfile(meta_path):
                record.update(status=STATUS_SKIPPED, note='artifacts missing')
                records.append(record)
                continue

            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception as exc:
                record.update(status=STATUS_FAIL, note=f'metadata unreadable: {exc}')
                records.append(record)
                continue

            base_dir = _condition_dir(experiment_dir, dataset, cond == 'with_smote')
            record['model'] = meta.get('model')
            try:
                model = joblib.load(pkl_path)
                preds_df = pd.read_csv(os.path.join(base_dir, 'predictions.csv'))
                y_test = pd.Series(preds_df['y_test'].values, index=preds_df['customer_id'].astype(str))
                feats = pd.read_csv(
                    os.path.join(base_dir, 'test_features.csv'), index_col=0)
                feats = feats.reindex(preds_df['customer_id'].astype(str))
                proba_new = model.predict_proba(feats)[:, 1]
            except Exception as exc:
                record.update(status=STATUS_FAIL, note=f'reload failed: {exc}')
                records.append(record)
                continue

            stored_key = f"{meta.get('model')}_proba"
            stored_proba = preds_df[stored_key].values if stored_key in preds_df.columns else None
            proba_match = bool(
                stored_proba is not None
                and len(stored_proba) == len(proba_new)
                and np.allclose(stored_proba, proba_new, atol=AUC_TOLERANCE)
            )

            expected_auc = float(meta.get('roc_auc'))
            recomputed_auc = None
            auc_match = False
            try:
                recomputed_auc = float(roc_auc_score(y_test.values, proba_new))
                auc_match = (expected_auc is not None
                             and abs(recomputed_auc - expected_auc) <= AUC_TOLERANCE)
            except Exception as exc:
                record.update(status=STATUS_FAIL,
                              note=f'AUC recompute failed: {exc}')
                records.append(record)
                continue

            status = STATUS_PASS if (proba_match and auc_match) else STATUS_FAIL
            record.update(
                status=status,
                proba_match=proba_match,
                expected_roc_auc=expected_auc,
                recomputed_roc_auc=round(recomputed_auc, 6),
                auc_within_tolerance=auc_match,
                n_test=len(y_test),
                note='reload-verify PASS' if status == STATUS_PASS
                     else 'reload-verify FAIL (predictions or AUC differ)',
            )
            logger.validation("Model persistence verify | %s/%s → %s",
                              dataset, cond, status)
            records.append(record)

    report = pd.DataFrame(records)
    _save_verification_report(experiment_dir, report)
    return report


def overall_persistence_status(report: pd.DataFrame) -> str:
    """Overall model-persistence status (FAIL if any row fails)."""
    if report is None or report.empty:
        return STATUS_SKIPPED
    if (report['status'] == STATUS_FAIL).any():
        return STATUS_FAIL
    if (report['status'] == STATUS_PASS).any():
        return STATUS_PASS
    return STATUS_SKIPPED


def _save_report(experiment_dir: str, report: pd.DataFrame) -> str:
    master = ensure_dir(os.path.join(experiment_dir, 'results', 'master'))
    path = os.path.join(master, 'model_persistence_report.csv')
    report.to_csv(path, index=False)
    return path


def _save_verification_report(experiment_dir: str, report: pd.DataFrame) -> str:
    master = ensure_dir(os.path.join(experiment_dir, 'results', 'master'))
    path = os.path.join(master, 'model_persistence_verification.csv')
    report.to_csv(path, index=False)
    return path
