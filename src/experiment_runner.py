"""
Experiment runner for the Behavioral Churn Prediction Framework.

Orchestrates the full experiment matrix across all registered datasets,
SMOTE configurations, and models.  Produces publication-ready tables,
figures, and research artifacts.

This module does NOT implement ML logic.  It only calls the existing
framework and aggregates results.

Usage:
    from src.experiment_runner import run_all_experiments
    results = run_all_experiments()
"""
import datetime
import json
import os
import platform
import sys
import time
import traceback
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    FINAL_EXPERIMENT_DATASETS,
    FINAL_EXPERIMENT_MODELS,
    FRAMEWORK_VERSION,
    PROJECT_ROOT,
    RANDOM_SEED,
    SMOTE_CONDITIONS,
)
from src.utils import ensure_dir, get_logger, set_seed

logger = get_logger(__name__)
set_seed(RANDOM_SEED)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ── Default dataset list (Section 1 — exactly these 8) ──────────────
DEFAULT_DATASETS = list(FINAL_EXPERIMENT_DATASETS)

DEFAULT_MODELS = list(FINAL_EXPERIMENT_MODELS)

STATUS_SUCCESS = 'success'
STATUS_FAILED = 'failed'
STATUS_SKIPPED = 'skipped'


# ═════════════════════════════════════════════════════════════════════
# SYSTEM INFORMATION
# ═════════════════════════════════════════════════════════════════════

def collect_system_info() -> dict:
    info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'processor': platform.processor(),
        'machine': platform.machine(),
        'ram_gb': _get_ram_gb(),
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'framework_version': FRAMEWORK_VERSION,
        'random_seed': RANDOM_SEED,
    }
    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
        else:
            info['cuda_version'] = None
            info['gpu_name'] = None
    except ImportError:
        info['cuda_available'] = False
        info['cuda_version'] = None
        info['gpu_name'] = None

    try:
        import imblearn
        import lightgbm
        import shap
        import sklearn
        import xgboost
        info['library_versions'] = {
            'scikit-learn': sklearn.__version__,
            'xgboost': xgboost.__version__,
            'lightgbm': lightgbm.__version__,
            'shap': shap.__version__,
            'imbalanced-learn': imblearn.__version__,
            'pandas': pd.__version__,
            'numpy': np.__version__,
        }
    except Exception:
        info['library_versions'] = {}

    return info


def _get_ram_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal'):
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 1)
        except Exception:
            pass
    return 0.0


def export_system_info(output_dir: str) -> str:
    info = collect_system_info()
    path = os.path.join(output_dir, 'system_information.json')
    with open(path, 'w') as f:
        json.dump(info, f, indent=2, default=str)
    logger.info("System info exported: %s", path)
    return path


# ═════════════════════════════════════════════════════════════════════
# DATASET VALIDATION
# ═════════════════════════════════════════════════════════════════════

def validate_datasets(
    datasets: List[str],
    data_dirs: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    from src.datasets import get_dataset, list_datasets

    records = []
    all_registered = list_datasets()

    for name in datasets:
        record = {
            'dataset': name,
            'registered': name in all_registered,
            'required_columns': False,
            'missing_target': False,
            'duplicate_records': False,
            'invalid_timestamps': False,
            'empty_dataset': False,
            'feature_availability': False,
            'class_distribution_ok': False,
            'valid': False,
            'warnings': [],
            'errors': [],
        }

        if name not in all_registered:
            record['errors'].append(f"Dataset '{name}' not registered")
            records.append(record)
            continue

        try:
            data_dir = data_dirs.get(name) if data_dirs else None
            adapter = get_dataset(name, data_dir=data_dir)
            df = adapter.load_raw_data()

            if df.empty:
                record['empty_dataset'] = True
                record['errors'].append("Dataset is empty")
                records.append(record)
                continue

            df = adapter.preprocess(df)
            df = adapter.standardize_schema(df)

            required = ['customer_id', 'event_time']
            missing_cols = [c for c in required if c not in df.columns]
            if missing_cols:
                record['errors'].append(f"Missing required columns: {missing_cols}")
            else:
                record['required_columns'] = True

            if 'customer_id' in df.columns:
                n_dup = df['customer_id'].duplicated().sum()
                if n_dup > 0:
                    record['duplicate_records'] = True
                    record['warnings'].append(f"{n_dup} duplicate customer records")

            if 'event_time' in df.columns:
                n_nat = df['event_time'].isna().sum()
                if n_nat > 0:
                    record['invalid_timestamps'] = True
                    record['warnings'].append(f"{n_nat} invalid timestamps")

            n_customers = df['customer_id'].nunique() if 'customer_id' in df.columns else 0
            record['n_customers'] = n_customers

            if adapter.uses_native_churn_label:
                try:
                    labels = adapter.get_native_churn_labels(df, df['event_time'].max())
                    if 'churn' in labels.columns:
                        churn_rate = labels['churn'].mean()
                        record['class_distribution_ok'] = 0.01 < churn_rate < 0.99
                        record['churn_rate'] = float(churn_rate)
                except Exception:
                    record['warnings'].append("Could not validate class distribution")
            elif adapter.has_temporal_data:
                # Temporal inactivity-based datasets: actually construct the
                # churn labels at the pipeline's train cutoff and verify both
                # classes are present.  This catches zero-churn datasets (e.g.
                # a broken synthetic timeline) before they ever enter the
                # experiment matrix, instead of letting them surface as NaN
                # metrics after a long pipeline run.
                try:
                    from src.config import PREDICTION_WINDOW_DAYS, TRAIN_SPLIT_QUANTILE
                    from src.pipeline import create_churn_labels, get_train_test_cutoffs
                    window = adapter.churn_window_days or PREDICTION_WINDOW_DAYS
                    train_cutoff, _ = get_train_test_cutoffs(
                        df, TRAIN_SPLIT_QUANTILE, window,
                    )
                    labels = create_churn_labels(
                        df, train_cutoff, prediction_window_days=window,
                    )
                    churn_rate = float(labels['churn'].mean())
                    record['churn_rate'] = churn_rate
                    record['class_distribution_ok'] = 0.01 < churn_rate < 0.99
                    if not record['class_distribution_ok']:
                        record['errors'].append(
                            f"Churn construction yields (near-)single-class "
                            f"labels — train churn rate {churn_rate:.3%} is "
                            f"unusable for binary prediction"
                        )
                except Exception as exc:
                    record['class_distribution_ok'] = False
                    record['errors'].append(
                        f"Could not validate churn labels: {exc}"
                    )
            else:
                record['class_distribution_ok'] = True

            record['feature_availability'] = True
            record['valid'] = not record['errors']

        except Exception as exc:
            record['errors'].append(str(exc))

        records.append(record)

    report = pd.DataFrame(records)
    return report


# ═════════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORY MANAGEMENT
# ═════════════════════════════════════════════════════════════════════

def create_output_structure(
    base_dir: str,
    datasets: List[str],
    smote_configs: List[str],
    timestamp: str = None,
) -> str:
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    experiment_dir = os.path.join(base_dir, f'experiment_{timestamp}')

    # Isolated per-condition outputs (Section 30) live under
    # {experiment_dir}/results/{without_smote|with_smote}/{dataset}/
    for smote_label in smote_configs:
        for ds in datasets:
            ds_dir = os.path.join(experiment_dir, 'results', smote_label, ds)
            ensure_dir(ds_dir)

    ensure_dir(os.path.join(experiment_dir, 'results', 'master'))
    ensure_dir(os.path.join(experiment_dir, 'results', 'framework'))
    ensure_dir(os.path.join(experiment_dir, 'figures'))
    ensure_dir(os.path.join(experiment_dir, 'publication_tables'))
    ensure_dir(os.path.join(experiment_dir, 'publication_figures'))

    return experiment_dir


# ═════════════════════════════════════════════════════════════════════
# SINGLE EXPERIMENT EXECUTION
# ═════════════════════════════════════════════════════════════════════

def run_single_experiment(
    dataset: str,
    use_smote: bool,
    data_dir: Optional[str] = None,
    churn_window_override: Optional[int] = None,
    model_names: Optional[List[str]] = None,
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    start = time.time()
    status = STATUS_SUCCESS
    error_msg = ''
    pipeline_meta = {}

    try:
        # Stateless design (Section 8): SMOTE condition and model subset are
        # explicit parameters — the pipeline NEVER mutates global config.
        from src.pipeline import run_pipeline
        pipeline_meta = run_pipeline(
            dataset=dataset,
            sensitivity=False,
            churn_window_override=churn_window_override,
            data_dir=data_dir,
            use_smote=use_smote,
            model_names=model_names,
            results_dir=results_dir,
        )
    except Exception as exc:
        status = STATUS_FAILED
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Experiment failed [%s, SMOTE=%s]: %s", dataset, use_smote, exc)
        logger.debug(traceback.format_exc())

    elapsed = time.time() - start

    return {
        'dataset': dataset,
        'use_smote': use_smote,
        'status': status,
        'error': error_msg,
        'duration_seconds': elapsed,
        'pipeline_meta': pipeline_meta,
        'timestamp': datetime.datetime.utcnow().isoformat(),
    }


# ═════════════════════════════════════════════════════════════════════
# TEST-IDENTITY VALIDATION (Section 7)
# ═════════════════════════════════════════════════════════════════════

def validate_test_identity(
    exp_no_smote: Dict[str, Any],
    exp_with_smote: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify that SMOTE did not alter the test set (Section 7).

    Compares the test customer ids / labels fingerprints captured by the
    pipeline for the two conditions of the same dataset.  A mismatch means
    the experiment is invalid.
    """
    a = exp_no_smote.get('pipeline_meta') or {}
    b = exp_with_smote.get('pipeline_meta') or {}
    result = {
        'dataset': exp_no_smote.get('dataset'),
        'test_ids_match': None,
        'test_y_match': None,
        'valid': False,
        'note': '',
    }
    if exp_no_smote.get('status') != STATUS_SUCCESS or exp_with_smote.get('status') != STATUS_SUCCESS:
        result['note'] = 'skipped — one or both conditions failed'
        return result

    id_a = a.get('test_ids_hash')
    id_b = b.get('test_ids_hash')
    y_a = a.get('test_y_hash')
    y_b = b.get('test_y_hash')
    result['test_ids_match'] = (id_a == id_b)
    result['test_y_match'] = (y_a == y_b)
    result['valid'] = bool(result['test_ids_match'] and result['test_y_match'])
    result['note'] = (
        'PASS' if result['valid'] else
        'FAIL — test set differs between SMOTE conditions (invalid experiment)'
    )
    return result


# ═════════════════════════════════════════════════════════════════════
# RESULT COLLECTION
# ═════════════════════════════════════════════════════════════════════

def collect_experiment_results(
    experiment_dir: str,
    dataset: str,
    use_smote: bool,
) -> Dict[str, Any]:
    """Collect the isolated per-experiment artifacts (Section 30).

    Reads {results_dir}/{with|without}_smote/{dataset}/metrics.csv written
    by save_experiment_artifacts inside run_pipeline.
    """
    results_dir = os.path.join(experiment_dir, 'results')
    smote_label = 'with_smote' if use_smote else 'without_smote'
    base = os.path.join(results_dir, smote_label, dataset)

    results = {}

    metrics_path = os.path.join(base, 'metrics.csv')
    if os.path.isfile(metrics_path):
        try:
            results['metrics'] = pd.read_csv(metrics_path)
        except Exception:
            pass

    predictions_path = os.path.join(base, 'predictions.csv')
    if os.path.isfile(predictions_path):
        try:
            results['predictions'] = pd.read_csv(predictions_path)
        except Exception:
            pass

    meta_path = os.path.join(base, 'experiment_metadata.json')
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                results['metadata'] = json.load(f)
        except Exception:
            pass

    return results


# ═════════════════════════════════════════════════════════════════════
# MASTER TABLE GENERATION
# ═════════════════════════════════════════════════════════════════════

def generate_all_results(
    all_experiments: List[Dict],
    experiment_dir: str,
) -> pd.DataFrame:
    rows = []
    for exp in all_experiments:
        if exp['status'] != STATUS_SUCCESS:
            rows.append({
                'dataset': exp['dataset'],
                'model': 'FAILED',
                'smote': 'Yes' if exp['use_smote'] else 'No',
                'status': exp['status'],
                'error': exp['error'],
                'duration_seconds': exp['duration_seconds'],
            })
            continue

        results = exp.get('results', {})
        metrics_df = results.get('metrics')
        if metrics_df is None or metrics_df.empty:
            continue

        smote_str = 'Yes' if exp['use_smote'] else 'No'
        for _, row in metrics_df.iterrows():
            rows.append({
                'dataset': exp['dataset'],
                'model': row.get('model', 'unknown'),
                'smote': smote_str,
                'accuracy': row.get('accuracy'),
                'precision': row.get('precision'),
                'recall': row.get('recall'),
                'f1': row.get('f1'),
                'roc_auc': row.get('roc_auc'),
                'pr_auc': row.get('avg_precision'),
                'balanced_accuracy': row.get('balanced_accuracy'),
                'mcc': row.get('mcc'),
                'brier_score': row.get('brier_score'),
                'calibration_error': row.get('calibration_error'),
                'training_time': row.get('training_time'),
                'inference_time': row.get('inference_time'),
                'tn': row.get('tn'),
                'fp': row.get('fp'),
                'fn': row.get('fn'),
                'tp': row.get('tp'),
                'status': STATUS_SUCCESS,
                'duration_seconds': exp['duration_seconds'],
                'timestamp': exp['timestamp'],
            })

    df = pd.DataFrame(rows)

    master_path = os.path.join(experiment_dir, 'results', 'master', 'all_results.csv')
    df.to_csv(master_path, index=False)

    try:
        xlsx_path = os.path.join(experiment_dir, 'results', 'master', 'master_results.xlsx')
        df.to_excel(xlsx_path, index=False, engine='openpyxl')
    except Exception:
        pass

    return df


def generate_dataset_summary(all_results: pd.DataFrame, experiment_dir: str) -> pd.DataFrame:
    if all_results.empty:
        return pd.DataFrame()

    numeric_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc',
                    'pr_auc', 'balanced_accuracy', 'mcc', 'brier_score',
                    'calibration_error', 'training_time', 'inference_time']

    summary_rows = []
    for dataset in all_results['dataset'].unique():
        ds_data = all_results[all_results['dataset'] == dataset]
        for smote_val in ds_data['smote'].unique():
            sub = ds_data[ds_data['smote'] == smote_val]
            non_baseline = sub[~sub['model'].str.contains('baseline', na=False)]
            if non_baseline.empty:
                continue
            row = {'dataset': dataset, 'smote': smote_val}
            for col in numeric_cols:
                if col in non_baseline.columns:
                    vals = non_baseline[col].dropna()
                    if len(vals) > 0:
                        row[f'{col}_mean'] = vals.mean()
                        row[f'{col}_std'] = vals.std()
            best_idx = None
            if 'roc_auc' in non_baseline.columns:
                auc_vals = non_baseline['roc_auc'].dropna()
                if len(auc_vals) > 0:
                    best_idx = auc_vals.idxmax()
            if best_idx is not None and pd.notna(best_idx):
                row['best_model'] = non_baseline.loc[best_idx, 'model']
                row['best_roc_auc'] = non_baseline.loc[best_idx, 'roc_auc']
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    path = os.path.join(experiment_dir, 'results', 'master', 'dataset_summary.csv')
    summary.to_csv(path, index=False)
    return summary


def generate_model_summary(all_results: pd.DataFrame, experiment_dir: str) -> pd.DataFrame:
    if all_results.empty:
        return pd.DataFrame()

    numeric_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc',
                    'pr_auc', 'balanced_accuracy', 'mcc', 'brier_score',
                    'calibration_error', 'training_time', 'inference_time']

    summary_rows = []
    for model in all_results['model'].unique():
        mod_data = all_results[all_results['model'] == model]
        for smote_val in mod_data['smote'].unique():
            sub = mod_data[mod_data['smote'] == smote_val]
            row = {'model': model, 'smote': smote_val, 'n_datasets': len(sub)}
            for col in numeric_cols:
                if col in sub.columns:
                    vals = sub[col].dropna()
                    if len(vals) > 0:
                        row[f'{col}_mean'] = vals.mean()
                        row[f'{col}_std'] = vals.std()
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    path = os.path.join(experiment_dir, 'results', 'master', 'model_summary.csv')
    summary.to_csv(path, index=False)
    return summary


def generate_smote_comparison(all_results: pd.DataFrame, experiment_dir: str) -> pd.DataFrame:
    if all_results.empty:
        return pd.DataFrame()

    non_baseline = all_results[~all_results['model'].str.contains('baseline', na=False)].copy()
    if non_baseline.empty:
        return pd.DataFrame()

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    metrics = [m for m in metrics if m in non_baseline.columns]
    rows = []

    for dataset in non_baseline['dataset'].unique():
        ds = non_baseline[non_baseline['dataset'] == dataset]
        for model in ds['model'].unique():
            mod = ds[ds['model'] == model]
            row = {'dataset': dataset, 'model': model}
            for m in metrics:
                no_smote = mod[mod['smote'] == 'No'][m].mean()
                with_smote = mod[mod['smote'] == 'Yes'][m].mean()
                row[f'{m}_no_smote'] = no_smote
                row[f'{m}_with_smote'] = with_smote
                row[f'{m}_delta'] = with_smote - no_smote
            rows.append(row)

    comparison = pd.DataFrame(rows)
    path = os.path.join(experiment_dir, 'results', 'master', 'smote_comparison.csv')
    comparison.to_csv(path, index=False)
    return comparison


# ═════════════════════════════════════════════════════════════════════
# PUBLICATION TABLES
# ═════════════════════════════════════════════════════════════════════

def generate_publication_tables(
    all_results: pd.DataFrame,
    experiment_dir: str,
) -> List[str]:
    tables_dir = os.path.join(experiment_dir, 'publication_tables')
    ensure_dir(tables_dir)
    exported = []

    non_baseline = all_results[
        ~all_results['model'].str.contains('baseline', na=False)
    ].copy() if not all_results.empty else pd.DataFrame()

    if non_baseline.empty:
        return exported

    # Best model per dataset
    best_rows = []
    datasets_with_valid_auc = 0
    for ds in non_baseline['dataset'].unique():
        ds_data = non_baseline[non_baseline['dataset'] == ds]
        if 'roc_auc' in ds_data.columns:
            auc_vals = ds_data['roc_auc'].dropna()
            if len(auc_vals) == 0:
                # All ROC-AUC values are NaN (e.g. every model failed to
                # produce a usable ROC curve).  Never call .idxmax() on an
                # all-NaN column — it raises (KeyError: nan / ValueError).
                # The dataset is surfaced in the dataset-level audit report
                # instead of being silently dropped.
                logger.warning(
                    "Publication tables: dataset %s has no usable ROC-AUC — "
                    "omitted from best-model table but reported as failed",
                    ds,
                )
                continue
            best = ds_data.loc[auc_vals.idxmax()]
            datasets_with_valid_auc += 1
            best_rows.append({
                'Dataset': ds,
                'Best Model': best.get('model', ''),
                'ROC-AUC': best.get('roc_auc', ''),
                'F1': best.get('f1', ''),
                'SMOTE': best.get('smote', ''),
            })
    best_df = pd.DataFrame(best_rows)
    _export_table(best_df, os.path.join(tables_dir, 'best_model_per_dataset'))

    # Average performance by model
    agg_dict = {}
    for col in ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']:
        if col in non_baseline.columns:
            agg_dict[col] = ['mean', 'std']
    if agg_dict:
        model_perf = non_baseline.groupby('model').agg(agg_dict).round(4)
        model_perf.columns = ['_'.join(col) for col in model_perf.columns]
        model_perf = model_perf.reset_index()
        _export_table(model_perf, os.path.join(tables_dir, 'model_performance'))

    # Average performance by dataset
    agg_dict2 = {}
    for col in ['accuracy', 'f1', 'roc_auc']:
        if col in non_baseline.columns:
            agg_dict2[col] = ['mean', 'std']
    if agg_dict2:
        ds_perf = non_baseline.groupby('dataset').agg(agg_dict2).round(4)
        ds_perf.columns = ['_'.join(col) for col in ds_perf.columns]
        ds_perf = ds_perf.reset_index()
        _export_table(ds_perf, os.path.join(tables_dir, 'dataset_performance'))

    # Overall ranking
    ranking_cols = {}
    if 'roc_auc' in non_baseline.columns:
        ranking_cols['roc_auc'] = 'mean'
    if 'f1' in non_baseline.columns:
        ranking_cols['f1'] = 'mean'
    if ranking_cols:
        sort_col = 'roc_auc' if 'roc_auc' in ranking_cols else list(ranking_cols.keys())[0]
        ranking = non_baseline.groupby('model').agg(ranking_cols).round(4).sort_values(sort_col, ascending=False).reset_index()
        ranking['rank'] = range(1, len(ranking) + 1)
        _export_table(ranking, os.path.join(tables_dir, 'overall_ranking'))

    return exported


def _export_table(df: pd.DataFrame, base_path: str) -> None:
    df.to_csv(f"{base_path}.csv", index=False)
    try:
        df.to_excel(f"{base_path}.xlsx", index=False, engine='openpyxl')
    except Exception:
        pass
    try:
        df.to_latex(f"{base_path}.tex", index=False, escape=True)
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════
# PUBLICATION FIGURES
# ═════════════════════════════════════════════════════════════════════

def generate_publication_figures(
    all_results: pd.DataFrame,
    experiment_dir: str,
) -> List[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir = os.path.join(experiment_dir, 'publication_figures')
    ensure_dir(fig_dir)
    exported = []

    non_baseline = all_results[
        ~all_results['model'].str.contains('baseline', na=False)
    ].copy() if not all_results.empty else pd.DataFrame()

    if non_baseline.empty:
        return exported

    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    palette = sns.color_palette("colorblind")

    # Model comparison heatmap
    try:
        pivot = non_baseline.pivot_table(
            index='model', columns='dataset', values='roc_auc', aggfunc='mean',
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                    linewidths=0.5, vmin=0.4, vmax=1.0)
        ax.set_title('Model Comparison — Average ROC-AUC')
        ax.set_ylabel('Model')
        ax.set_xlabel('Dataset')
        fig.tight_layout()
        for fmt in ['png', 'pdf']:
            fig.savefig(os.path.join(fig_dir, f'model_comparison_heatmap.{fmt}'),
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
        exported.append('model_comparison_heatmap')
    except Exception as exc:
        logger.warning("Model comparison heatmap failed: %s", exc)

    # Dataset comparison heatmap
    try:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        available = [m for m in metrics if m in non_baseline.columns]
        pivot = non_baseline.pivot_table(
            index='dataset', columns=available, values=available,
            aggfunc='mean',
        )
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                        linewidths=0.5)
            ax.set_title('Dataset Comparison — Average Metrics')
            fig.tight_layout()
            for fmt in ['png', 'pdf']:
                fig.savefig(os.path.join(fig_dir, f'dataset_comparison_heatmap.{fmt}'),
                            dpi=300, bbox_inches='tight')
            plt.close(fig)
            exported.append('dataset_comparison_heatmap')
    except Exception as exc:
        logger.warning("Dataset comparison heatmap failed: %s", exc)

    # Overall ranking bar chart
    try:
        ranking = non_baseline.groupby('model')['roc_auc'].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ranking.plot(kind='barh', ax=ax, color=palette[:len(ranking)])
        ax.set_xlabel('Average ROC-AUC')
        ax.set_title('Model Ranking by Average ROC-AUC')
        ax.set_xlim(0.4, 1.0)
        fig.tight_layout()
        for fmt in ['png', 'pdf']:
            fig.savefig(os.path.join(fig_dir, f'overall_ranking.{fmt}'),
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
        exported.append('overall_ranking')
    except Exception as exc:
        logger.warning("Overall ranking failed: %s", exc)

    # SMOTE comparison
    try:
        smote_no = non_baseline[non_baseline['smote'] == 'No'].groupby('model')['roc_auc'].mean()
        smote_yes = non_baseline[non_baseline['smote'] == 'Yes'].groupby('model')['roc_auc'].mean()
        common_models = smote_no.index.intersection(smote_yes.index)
        if len(common_models) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(common_models))
            width = 0.35
            ax.bar(x - width/2, smote_no[common_models], width, label='Without SMOTE', color=palette[0])
            ax.bar(x + width/2, smote_yes[common_models], width, label='With SMOTE', color=palette[3])
            ax.set_xticks(x)
            ax.set_xticklabels(common_models, rotation=45, ha='right')
            ax.set_ylabel('Average ROC-AUC')
            ax.set_title('SMOTE Impact on Model Performance')
            ax.legend()
            fig.tight_layout()
            for fmt in ['png', 'pdf']:
                fig.savefig(os.path.join(fig_dir, f'smote_comparison.{fmt}'),
                            dpi=300, bbox_inches='tight')
            plt.close(fig)
            exported.append('smote_comparison')
    except Exception as exc:
        logger.warning("SMOTE comparison failed: %s", exc)

    # Metric distribution boxplots
    try:
        plot_metrics = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
        available = [m for m in plot_metrics if m in non_baseline.columns]
        if available:
            fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5))
            if len(available) == 1:
                axes = [axes]
            for ax, metric in zip(axes, available):
                non_baseline.boxplot(column=metric, by='model', ax=ax, grid=False)
                ax.set_title(metric.upper())
                ax.set_xlabel('')
                ax.tick_params(axis='x', rotation=45)
            fig.suptitle('Metric Distribution Across Datasets', fontsize=14)
            fig.tight_layout()
            for fmt in ['png', 'pdf']:
                fig.savefig(os.path.join(fig_dir, f'metric_distribution.{fmt}'),
                            dpi=300, bbox_inches='tight')
            plt.close(fig)
            exported.append('metric_distribution')
    except Exception as exc:
        logger.warning("Metric distribution failed: %s", exc)

    return exported


# ═════════════════════════════════════════════════════════════════════
# DATASET CHARACTERISTICS
# ═════════════════════════════════════════════════════════════════════

def generate_dataset_characteristics(
    datasets: List[str],
    all_results: pd.DataFrame,
    experiment_dir: str,
    data_dirs: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    from src.datasets import get_dataset

    records = []
    for name in datasets:
        record = {'dataset': name}
        try:
            data_dir = data_dirs.get(name) if data_dirs else None
            adapter = get_dataset(name, data_dir=data_dir)
            df = adapter.load_raw_data()
            df = adapter.preprocess(df)
            df = adapter.standardize_schema(df)

            if 'customer_id' in df.columns:
                record['n_customers'] = df['customer_id'].nunique()
            record['n_records'] = len(df)

            if 'event_type' in df.columns:
                record['n_event_types'] = df['event_type'].nunique()

            if 'product_id' in df.columns:
                record['unique_products'] = df['product_id'].nunique()

            if 'customer_id' in df.columns and 'event_time' in df.columns:
                cust_events = df.groupby('customer_id').size()
                record['avg_events_per_customer'] = cust_events.mean()
                record['median_events_per_customer'] = cust_events.median()
                repeat = (cust_events > 1).sum()
                total = len(cust_events)
                record['repeat_customer_ratio'] = repeat / total if total > 0 else 0

            if 'event_time' in df.columns:
                record['date_range_start'] = str(df['event_time'].min())
                record['date_range_end'] = str(df['event_time'].max())
                span = (df['event_time'].max() - df['event_time'].min()).days
                record['time_span_days'] = span

            if adapter.uses_native_churn_label:
                try:
                    labels = adapter.get_native_churn_labels(df, df['event_time'].max())
                    record['churn_rate'] = labels['churn'].mean()
                except Exception:
                    pass

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            record['n_numerical_features'] = len(numeric_cols)
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            record['n_categorical_features'] = len(cat_cols)

            missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            record['missing_value_pct'] = missing_pct

            if not all_results.empty and name in all_results['dataset'].values:
                ds_results = all_results[all_results['dataset'] == name]
                non_bl = ds_results[~ds_results['model'].str.contains('baseline', na=False)]
                for col in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'brier_score']:
                    if col in non_bl.columns:
                        vals = non_bl[col].dropna()
                        if len(vals) > 0:
                            record[f'avg_{col}'] = vals.mean()

        except Exception as exc:
            record['error'] = str(exc)

        records.append(record)

    chars = pd.DataFrame(records)
    path = os.path.join(experiment_dir, 'results', 'master', 'dataset_characteristics.csv')
    chars.to_csv(path, index=False)
    return chars


# ═════════════════════════════════════════════════════════════════════
# RESEARCH DATA EXPORT
# ═════════════════════════════════════════════════════════════════════

def generate_research_summary(
    all_experiments: List[Dict],
    all_results: pd.DataFrame,
    experiment_dir: str,
) -> pd.DataFrame:
    if all_results.empty:
        return pd.DataFrame()

    research = all_results.copy()
    path = os.path.join(experiment_dir, 'results', 'master', 'research_summary.csv')
    research.to_csv(path, index=False)
    return research


# ═════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY EXPORT
# ═════════════════════════════════════════════════════════════════════

def export_reproducibility(experiment_dir: str) -> None:
    info = collect_system_info()
    fw_config = {
        'framework_version': FRAMEWORK_VERSION,
        'random_seed': RANDOM_SEED,
        'timestamp': info['timestamp'],
        'python_version': info['python_version'],
        'library_versions': info.get('library_versions', {}),
    }
    path = os.path.join(experiment_dir, 'results', 'framework', 'framework_configuration.json')
    with open(path, 'w') as f:
        json.dump(fw_config, f, indent=2, default=str)

    env_path = os.path.join(experiment_dir, 'results', 'framework', 'environment.json')
    with open(env_path, 'w') as f:
        json.dump(info, f, indent=2, default=str)

    try:
        import pkg_resources
        req_path = os.path.join(experiment_dir, 'results', 'framework', 'requirements.txt')
        with open(req_path, 'w') as f:
            for dist in pkg_resources.working_set:
                f.write(f"{dist.project_name}=={dist.version}\n")
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════
# POST-EXPERIMENT PROCESSING (final build phase)
# ═════════════════════════════════════════════════════════════════════

def compute_kkbox_status(data_dirs: Optional[Dict[str, str]]) -> str:
    """Resolve the honest KKBox validation status for the audit."""
    try:
        from src.kkbox.validation import check_data_availability
    except Exception as exc:
        return f'UNKNOWN — kkbox modules unavailable ({exc})'
    data_dir = data_dirs.get('kkbox') if data_dirs else None
    if not data_dir or not os.path.isdir(data_dir):
        return 'PENDING — KKBox data not present'
    try:
        availability = check_data_availability(data_dir)
        if not availability.get('transactions'):
            return 'PENDING — KKBox transactions not present'
        if not availability.get('train'):
            return 'PENDING — official labels absent (train/train_v2 missing)'
        try:
            from src.kkbox.validation import run_kkbox_validation
            report = run_kkbox_validation(data_dir)
            return f"VALIDATED ({report.get('status', 'UNKNOWN')})"
        except Exception as exc:
            return f'UNKNOWN — validation attempt failed ({exc})'
    except Exception as exc:
        return f'UNKNOWN — availability check failed ({exc})'


def run_post_processing(
    experiment_dir: str,
    all_results: pd.DataFrame,
    identity_results: List[Dict[str, Any]],
    valid_datasets: List[str],
    data_dirs: Optional[Dict[str, str]] = None,
    all_experiments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the post-experiment stages and record their outcomes.

    Each stage is isolated so a failure in one never aborts the runner.
    Stages (in order):
        1. statistical comparison (Friedman / Nemenyi / Wilcoxon + CD)
        2. framework quadrant analysis
        3. best-model persistence + reload verification
        4. publication figures (supplementary + main)
        5. integrity audit + completion report (csv/json/txt)
    """
    stages: Dict[str, Any] = {}

    # 1. Statistical comparison
    try:
        from src.statistical_comparison import run_statistical_comparison
        stats = run_statistical_comparison(experiment_dir, all_results)
        stages['statistical_comparison'] = (
            'DONE' if (stats.get('friedman') or stats.get('wilcoxon') is not None)
            else 'SKIPPED (no data)')
    except Exception as exc:
        stages['statistical_comparison'] = f'FAILED ({exc})'
        logger.warning("Statistical comparison failed: %s", exc)

    # 2. Framework quadrant
    try:
        from src.framework_analysis import run_framework_analysis
        fw = run_framework_analysis(experiment_dir, all_results)
        stages['framework_quadrant'] = (
            'DONE' if fw else 'SKIPPED (no characteristics)')
    except Exception as exc:
        stages['framework_quadrant'] = f'FAILED ({exc})'
        logger.warning("Framework quadrant failed: %s", exc)

    # 3. Model persistence + verification
    try:
        from src.model_persistence import (
            overall_persistence_status,
            persist_best_models,
            verify_best_models,
        )
        pers_report = persist_best_models(experiment_dir, all_results)
        ver_report = verify_best_models(experiment_dir)
        if not ver_report.empty:
            stages['model_persistence'] = overall_persistence_status(ver_report)
        else:
            stages['model_persistence'] = overall_persistence_status(pers_report)
    except Exception as exc:
        stages['model_persistence'] = f'FAILED ({exc})'
        logger.warning("Model persistence failed: %s", exc)

    # 4. Publication figures
    try:
        from src.publication_figures import generate_all_figures
        figs = generate_all_figures(experiment_dir, all_results)
        stages['figures_main'] = len(figs['main'])
        stages['figures_supplementary'] = len(figs['supplementary'])
    except Exception as exc:
        stages['figures'] = f'FAILED ({exc})'
        logger.warning("Publication figures failed: %s", exc)

    # 5. Integrity audit + completion report
    # The KKBox status check is only included when a 'kkbox' data directory
    # is actually supplied; otherwise the audit omits the KKBox row entirely
    # (kkbox is not part of the default experiment matrix).
    kkbox_status = None
    if data_dirs and 'kkbox' in data_dirs:
        kkbox_status = compute_kkbox_status(data_dirs)
    stages['kkbox_status'] = kkbox_status
    try:
        from src.final_audit import audit_results, write_completion_report
        audit = audit_results(
            experiment_dir, all_results, identity_results,
            kkbox_status=kkbox_status)
        if all_experiments:
            successful = sum(1 for e in all_experiments
                             if e.get('status') == STATUS_SUCCESS)
            failed = sum(1 for e in all_experiments
                         if e.get('status') != STATUS_SUCCESS)
        else:
            successful = len(all_results)
            failed = 0
        write_completion_report(
            experiment_dir, audit, all_results,
            extras={'successful': successful, 'failed': failed,
                    'kkbox_status': kkbox_status,
                    'statistical': stages.get('statistical_comparison'),
                    'framework': stages.get('framework_quadrant'),
                    'model_persistence': stages.get('model_persistence'),
                    'figures': stages.get('figures_main', 0) + stages.get(
                        'figures_supplementary', 0)})
        stages['integrity_audit'] = final_audit_overall(audit)
        stages['completion_report'] = 'DONE'
    except Exception as exc:
        stages['integrity_audit'] = f'FAILED ({exc})'
        logger.warning("Integrity audit failed: %s", exc)

    return stages


def final_audit_overall(audit: pd.DataFrame) -> str:
    from src.final_audit import overall_audit_status
    return overall_audit_status(audit)


# ═════════════════════════════════════════════════════════════════════
# COMPLETION REPORT
# ═════════════════════════════════════════════════════════════════════

def generate_completion_report(
    all_experiments: List[Dict],
    experiment_dir: str,
    publication_tables: List[str],
    publication_figures: List[str],
    start_time: float,
) -> None:
    total = len(all_experiments)
    successful = sum(1 for e in all_experiments if e['status'] == STATUS_SUCCESS)
    failed = total - successful
    datasets_done = len(set(e['dataset'] for e in all_experiments if e['status'] == STATUS_SUCCESS))
    avg_runtime = np.mean([e['duration_seconds'] for e in all_experiments]) if all_experiments else 0
    total_runtime = time.time() - start_time

    report = f"""
{'='*70}
BEHAVIORAL CHURN PREDICTION FRAMEWORK — EXPERIMENT COMPLETE
{'='*70}

Total Experiments:           {total}
Successful Experiments:     {successful}
Failed Experiments:         {failed}
Datasets Completed:         {datasets_done} / {len(DEFAULT_DATASETS)}
Average Experiment Runtime: {avg_runtime:.1f}s
Total Runtime:              {total_runtime:.1f}s

Publication Tables:         {len(publication_tables)}
Publication Figures:        {len(publication_figures)}

Output Directory:           {experiment_dir}
{'='*70}

Generated Files:
"""
    for root, dirs, files in os.walk(experiment_dir):
        level = root.replace(experiment_dir, '').count(os.sep)
        indent = '  ' * level
        basename = os.path.basename(root)
        report += f"{indent}{basename}/\n"
        subindent = '  ' * (level + 1)
        for file in sorted(files)[:10]:
            report += f"{subindent}{file}\n"
        if len(files) > 10:
            report += f"{subindent}... and {len(files) - 10} more files\n"

    if failed > 0:
        report += "\nFailed Experiments:\n"
        for e in all_experiments:
            if e['status'] == STATUS_FAILED:
                report += f"  - {e['dataset']} (SMOTE={'Yes' if e['use_smote'] else 'No'}): {e['error']}\n"

    report_path = os.path.join(experiment_dir, 'COMPLETION_REPORT.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    print(report)


# ═════════════════════════════════════════════════════════════════════
# FAILURE REPORT (Section 32)
# ═════════════════════════════════════════════════════════════════════

def generate_failure_report(
    all_experiments: List[Dict],
    identity_results: List[Dict[str, Any]],
    experiment_dir: str,
) -> pd.DataFrame:
    """Write results/{master}/failure_report.csv (Section 32).

    Every failed experiment and every failed integrity (test-identity)
    check is recorded with its traceback message.
    """
    rows = []
    for exp in all_experiments:
        if exp['status'] == STATUS_FAILED:
            rows.append({
                'dataset': exp['dataset'],
                'condition': 'with_smote' if exp['use_smote'] else 'without_smote',
                'check': 'experiment',
                'status': STATUS_FAILED,
                'error': exp['error'],
            })

    for check in identity_results:
        if not check['valid']:
            rows.append({
                'dataset': check['dataset'],
                'condition': 'both',
                'check': 'test_identity',
                'status': STATUS_FAILED,
                'error': check['note'],
            })

    report = pd.DataFrame(rows, columns=[
        'dataset', 'condition', 'check', 'status', 'error',
    ])

    master_dir = ensure_dir(os.path.join(experiment_dir, 'results', 'master'))
    path = os.path.join(master_dir, 'failure_report.csv')
    report.to_csv(path, index=False)

    if len(report) > 0:
        logger.warning("Failure report: %d failure(s) written to %s",
                       len(report), path)
    else:
        logger.validation("Failure report: no failures — %s", path)
    return report


# ═════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

def _resolve_output_dir(
    output_dir: Optional[str],
    smoke_test: bool,
    now: Optional[datetime.datetime] = None,
) -> str:
    """Resolve the output directory, isolating smoke-test runs.

    Smoke-test runs are redirected to ``<output>/smoke_test_<timestamp>`` so
    they can never contaminate the final results directory.
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, 'outputs')
    if smoke_test:
        now = now or datetime.datetime.now()
        output_dir = os.path.join(
            output_dir, f'smoke_test_{now.strftime("%Y%m%d_%H%M%S")}')
    return output_dir


def run_all_experiments(
    datasets: Optional[List[str]] = None,
    data_dirs: Optional[Dict[str, str]] = None,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    churn_window_override: Optional[int] = None,
    model_names: Optional[List[str]] = None,
    skip_validation: bool = False,
    smoke_test: bool = False,
) -> str:
    start_time = time.time()

    if datasets is None:
        datasets = list(DEFAULT_DATASETS)

    if model_names is None:
        model_names = list(DEFAULT_MODELS)

    # ── SMOKE_TEST mode ──────────────────────────────────────────────
    # When active (env SMOKE_TEST=1/true/yes, or smoke_test=True), all
    # outputs are isolated into a dedicated smoke-test output directory so
    # the final results can never be contaminated.
    smoke_test = smoke_test or os.environ.get('SMOKE_TEST', 'false').lower() \
        in ('1', 'true', 'yes')

    output_dir = _resolve_output_dir(output_dir, smoke_test)

    smote_configs = list(SMOTE_CONDITIONS)

    logger.info("=" * 70)
    logger.info("BEHAVIORAL CHURN PREDICTION FRAMEWORK — Experiment Runner")
    logger.info("=" * 70)
    logger.info("Datasets: %s", datasets)
    logger.info("Models: %s", model_names)
    logger.info("SMOTE configs: %s", smote_configs)
    logger.info("Random seed: %d", RANDOM_SEED)

    # Ensure output directory exists before anything writes to it
    ensure_dir(output_dir)

    # Export system info
    export_system_info(output_dir)

    # Validate datasets
    logger.info("── Validating datasets ──")
    validation_report = validate_datasets(datasets, data_dirs)
    valid_path = os.path.join(output_dir, 'dataset_validation_report.csv')
    validation_report.to_csv(valid_path, index=False)

    valid_datasets = validation_report[validation_report['valid']]['dataset'].tolist()
    invalid_datasets = validation_report[~validation_report['valid']]['dataset'].tolist()

    if invalid_datasets and not skip_validation:
        logger.warning("Skipping invalid datasets: %s", invalid_datasets)

    if not valid_datasets:
        logger.error("No valid datasets found. Aborting.")
        return output_dir

    # Create output structure
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = create_output_structure(output_dir, valid_datasets, smote_configs, timestamp)

    logger.info("Experiment directory: %s", experiment_dir)

    if smoke_test:
        logger.warning("SMOKE_TEST mode active — outputs isolated to %s",
                       experiment_dir)
        with open(os.path.join(experiment_dir, 'SMOKE_TEST.txt'), 'w') as f:
            f.write('This experiment run was executed in SMOKE_TEST mode and\n'
                    'is isolated from the final results directory.\n')

    # Run experiment matrix
    all_experiments = []
    total_experiments = len(valid_datasets) * len(smote_configs)

    for i, dataset in enumerate(valid_datasets):
        for j, smote_label in enumerate(smote_configs):
            use_smote = (smote_label == 'with_smote')
            exp_num = i * len(smote_configs) + j + 1

            logger.info(
                "── Experiment %d/%d: %s (SMOTE=%s) ──",
                exp_num, total_experiments, dataset, use_smote,
            )

            data_dir = data_dirs.get(dataset) if data_dirs else None
            result = run_single_experiment(
                dataset=dataset,
                use_smote=use_smote,
                data_dir=data_dir,
                churn_window_override=churn_window_override,
                model_names=model_names,
                results_dir=os.path.join(experiment_dir, 'results'),
            )

            result['results'] = collect_experiment_results(
                experiment_dir, dataset, use_smote,
            )

            all_experiments.append(result)

            status_icon = "✓" if result['status'] == STATUS_SUCCESS else "✗"
            logger.info(
                "  %s %s completed in %.1fs",
                status_icon, dataset, result['duration_seconds'],
            )

    # ── Test-identity validation (Section 7) ─────────────────────────
    logger.info("── Validating test-set identity across SMOTE conditions ──")
    identity_results: List[Dict[str, Any]] = []
    for dataset in valid_datasets:
        no_smote = next(
            (e for e in all_experiments
             if e['dataset'] == dataset and not e['use_smote']), None)
        with_smote = next(
            (e for e in all_experiments
             if e['dataset'] == dataset and e['use_smote']), None)
        if no_smote is None or with_smote is None:
            continue
        check = validate_test_identity(no_smote, with_smote)
        identity_results.append(check)
        logger.validation("  %s test-identity: %s", dataset, check['note'])

    # Generate master outputs
    logger.info("── Generating master outputs ──")
    all_results = generate_all_results(all_experiments, experiment_dir)
    generate_dataset_summary(all_results, experiment_dir)
    generate_model_summary(all_results, experiment_dir)
    generate_smote_comparison(all_results, experiment_dir)

    # Failure report (Section 32)
    generate_failure_report(all_experiments, identity_results, experiment_dir)

    # Generate publication tables
    logger.info("── Generating publication tables ──")
    pub_tables = generate_publication_tables(all_results, experiment_dir)

    # Generate publication figures
    logger.info("── Generating publication figures ──")
    pub_figures = generate_publication_figures(all_results, experiment_dir)

    # Dataset characteristics
    logger.info("── Generating dataset characteristics ──")
    generate_dataset_characteristics(valid_datasets, all_results, experiment_dir, data_dirs)

    # Post-experiment stages (statistics, framework quadrant, persistence,
    # publication figures, integrity audit, completion report)
    logger.info("── Post-experiment processing ──")
    post_processing = run_post_processing(
        experiment_dir=experiment_dir,
        all_results=all_results,
        identity_results=identity_results,
        valid_datasets=valid_datasets,
        data_dirs=data_dirs,
        all_experiments=all_experiments,
    )
    for stage, value in post_processing.items():
        logger.validation("  Post-processing %-28s %s", stage, value)

    # Research summary
    logger.info("── Generating research summary ──")
    generate_research_summary(all_experiments, all_results, experiment_dir)

    # Reproducibility
    export_reproducibility(experiment_dir)

    # Export experiment log
    log_df = pd.DataFrame([{
        'dataset': e['dataset'],
        'model': e.get('pipeline_meta', {}).get('best_model', ''),
        'smote': e['use_smote'],
        'status': e['status'],
        'duration_seconds': e['duration_seconds'],
        'timestamp': e['timestamp'],
    } for e in all_experiments])
    log_path = os.path.join(experiment_dir, 'results', 'master', 'experiment_log.csv')
    log_df.to_csv(log_path, index=False)

    # Completion report
    generate_completion_report(
        all_experiments, experiment_dir, pub_tables, pub_figures, start_time,
    )

    return experiment_dir
