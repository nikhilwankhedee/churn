"""
Sensitivity analysis for churn window thresholds.

Runs the pipeline with multiple churn window definitions per dataset to
demonstrate directional stability of findings.  Results are exported to:

    results/sensitivity_analysis/{dataset}_sensitivity.csv

This module is optional and invoked separately from the main pipeline.
"""
import os
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from src.config import (
    RESULTS_DIR, RANDOM_SEED, SENSITIVITY_RESULTS_DIR,
    SHAP_SAMPLE_SIZE, TRAIN_SPLIT_QUANTILE,
)
from src.datasets import get_dataset
from src.churn_labeling import (
    create_churn_labels, get_train_test_cutoffs, compute_imbalance_ratio,
)
from src.feature_engineering import engineer_features
from src.modeling import train_models
from src.evaluation import evaluate_model, compute_imbalance_metrics
from src.baselines import random_baseline
from src.utils import ensure_dir, get_logger, set_seed, timeit

logger = get_logger(__name__)


def run_sensitivity_analysis(
    dataset: str = "olist",
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Run the pipeline with multiple churn windows for a single dataset.

    Parameters
    ----------
    dataset : str
        Dataset name.
    windows : list of int, optional
        Churn windows to test.  If None, uses defaults from config.

    Returns
    -------
    pd.DataFrame with columns:
        dataset, churn_window_days, churn_rate, imbalance_ratio,
        model, roc_auc, avg_precision, f1, brier_score.
    """
    from src.config import SENSITIVITY_WINDOWS

    if windows is None:
        windows = SENSITIVITY_WINDOWS.get(dataset, [])

    if not windows:
        logger.info(
            "Sensitivity | No windows defined for '%s' — skipping", dataset,
        )
        return pd.DataFrame()

    set_seed(RANDOM_SEED)
    adapter = get_dataset(dataset)
    output_dir = ensure_dir(
        os.path.join(RESULTS_DIR, SENSITIVITY_RESULTS_DIR)
    )

    # ── Load data once, reuse across windows ─────────────────────────
    logger.info("Sensitivity | Loading data for '%s' …", dataset)
    df = adapter.load_raw_data()
    df = adapter.preprocess(df)
    df = adapter.standardize_schema(df)

    all_results = []
    default_window = adapter.churn_window_days or 180

    for window in sorted(windows):
        logger.info(
            "Sensitivity | %s — churn window: %d days",
            dataset, window,
        )

        try:
            train_cutoff, test_cutoff = get_train_test_cutoffs(
                df, TRAIN_SPLIT_QUANTILE, prediction_window_days=window,
            )
        except Exception as exc:
            logger.warning(
                "Sensitivity | Cutoff failed for %s window=%d: %s",
                dataset, window, exc,
            )
            continue

        # Labels
        train_labels = create_churn_labels(
            df, train_cutoff, prediction_window_days=window,
        )
        test_labels = create_churn_labels(
            df, test_cutoff, prediction_window_days=window,
        )

        # Features
        train_feats = engineer_features(
            df, train_cutoff,
            customer_ids=train_labels['customer_id'].tolist(),
            available_groups=adapter.available_feature_groups,
        )
        test_feats = engineer_features(
            df, test_cutoff,
            customer_ids=test_labels['customer_id'].tolist(),
            available_groups=adapter.available_feature_groups,
        )

        if train_feats.empty or test_feats.empty:
            logger.warning(
                "Sensitivity | Empty feature matrix for window=%d", window,
            )
            continue

        # Align
        train_labels = train_labels.set_index('customer_id')
        test_labels = test_labels.set_index('customer_id')
        common_train = train_feats.index.intersection(train_labels.index)
        common_test = test_feats.index.intersection(test_labels.index)
        if len(common_train) < 10 or len(common_test) < 10:
            logger.warning(
                "Sensitivity | Too few customers for window=%d: "
                "train=%d, test=%d", window, len(common_train), len(common_test),
            )
            continue

        X_train = train_feats.loc[common_train]
        y_train = train_labels.loc[common_train, 'churn']
        X_test = test_feats.loc[common_test]
        y_test = test_labels.loc[common_test, 'churn']

        # Align columns
        for c in set(X_train.columns) - set(X_test.columns):
            X_test[c] = 0.0
        X_test = X_test[X_train.columns]

        # Train models
        from sklearn.model_selection import train_test_split as tts
        X_tr, X_val, y_tr, y_val = tts(
            X_train, y_train, test_size=0.1,
            random_state=RANDOM_SEED, stratify=y_train,
        )
        models = train_models(X_tr, y_tr, X_val, y_val)

        imb = compute_imbalance_metrics(y_test)

        for name, model in models.items():
            metrics, _, _ = evaluate_model(model, X_test, y_test, name)
            all_results.append({
                'dataset': dataset,
                'churn_window_days': window,
                'churn_rate': round(imb['churn_rate'], 4),
                'imbalance_ratio': round(imb['imbalance_ratio'], 2),
                'model': name,
                'roc_auc': metrics.get('roc_auc', np.nan),
                'avg_precision': metrics.get('avg_precision', np.nan),
                'f1': metrics.get('f1', np.nan),
                'precision': metrics.get('precision', np.nan),
                'recall': metrics.get('recall', np.nan),
                'brier_score': metrics.get('brier_score', np.nan),
                'calibration_error': metrics.get('calibration_error', np.nan),
                'n_test': metrics.get('n_test', 0),
                'n_pos': metrics.get('n_pos', 0),
            })

    if not all_results:
        logger.warning("Sensitivity | No results for dataset '%s'", dataset)
        return pd.DataFrame()

    result_df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, f'{dataset}_sensitivity.csv')
    result_df.to_csv(csv_path, index=False)
    logger.info(
        "Sensitivity | Results saved to %s (%d rows, %d windows)",
        csv_path, len(result_df), len(windows),
    )

    # Log directional stability summary
    if 'roc_auc' in result_df.columns:
        for model in result_df['model'].unique():
            sub = result_df[result_df['model'] == model].sort_values('churn_window_days')
            if len(sub) > 1:
                aucs = sub['roc_auc'].dropna()
                if len(aucs) > 1:
                    trend = aucs.iloc[-1] - aucs.iloc[0]
                    logger.info(
                        "Sensitivity | %s/%s: AUC trend across windows = %.3f "
                        "(from %.3f to %.3f)",
                        dataset, model, trend, aucs.iloc[0], aucs.iloc[-1],
                    )

    return result_df


def run_sensitivity_all_datasets() -> pd.DataFrame:
    """Run sensitivity analysis for all registered datasets."""
    from src.datasets import list_datasets
    from src.config import SENSITIVITY_WINDOWS

    all_results = []
    for dataset in list_datasets():
        windows = SENSITIVITY_WINDOWS.get(dataset, [])
        if windows:
            try:
                result = run_sensitivity_analysis(dataset, windows=windows)
                if not result.empty:
                    all_results.append(result)
            except Exception as exc:
                logger.warning(
                    "Sensitivity | Failed for dataset '%s': %s",
                    dataset, exc,
                )

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_dir = ensure_dir(
            os.path.join(RESULTS_DIR, SENSITIVITY_RESULTS_DIR)
        )
        path = os.path.join(output_dir, 'all_datasets_sensitivity.csv')
        combined.to_csv(path, index=False)
        logger.info(
            "Sensitivity | Combined results saved to %s (%d rows)",
            path, len(combined),
        )
        return combined

    return pd.DataFrame()
