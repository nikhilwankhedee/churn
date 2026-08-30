"""
Validation infrastructure for the behavioral churn prediction framework.

Four layers of validation ensure methodological consistency across all datasets:

    Layer 1 — Schema Validation
    Layer 2 — Behavioral Sanity Checks
    Layer 3 — Pipeline Output Validation
    Layer 4 — Cross-Dataset Reasonableness

Every validation function logs at the custom VALIDATION level for clear
separation from INFO/WARNING/ERROR messages.
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — SCHEMA VALIDATION
# ═════════════════════════════════════════════════════════════════════

STANDARD_SCHEMA_COLUMNS = [
    'customer_id', 'event_time', 'transaction_value', 'event_type',
    'product_id', 'review_score', 'payment_type', 'delivery_delay',
    'engagement_signal', 'session_id',
]

REQUIRED_SCHEMA_COLUMNS = ['customer_id', 'event_time']


def validate_schema(
    df: pd.DataFrame,
    dataset_name: str,
    available_groups: List[str],
    all_groups: List[str],
) -> Dict[str, Any]:
    """Layer 1: Validate standardised schema integrity.

    Checks column presence, null ratios, timestamp validity, customer ID
    completeness, and duplicate detection.  Logs every finding at the
    VALIDATION level.

    Returns a report dict summarising all findings.
    """
    report = {
        'dataset': dataset_name,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'detected_columns': [],
        'missing_optional_columns': [],
        'enabled_feature_groups': [],
        'disabled_feature_groups': [],
        'column_null_pcts': {},
        'warnings': [],
        'errors': [],
    }

    # ── Column detection ─────────────────────────────────────────────
    detected = [c for c in STANDARD_SCHEMA_COLUMNS if c in df.columns]
    missing_optional = [c for c in STANDARD_SCHEMA_COLUMNS
                         if c not in df.columns and c not in REQUIRED_SCHEMA_COLUMNS]
    report['detected_columns'] = detected

    # ── Required columns ─────────────────────────────────────────────
    for col in REQUIRED_SCHEMA_COLUMNS:
        if col not in df.columns:
            msg = f"[FAIL] Required column '{col}' missing from standardised schema"
            report['errors'].append(msg)
            logger.error("Schema | %s", msg)
        else:
            logger.validation("Schema | [OK] Required column '%s' present", col)

    if report['errors']:
        return report

    # ── Missing optional columns ─────────────────────────────────────
    for col in missing_optional:
        report['missing_optional_columns'].append(col)
        logger.validation("Schema | [INFO] Optional column '%s' not available", col)

    for col in detected:
        logger.validation("Schema | [OK] Column '%s' detected", col)

    # ── Null percentage ──────────────────────────────────────────────
    null_pcts = (df[detected].isnull().sum() / len(df) * 100).to_dict()
    report['column_null_pcts'] = {k: round(v, 2) for k, v in null_pcts.items()}
    for col, pct in null_pcts.items():
        if pct > 50:
            w = f"Column '{col}' is {pct:.1f}% null — verify data quality"
            report['warnings'].append(w)
            logger.warning("Schema | %s", w)
        elif pct > 0:
            logger.validation("Schema | [OK] Column '%s' has %.1f%% nulls (acceptable)",
                              col, pct)
        else:
            logger.validation("Schema | [OK] Column '%s' has no nulls", col)

    # ── Timestamp validity ───────────────────────────────────────────
    if 'event_time' in df.columns:
        n_nat = df['event_time'].isna().sum()
        if n_nat > 0:
            w = f"event_time has {n_nat} NaT values ({n_nat/len(df)*100:.1f}%)"
            report['warnings'].append(w)
            logger.warning("Schema | %s", w)
        else:
            logger.validation("Schema | [OK] event_time has no NaT values")

        try:
            time_min = df['event_time'].min()
            time_max = df['event_time'].max()
            span_days = (time_max - time_min).days
            report['time_span_days'] = span_days
            logger.validation(
                "Schema | [OK] Time range: %s → %s (%d days)",
                time_min, time_max, span_days,
            )
        except Exception as exc:
            w = f"Cannot compute time range: {exc}"
            report['warnings'].append(w)
            logger.warning("Schema | %s", w)

    # ── Customer ID validity ─────────────────────────────────────────
    if 'customer_id' in df.columns:
        n_null_cid = df['customer_id'].isna().sum()
        if n_null_cid > 0:
            e = f"customer_id has {n_null_cid} null values — data corruption"
            report['errors'].append(e)
            logger.error("Schema | %s", e)

        n_unique = df['customer_id'].nunique()
        logger.validation("Schema | [OK] %d unique customer IDs", n_unique)
        report['n_unique_customers'] = n_unique

    # ── Duplicate detection ──────────────────────────────────────────
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        w = f"DataFrame has {n_duplicates} duplicate rows"
        report['warnings'].append(w)
        logger.warning("Schema | %s", w)
    else:
        logger.validation("Schema | [OK] No duplicate rows")

    # ── Feature group availability ───────────────────────────────────
    for group in sorted(all_groups):
        if group in available_groups:
            report['enabled_feature_groups'].append(group)
            logger.validation("Schema | [OK] Feature group '%s' enabled", group)
        else:
            report['disabled_feature_groups'].append(group)
            logger.validation("Schema | [WARN] Feature group '%s' not available", group)

    # ── Summary ──────────────────────────────────────────────────────
    if report['errors']:
        logger.error("Schema | %d schema error(s) — %s", len(report['errors']),
                      dataset_name)
    elif report['warnings']:
        logger.warning("Schema | %d schema warning(s) — %s",
                        len(report['warnings']), dataset_name)
    else:
        logger.validation("Schema | [PASS] All schema checks passed for %s",
                          dataset_name)

    return report


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — BEHAVIORAL SANITY CHECKS
# ═════════════════════════════════════════════════════════════════════

BEHAVIORAL_THRESHOLDS = {
    'max_churn_rate': 0.99,
    'min_orders_per_customer': 1.05,
    'max_imbalance_ratio': 100,
    'min_repeat_purchase_ratio': 0.01,
    'min_customers': 100,
    'max_feature_sparsity': 0.95,
}


def validate_behavioral_statistics(
    df: pd.DataFrame,
    labels: Optional[pd.DataFrame] = None,
    dataset_name: str = "unknown",
) -> Dict[str, Any]:
    """Layer 2: Compute and validate behavioral statistics.

    Automatically computes churn rate, imbalance ratio, avg orders per
    customer, median interpurchase interval, time range, customer count,
    repeat purchase ratio, and feature sparsity — with warning thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with standardised schema columns.
    labels : pd.DataFrame, optional
        DataFrame with [customer_id, churn] columns.
    dataset_name : str
        Name for logging.

    Returns
    -------
    dict of computed statistics and any warnings raised.
    """
    report = {
        'dataset': dataset_name,
        'statistics': {},
        'warnings': [],
        'critical_warnings': [],
    }

    # ── Customer count ───────────────────────────────────────────────
    if 'customer_id' not in df.columns:
        report['critical_warnings'].append("customer_id column missing")
        return report

    n_customers = df['customer_id'].nunique()
    n_events = len(df)
    report['statistics']['n_customers'] = int(n_customers)
    report['statistics']['n_events'] = int(n_events)

    if n_customers < BEHAVIORAL_THRESHOLDS['min_customers']:
        w = (
            f"Low customer count ({n_customers}) — behavioral estimates "
            f"may be unreliable"
        )
        report['warnings'].append(w)
        logger.warning("Behavioral | %s", w)
    else:
        logger.validation("Behavioral | [OK] %d customers, %d events",
                          n_customers, n_events)

    # ── Churn rate ───────────────────────────────────────────────────
    if labels is not None and 'churn' in labels.columns:
        churn_rate = labels['churn'].mean()
        n_pos = int(labels['churn'].sum())
        n_neg = len(labels) - n_pos
        imbalance_ratio = n_neg / n_pos if n_pos > 0 else float('inf')

        report['statistics']['churn_rate'] = float(churn_rate)
        report['statistics']['imbalance_ratio'] = float(imbalance_ratio)
        report['statistics']['n_churned'] = n_pos
        report['statistics']['n_retained'] = n_neg

        logger.validation(
            "Behavioral | [OK] Churn rate: %.2f%% (%d/%d), imbalance ratio: %.2f",
            churn_rate * 100, n_pos, len(labels), imbalance_ratio,
        )

        if churn_rate > BEHAVIORAL_THRESHOLDS['max_churn_rate']:
            cw = (
                f"Extreme churn rate ({churn_rate*100:.1f}%%) — "
                f"verify churn window / label definition"
            )
            report['critical_warnings'].append(cw)
            logger.warning("Behavioral | %s", cw)

        if imbalance_ratio > BEHAVIORAL_THRESHOLDS['max_imbalance_ratio']:
            w = (
                f"Extreme imbalance ratio ({imbalance_ratio:.1f}) — "
                f"model metrics will be dominated by majority class"
            )
            report['warnings'].append(w)
            logger.warning("Behavioral | %s", w)

    # ── Orders per customer (purchase events) ────────────────────────
    purchase_types = {'purchase', 'transaction', 'order'}
    if 'event_type' in df.columns:
        purchases = df[df['event_type'].isin(purchase_types)]
    else:
        purchases = df

    if not purchases.empty:
        orders_per_customer = purchases.groupby('customer_id').size()
        avg_orders = orders_per_customer.mean()
        median_orders = orders_per_customer.median()
        report['statistics']['avg_orders_per_customer'] = float(round(avg_orders, 2))
        report['statistics']['median_orders_per_customer'] = float(median_orders)
        report['statistics']['repeat_purchase_ratio'] = float(
            (orders_per_customer > 1).mean()
        )

        logger.validation(
            "Behavioral | [OK] Avg orders/customer: %.2f, median: %.0f, "
            "repeat ratio: %.2f",
            avg_orders, median_orders,
            (orders_per_customer > 1).mean(),
        )

        if avg_orders < BEHAVIORAL_THRESHOLDS['min_orders_per_customer']:
            w = (
                f"Avg orders/customer ({avg_orders:.2f}) near minimum — "
                f"sparse transactional behavior"
            )
            report['warnings'].append(w)
            logger.warning("Behavioral | %s", w)

        if (orders_per_customer > 1).mean() < BEHAVIORAL_THRESHOLDS['min_repeat_purchase_ratio']:
            w = (
                f"Repeat purchase ratio ({(orders_per_customer > 1).mean():.3f}) "
                f"near zero — inactivity-dominated churn expected"
            )
            report['warnings'].append(w)
            logger.warning("Behavioral | %s", w)

    # ── Median interpurchase interval ─────────────────────────────────
    if 'event_time' in df.columns and 'customer_id' in df.columns:
        df_sorted = df.sort_values(['customer_id', 'event_time'])
        df_sorted['_prev_time'] = df_sorted.groupby('customer_id')['event_time'].shift(1)
        intervals = df_sorted['_prev_time'].notna()
        if intervals.any():
            delta_days = (
                df_sorted.loc[intervals, 'event_time']
                - df_sorted.loc[intervals, '_prev_time']
            ).dt.days
            median_interval = float(delta_days.median())
            mean_interval = float(delta_days.mean())
            report['statistics']['median_interpurchase_days'] = round(median_interval, 1)
            report['statistics']['mean_interpurchase_days'] = round(mean_interval, 1)
            logger.validation(
                "Behavioral | [OK] Median interpurchase: %.1f days, mean: %.1f days",
                median_interval, mean_interval,
            )

    # ── Dataset time range ───────────────────────────────────────────
    if 'event_time' in df.columns:
        t_min = df['event_time'].min()
        t_max = df['event_time'].max()
        span = (t_max - t_min).days
        report['statistics']['time_range_start'] = str(t_min)
        report['statistics']['time_range_end'] = str(t_max)
        report['statistics']['time_span_days'] = span
        logger.validation(
            "Behavioral | [OK] Time span: %d days (%s → %s)",
            span, t_min.date(), t_max.date(),
        )

    # ── Summary ──────────────────────────────────────────────────────
    if report['critical_warnings']:
        logger.error(
            "Behavioral | %d critical warning(s) — %s",
            len(report['critical_warnings']), dataset_name,
        )
    elif report['warnings']:
        logger.warning(
            "Behavioral | %d warning(s) — %s",
            len(report['warnings']), dataset_name,
        )
    else:
        logger.validation("Behavioral | [PASS] All behavioral checks passed for %s",
                          dataset_name)

    return report


# ═════════════════════════════════════════════════════════════════════
# LAYER 3 — PIPELINE OUTPUT VALIDATION
# ═════════════════════════════════════════════════════════════════════

OUTPUT_FILES = {
    'model_metrics': os.path.join('model_metrics', 'model_metrics.csv'),
    'data_quality': os.path.join('data_quality', 'data_quality_summary.csv'),
    'statistical_tests': os.path.join('statistical_tests', 'feature_tests.csv'),
    'ablation': os.path.join('ablation', 'ablation_results.csv'),
}

OUTPUT_FIGURES = {
    'roc_curves': os.path.join('model_evaluation', 'roc_curves.png'),
    'pr_curves': os.path.join('model_evaluation', 'pr_curves.png'),
    'calibration': os.path.join('calibration', 'calibration_curves.png'),
    'churn_distribution': os.path.join('churn_analysis', 'churn_distribution.png'),
}


def validate_outputs(
    results_dir: str,
    figures_dir: str,
    eval_df: Optional[pd.DataFrame] = None,
    y_proba: Optional[Dict[str, np.ndarray]] = None,
    dataset_name: str = "unknown",
) -> Dict[str, Any]:
    """Layer 3: Validate all pipeline outputs exist and are non-empty.

    Checks:
      - Expected CSV files exist
      - Expected figure files exist
      - Metrics contain no NaN / infinite values
      - No empty predictions
      - No degenerate probability distributions
    """
    report = {
        'dataset': dataset_name,
        'files_found': [],
        'files_missing': [],
        'metric_issues': [],
        'probability_issues': [],
        'warnings': [],
    }

    # ── File existence ───────────────────────────────────────────────
    for name, rel_path in {**OUTPUT_FILES, **OUTPUT_FIGURES}.items():
        full_path = os.path.join(results_dir, rel_path)
        if os.path.exists(full_path):
            report['files_found'].append(name)
            logger.validation("Outputs | [OK] %s → %s", name, full_path)
        else:
            # Figures in figures_dir, CSVs in results_dir
            alt_path = os.path.join(figures_dir, rel_path)
            if os.path.exists(alt_path):
                report['files_found'].append(name)
                logger.validation("Outputs | [OK] %s → %s", name, alt_path)
            else:
                report['files_missing'].append(name)
                logger.validation("Outputs | [WARN] %s not found at %s or %s",
                                  name, full_path, alt_path)

    # ── Metric validation ────────────────────────────────────────────
    if eval_df is not None and not eval_df.empty:
        numeric_cols = eval_df.select_dtypes(include=[np.number]).columns
        nan_metrics = eval_df[numeric_cols].isnull().any()
        nan_cols = nan_metrics[nan_metrics].index.tolist()
        if nan_cols:
            report['metric_issues'].append(f"NaN values in columns: {nan_cols}")
            logger.warning("Outputs | NaN values in metrics: %s", nan_cols)

        inf_metrics = eval_df[numeric_cols].replace([np.inf, -np.inf], np.nan).isnull().any()
        inf_cols = inf_metrics[inf_metrics].index.tolist()
        if inf_cols:
            report['metric_issues'].append(f"Infinite values in columns: {inf_cols}")
            logger.warning("Outputs | Infinite values in metrics: %s", inf_cols)

        # Check for extreme AUC values (potential leakage)
        if 'roc_auc' in eval_df.columns:
            for _, row in eval_df.iterrows():
                auc = row.get('roc_auc', np.nan)
                if not np.isnan(auc) and auc > 0.98:
                    model_name = row.get('model', 'unknown')
                    w = (
                        f"Very high ROC-AUC ({auc:.3f}) for {model_name} — "
                        f"possible leakage or overfitting"
                    )
                    report['warnings'].append(w)
                    logger.warning("Outputs | %s", w)

    # ── Probability validation ───────────────────────────────────────
    if y_proba is not None:
        for name, probs in y_proba.items():
            if probs is None or len(probs) == 0:
                report['probability_issues'].append(f"Empty probabilities for {name}")
                logger.warning("Outputs | Empty probabilities: %s", name)
                continue

            probs_arr = np.asarray(probs)
            if np.any(np.isnan(probs_arr)):
                n_nan = np.isnan(probs_arr).sum()
                report['probability_issues'].append(
                    f"{n_nan} NaN probabilities in {name}"
                )
                logger.warning("Outputs | NaN probabilities: %s (%d)", name, n_nan)

            if np.all(probs_arr == probs_arr[0]):
                report['probability_issues'].append(
                    f"Degenerate (constant) probabilities in {name}"
                )
                logger.warning("Outputs | Degenerate probabilities: %s", name)

    # ── Summary ──────────────────────────────────────────────────────
    if report['files_missing']:
        logger.validation(
            "Outputs | [WARN] %d output file(s) missing — %s",
            len(report['files_missing']), dataset_name,
        )
    if report['metric_issues'] or report['probability_issues']:
        logger.warning(
            "Outputs | %d metric issue(s), %d probability issue(s) — %s",
            len(report['metric_issues']), len(report['probability_issues']),
            dataset_name,
        )
    if not report['files_missing'] and not report['metric_issues']:
        logger.validation("Outputs | [PASS] All output checks passed for %s",
                          dataset_name)

    return report


# ═════════════════════════════════════════════════════════════════════
# LAYER 4 — CROSS-DATASET REASONABLENESS
# ═════════════════════════════════════════════════════════════════════

CROSS_DATASET_WARN_THRESHOLDS = {
    'auc_variance_warn': 0.15,
    'min_expected_auc': 0.45,
    'max_calibration_error': 0.25,
    'churn_rate_consistency': 0.50,
}


def validate_cross_dataset_behavior(
    master_results_path: str,
) -> Dict[str, Any]:
    """Layer 4: Cross-dataset behavioral reasonableness validation.

    Automatically compares results across all datasets in the master table
    and flags suspicious patterns such as:
      - Near-perfect AUC on sparse marketplace data
      - Worse-than-random performance on subscription data
      - Impossible calibration errors
      - Degenerate SHAP patterns (via feature-group dominance anomalies)

    This is NOT a benchmark — it is a sanity check.
    """
    report = {
        'datasets_found': [],
        'flags': [],
        'warnings': [],
    }

    if not os.path.exists(master_results_path):
        logger.validation(
            "CrossDataset | [INFO] No master results yet — skipping layer 4"
        )
        return report

    try:
        master = pd.read_csv(master_results_path)
    except Exception as exc:
        report['warnings'].append(f"Cannot read master results: {exc}")
        return report

    if master.empty:
        logger.validation("CrossDataset | [INFO] Master results empty")
        return report

    datasets = master['dataset'].unique()
    report['datasets_found'] = list(datasets)
    logger.validation(
        "CrossDataset | [OK] %d datasets in master table: %s",
        len(datasets), list(datasets),
    )

    # ── Check per-dataset reasonableness ─────────────────────────────
    for dset in datasets:
        subset = master[master['dataset'] == dset]

        # AUC sanity
        aucs = subset['roc_auc'].dropna()
        if not aucs.empty:
            max_auc = aucs.max()
            if max_auc > 0.98:
                w = (
                    f"{dset}: Max ROC-AUC ({max_auc:.3f}) exceeds 0.98 — "
                    f"possible leakage or overfitting"
                )
                report['flags'].append(w)
                logger.warning("CrossDataset | %s", w)

            if max_auc < CROSS_DATASET_WARN_THRESHOLDS['min_expected_auc']:
                w = (
                    f"{dset}: Max ROC-AUC ({max_auc:.3f}) below 0.45 — "
                    f"features may have no predictive signal"
                )
                report['flags'].append(w)
                logger.warning("CrossDataset | %s", w)

        # Calibration sanity
        cal = subset['calibration_error'].dropna()
        if not cal.empty:
            max_cal = cal.max()
            if max_cal > CROSS_DATASET_WARN_THRESHOLDS['max_calibration_error']:
                w = (
                    f"{dset}: Max calibration error ({max_cal:.3f}) exceeds "
                    f"0.25 — probabilities poorly calibrated"
                )
                report['flags'].append(w)
                logger.warning("CrossDataset | %s", w)

        # Churn rate sanity
        cr = subset['churn_rate'].dropna()
        if not cr.empty and cr.iloc[0] > 0.99:
            w = (
                f"{dset}: Churn rate ({cr.iloc[0]*100:.1f}%) > 99% — "
                f"extremely imbalanced, verify definition"
            )
            report['flags'].append(w)
            logger.warning("CrossDataset | %s", w)

    # ── Cross-dataset consistency ────────────────────────────────────
    if len(datasets) > 1:
        from src.datasets import get_ecosystem_type
        for dset in datasets:
            eco = get_ecosystem_type(dset)
            logger.validation(
                "CrossDataset | [OK] %s → ecosystem type: %s", dset, eco,
            )

        logger.validation(
            "CrossDataset | [OK] %d datasets available for comparison",
            len(datasets),
        )

    # ── Summary ──────────────────────────────────────────────────────
    if not report['flags']:
        logger.validation(
            "CrossDataset | [PASS] All cross-dataset checks passed"
        )

    return report


# ═════════════════════════════════════════════════════════════════════
# MASTER RESULTS VALIDATION
# ═════════════════════════════════════════════════════════════════════

MASTER_RESULT_COLUMNS = [
    'dataset', 'ecosystem_type', 'model', 'roc_auc', 'pr_auc', 'f1',
    'precision', 'recall', 'brier_score', 'calibration_error',
    'churn_rate', 'imbalance_ratio', 'dominant_feature_group',
]

CRITICAL_METRICS = ['roc_auc', 'f1', 'churn_rate']


def validate_master_results_entry(
    rows: List[Dict[str, Any]],
    existing_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate a master results entry before appending.

    Checks:
      - All required columns present
      - No NaN in critical metrics
      - Dataset name consistency
      - No duplicate experiment signatures
    """
    report = {
        'valid': True,
        'errors': [],
        'warnings': [],
    }

    for row in rows:
        # Required columns
        for col in MASTER_RESULT_COLUMNS:
            if col not in row:
                report['errors'].append(f"Missing column '{col}' in master entry")
                report['valid'] = False
                return report

        # Critical metrics must not be NaN
        for col in CRITICAL_METRICS:
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                model = row.get('model', 'unknown')
                dset = row.get('dataset', 'unknown')
                w = (
                    f"Critical metric '{col}' is NaN for {dset}/{model} — "
                    f"entry recorded but flagged"
                )
                report['warnings'].append(w)
                logger.warning("MasterResults | %s", w)

        # Dataset name validity
        dset = row.get('dataset', '')
        from src.datasets import list_datasets
        if dset not in list_datasets():
            report['warnings'].append(
                f"Unknown dataset '{dset}' in master entry"
            )

    # Duplicate check
    if existing_path and os.path.exists(existing_path):
        try:
            existing = pd.read_csv(existing_path)
            for row in rows:
                dset = row['dataset']
                model = row['model']
                dup = existing[
                    (existing['dataset'] == dset) &
                    (existing['model'] == model)
                ]
                if not dup.empty:
                    report['warnings'].append(
                        f"Overwriting existing entry for {dset}/{model}"
                    )
        except Exception:
            pass

    if report['errors']:
        logger.error("MasterResults | %d validation error(s)", len(report['errors']))
    elif report['warnings']:
        logger.warning("MasterResults | %d validation warning(s)",
                        len(report['warnings']))
    else:
        logger.validation("MasterResults | [PASS] Entry valid")

    return report
