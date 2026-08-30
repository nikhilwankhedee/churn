"""
Master pipeline — orchestrates the full behavioural churn prediction workflow
across any registered dataset adapter, with integrated validation.

Execution flow:
  1. Load dataset adapter from registry
  2. Load, preprocess, and standardise raw data
  3. Schema validation (Layer 1)
  4. Temporal train/test cutoff generation
  5. Churn label creation (inactivity-based or native)
  6. Feature engineering on standardised schema
  7. Behavioral sanity checks (Layer 2)
  8. Model training (LR, RF, XGBoost) + baselines
  9. Evaluation & threshold analysis
  10. Calibration curves
  11. SHAP explainability
  12. Customer segmentation
  13. Statistical tests
  14. Ablation study
  15. Behavioural insights
  16. Risk scoring
  17. Failure analysis
  18. Output validation (Layer 3)
  19. Experiment tracking
  20. Cross-dataset master results + validation (Layer 4)

All components degrade gracefully — optional failures are logged but
do not halt the pipeline.

Usage:
    from src.pipeline import run_pipeline
    run_pipeline(dataset="olist")
    run_pipeline(dataset="retailrocket")
    run_pipeline(dataset="telco")
"""
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split as tts
from typing import Optional, Dict, Any, List

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    SMOTE = None
    _SMOTE_AVAILABLE = False

from src.config import (
    FIGURE_SUBDIRS, RESULT_SUBDIRS,
    TRAIN_SPLIT_QUANTILE, RANDOM_SEED, PREDICTION_WINDOW_DAYS,
    SHAP_SAMPLE_SIZE, SENSITIVITY_ENABLED, CALIBRATION_N_BINS,
)
from src.run_context import (
    figures_dir, models_dir, processed_dir, results_dir,
    master_results_path,
)
from src.utils import (
    set_seed, ensure_dir, get_logger, timeit,
)
from src.datasets import get_dataset, get_ecosystem_type
from src.churn_labeling import (
    create_churn_labels, get_train_test_cutoffs,
)
from src.feature_engineering import engineer_features
from src.modeling import train_models
from src.evaluation import (
    evaluate_model, threshold_analysis, get_pr_data,
    compute_imbalance_metrics,
)
from src.baselines import majority_class_baseline, random_baseline
from src.calibration import plot_calibration_curves
from src.explainability import shap_analysis
from src.visualization import (
    plot_roc_curves, plot_pr_curves, plot_confusion_matrices,
    plot_feature_importance, plot_correlation_heatmap,
    plot_segmentation, plot_churn_distribution,
    plot_ablation_results, plot_behavioral_insights,
    plot_threshold_analysis, plot_delivery_delay_distribution,
)
from src.segmentation import segment_customers
from src.ablation import run_ablation
from src.risk_scoring import generate_risk_table
from src.statistical_tests import feature_distribution_tests
from src.data_quality import generate_data_quality_report
from src.failure_analysis import analyze_errors, behavioral_comparison
from src.exports import (
    save_models, save_processed_data, save_evaluation_table,
    save_shap_values, save_risk_scores, save_data_quality_report,
    save_experiment_metadata, append_to_master_results,
)
from src.experiment_tracker import log_experiment
from src.validators import (
    validate_schema, validate_behavioral_statistics,
    validate_outputs, validate_cross_dataset_behavior,
)

logger = get_logger(__name__)
set_seed(RANDOM_SEED)


def _create_directories() -> None:
    from src.run_context import (
        figures_dir, results_dir, models_dir, processed_dir,
    )
    models_dir()
    processed_dir()
    for sub in FIGURE_SUBDIRS:
        figures_dir(sub)
    for sub in RESULT_SUBDIRS:
        results_dir(sub)


def _compute_dominant_group(
    ablation_result: pd.DataFrame,
    best_model_name: str,
) -> str:
    """Determine the most impactful feature group from ablation results."""
    if ablation_result is None or ablation_result.empty:
        return 'unknown'
    best_rows = ablation_result[
        ablation_result['model'] == best_model_name
    ]
    if best_rows.empty:
        return 'unknown'
    full_rows = best_rows[
        best_rows['feature_set'] == 'all_features'
    ]
    if full_rows.empty:
        return 'unknown'
    full_auc = full_rows['mean_roc_auc'].values[0]
    drops = best_rows[
        best_rows['feature_set'] != 'all_features'
    ].copy()
    if drops.empty:
        return 'unknown'
    drops['_drop'] = full_auc - drops['mean_roc_auc']
    dominant = drops.sort_values('_drop', ascending=False).iloc[0]
    return str(dominant['feature_set']).replace('without_', '')


@timeit
def run_pipeline(
    dataset: str = "olist",
    sensitivity: bool = False,
    churn_window_override: Optional[int] = None,
    use_smote: bool = False,
    collect_calibration: bool = False,
) -> Dict[str, Any]:
    """Run the full behavioural churn prediction pipeline for a dataset.

    Parameters
    ----------
    dataset : str
        One of the registered dataset names: olist, rees46, retailrocket,
        online_retail_ii, instacart, telco, lastfm, credit_card.
    sensitivity : bool
        If True, also run sensitivity analysis for this dataset.
    churn_window_override : int, optional
        Override the default churn window for this run.
    use_smote : bool
        If True, apply SMOTE to the training fold (removing class weights) and
        suffix all outputs with ``_smote``.
    collect_calibration : bool
        If True, include the per-model probability arrays and the test labels
        in the returned metadata (used by :func:`run_smote_comparison`).

    Returns
    -------
    dict of pipeline metadata (metrics, validation reports, timing).
    """
    start_time = datetime.datetime.utcnow()
    logger.info("=" * 60)
    mode_suffix = '_smote' if use_smote else ''
    from src.run_context import (
        set_run_scope, results_dir, figures_dir, master_results_path,
    )
    set_run_scope(dataset, use_smote)
    logger.info("Behavioural Churn Prediction Pipeline — dataset: %s", dataset)
    logger.info("Training mode: %s", "smote" if use_smote else "original")
    logger.info("=" * 60)

    # ── 0. Load dataset adapter ──────────────────────────────────────
    adapter = get_dataset(dataset)
    ecosystem_type = get_ecosystem_type(dataset)
    meta = adapter.metadata

    if churn_window_override is not None:
        churn_window = churn_window_override
    else:
        churn_window = adapter.churn_window_days or PREDICTION_WINDOW_DAYS

    available_groups = list(adapter.available_feature_groups)
    logger.info("Ecosystem type: %s", ecosystem_type)
    logger.validation("Config | Churn window: %s days", churn_window)
    logger.validation("Config | Available feature groups: %s", available_groups)
    logger.validation("Config | Native churn label: %s", adapter.uses_native_churn_label)

    _create_directories()
    logger.validation("Config | Output directories created")

    # ── 1. Load raw data ────────────────────────────────────────────
    logger.info("── Step 1/17: Load raw data ──")
    df = adapter.load_raw_data()
    dq_report = generate_data_quality_report(df)
    save_data_quality_report(dq_report, suffix=mode_suffix)
    logger.validation("Data | %d rows, %d columns loaded", df.shape[0], df.shape[1])

    # ── 2. Preprocess ───────────────────────────────────────────────
    logger.info("── Step 2/17: Preprocess ──")
    df = adapter.preprocess(df)
    logger.validation("Data | Preprocessed shape: %d × %d", df.shape[0], df.shape[1])

    # ── 3. Standardise schema ───────────────────────────────────────
    logger.info("── Step 3/17: Standardise schema ──")
    df = adapter.standardize_schema(df)
    logger.validation("Data | Standardised columns: %s", list(df.columns))

    # ── 4. Schema validation (Layer 1) ──────────────────────────────
    logger.info("── Layer 1: Schema validation ──")
    schema_report = adapter.validate_schema(df)
    if schema_report.get('errors'):
        logger.error("Schema validation failed — halting pipeline")
        raise RuntimeError(
            f"Schema validation failed for {dataset}: {schema_report['errors']}"
        )

    custom_user_relative = hasattr(adapter, 'build_user_relative_modeling_data')
    custom_native_split = hasattr(adapter, 'build_native_modeling_data')

    if custom_user_relative:
        logger.info("── Step 4/17: User-relative modeling split ──")
        train_features, test_features, train_labels_df, test_labels_df = (
            adapter.build_user_relative_modeling_data(df)
        )
        train_cutoff = 'user_relative_observation'
        test_cutoff = 'user_relative_future'
        logger.validation(
            "UserRelative | Train: %d customers | Test: %d customers",
            len(train_features), len(test_features),
        )
    elif custom_native_split:
        logger.info("── Step 4/17: Native-label stratified split ──")
        train_features, test_features, train_labels_df, test_labels_df = (
            adapter.build_native_modeling_data(df)
        )
        train_cutoff = 'native_stratified_train'
        test_cutoff = 'native_stratified_test'
        logger.validation(
            "NativeSplit | Train: %d customers | Test: %d customers",
            len(train_features), len(test_features),
        )
    else:
        # ── 5. Temporal cutoffs ─────────────────────────────────────────
        logger.info("── Step 4/17: Temporal cutoffs ──")
        if adapter.uses_native_churn_label:
            max_date = df['event_time'].max()
            train_cutoff = df['event_time'].quantile(TRAIN_SPLIT_QUANTILE)
            test_cutoff = max_date
            logger.validation(
                "Cutoffs | Native label — train: %s, test: %s",
                train_cutoff.date(), test_cutoff.date(),
            )
        else:
            train_cutoff, test_cutoff = get_train_test_cutoffs(
                df, TRAIN_SPLIT_QUANTILE, churn_window,
            )

        # ── 6. Training labels ──────────────────────────────────────────
        logger.info("── Step 5/17: Train labels ──")
        if adapter.uses_native_churn_label:
            train_labels_df = adapter.get_native_churn_labels(df, train_cutoff)
            train_cust_ids = train_labels_df['customer_id'].tolist()
        else:
            train_labels_df = create_churn_labels(
                df, train_cutoff, prediction_window_days=churn_window,
            )
            train_cust_ids = train_labels_df['customer_id'].tolist()

        # ── 7. Training features ────────────────────────────────────────
        logger.info("── Step 6/17: Train features ──")
        train_features = engineer_features(
            df, train_cutoff, customer_ids=train_cust_ids,
            available_groups=available_groups,
        )
        if train_features.empty:
            raise RuntimeError("Training feature matrix is empty — check data/snapshot")
        train_labels_df = train_labels_df.set_index('customer_id')
        train_labels_df = train_labels_df.loc[
            train_features.index.intersection(train_labels_df.index)
        ]
        train_features = train_features.loc[train_labels_df.index]
        churn_rate_train = train_labels_df['churn'].mean()
        logger.validation(
            "Train | %d customers, churn rate %.2f%%",
            len(train_features), churn_rate_train * 100,
        )

        # ── 8. Test labels ──────────────────────────────────────────────
        logger.info("── Step 7/17: Test labels ──")
        if adapter.uses_native_churn_label:
            test_labels_df = adapter.get_native_churn_labels(df, test_cutoff)
            test_cust_ids = test_labels_df['customer_id'].tolist()
        else:
            test_labels_df = create_churn_labels(
                df, test_cutoff, prediction_window_days=churn_window,
            )
            test_cust_ids = test_labels_df['customer_id'].tolist()

        # ── 9. Test features ────────────────────────────────────────────
        logger.info("── Step 8/17: Test features ──")
        test_features = engineer_features(
            df, test_cutoff, customer_ids=test_cust_ids,
            available_groups=available_groups,
        )
        if test_features.empty:
            raise RuntimeError("Test feature matrix is empty")
        test_labels_df = test_labels_df.set_index('customer_id')
        test_labels_df = test_labels_df.loc[
            test_features.index.intersection(test_labels_df.index)
        ]
        test_features = test_features.loc[test_labels_df.index]

    if 'customer_id' in train_labels_df.columns:
        train_labels_df = train_labels_df.set_index('customer_id')
    if 'customer_id' in test_labels_df.columns:
        test_labels_df = test_labels_df.set_index('customer_id')
    train_labels_df = train_labels_df.loc[
        train_features.index.intersection(train_labels_df.index)
    ]
    train_features = train_features.loc[train_labels_df.index]
    test_labels_df = test_labels_df.loc[
        test_features.index.intersection(test_labels_df.index)
    ]
    test_features = test_features.loc[test_labels_df.index]

    churn_rate_train = train_labels_df['churn'].mean()
    logger.validation(
        "Train | %d customers, churn rate %.2f%%",
        len(train_features), churn_rate_train * 100,
    )

    # Align train/test columns
    for c in set(train_features.columns) - set(test_features.columns):
        test_features[c] = 0.0
    for c in set(test_features.columns) - set(train_features.columns):
        train_features[c] = 0.0
    test_features = test_features[train_features.columns]
    churn_rate_test = test_labels_df['churn'].mean()
    logger.validation(
        "Test | %d customers, churn rate %.2f%%",
        len(test_features), churn_rate_test * 100,
    )

    # ── Behavioral sanity checks (Layer 2) ──────────────────────────
    logger.info("── Layer 2: Behavioral sanity checks ──")
    train_labels_for_bh = train_labels_df.reset_index()
    behavioral_report = adapter.validate_behavioral_statistics(
        df=df,
        labels=train_labels_for_bh,
    )

    # ── 10. Save processed data ─────────────────────────────────────
    logger.info("── Step 9/17: Save processed data ──")
    save_processed_data(train_features, test_features,
                         train_labels_df, test_labels_df,
                         suffix=mode_suffix)

    # ── 11. Train models ────────────────────────────────────────────
    logger.info("── Step 10/17: Train models ──")
    X_train, y_train = train_features, train_labels_df['churn']
    X_test, y_test = test_features, test_labels_df['churn']

    imb_train = compute_imbalance_metrics(y_train)
    imb_test = compute_imbalance_metrics(y_test)
    logger.validation(
        "Imbalance | Train — churn: %.2f%%, ratio: %.2f | "
        "Test — churn: %.2f%%, ratio: %.2f",
        imb_train['churn_rate'] * 100, imb_train['imbalance_ratio'],
        imb_test['churn_rate'] * 100, imb_test['imbalance_ratio'],
    )
    if imb_train['imbalance_ratio'] > 10:
        logger.warning(
            "IMBALANCE | High ratio (%.1f) — metrics may be dominated "
            "by majority class", imb_train['imbalance_ratio'],
        )

    X_tr, X_val, y_tr, y_val = tts(
        X_train, y_train, test_size=0.1,
        random_state=RANDOM_SEED, stratify=y_train,
    )
    if use_smote:
        if not _SMOTE_AVAILABLE:
            raise ImportError("imblearn is required for use_smote=True")
        sm = SMOTE(random_state=RANDOM_SEED)
        X_tr_resampled, y_tr_resampled = sm.fit_resample(X_tr, y_tr)
        X_tr = pd.DataFrame(X_tr_resampled, columns=X_train.columns)
        y_tr = pd.Series(y_tr_resampled, name='churn')
        logger.validation(
            "SMOTE | Resampled training fold to %d rows, churn rate %.2f%%",
            len(X_tr), y_tr.mean() * 100,
        )
    models = train_models(
        X_tr, y_tr, X_val, y_val,
        dataset_name=dataset,
        use_smote=use_smote,
    )
    save_models(models, suffix=mode_suffix)

    # ── 12. Evaluate models ─────────────────────────────────────────
    logger.info("── Step 11/17: Evaluate models ──")
    prob_dict: Dict[str, np.ndarray] = {}
    pr_data: Dict[str, tuple] = {}
    cm_dict: Dict[str, np.ndarray] = {}
    metrics_list = []

    # Baselines
    logger.info("── Baseline models ──")
    try:
        y_maj = majority_class_baseline(y_train, y_test)
        maj_metrics, _, _ = _evaluate_predictions(
            y_test, y_maj, None, 'majority_class',
        )
        metrics_list.append(maj_metrics)
    except Exception as exc:
        logger.warning("Majority-class baseline failed: %s", exc)

    try:
        y_rand, p_rand = random_baseline(y_train, y_test)
        rand_metrics, _, _ = _evaluate_predictions(
            y_test, y_rand, p_rand, 'random_baseline',
        )
        metrics_list.append(rand_metrics)
        prob_dict['random_baseline'] = p_rand
    except Exception as exc:
        logger.warning("Random baseline failed: %s", exc)

    for name, model in models.items():
        metrics, cm, y_proba = evaluate_model(model, X_test, y_test, name)
        metrics_list.append(metrics)
        cm_dict[name] = cm

        if y_proba is not None:
            prob_dict[name] = y_proba
            try:
                prec, rec, ap = get_pr_data(y_test, y_proba)
                pr_data[name] = (prec, rec, ap)
            except Exception as exc:
                logger.warning("PR data failed for %s: %s", name, exc)

            try:
                tdf = threshold_analysis(y_test, y_proba)
                plot_threshold_analysis(tdf, name, suffix=mode_suffix)
            except Exception as exc:
                logger.warning("Threshold analysis failed for %s: %s", name, exc)

        if hasattr(model, 'feature_importances_'):
            try:
                imp_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_,
                }).sort_values('importance', ascending=False)
                plot_feature_importance(
                    imp_df,
                    title=f'{name} — Feature Importance',
                    save_path=os.path.join(
                        figures_dir('model_evaluation'),
                        f'{name}_importance{mode_suffix}.png',
                    ),
                )
            except Exception as exc:
                logger.warning("Feature importance plot failed for %s: %s",
                                name, exc)

    eval_df = pd.DataFrame(metrics_list)
    save_evaluation_table(eval_df, filename=f'model_metrics{mode_suffix}.csv')
    logger.validation("Metrics | %d model evaluations saved", len(eval_df))

    # ── 13. Visualisations ──────────────────────────────────────────
    logger.info("── Step 12/17: Visualisations ──")
    viz_steps = [
        (plot_roc_curves, "ROC", [prob_dict, y_test, mode_suffix]),
        (plot_confusion_matrices, "Confusion matrix", [cm_dict, mode_suffix]),
        (plot_pr_curves, "PR", [pr_data, y_test, mode_suffix]),
        (plot_churn_distribution, "Churn distribution", [y_test, mode_suffix]),
        (plot_correlation_heatmap, "Correlation heatmap", [X_test, mode_suffix]),
        (plot_delivery_delay_distribution, "Delivery delay", [X_test, mode_suffix]),
    ]
    for fn, desc, args in viz_steps:
        try:
            fn(*args)
        except Exception as exc:
            logger.warning("Visualisation '%s' failed: %s", desc, exc)

    # ── 14. Calibration ─────────────────────────────────────────────
    logger.info("── Step 13/17: Calibration ──")
    try:
        plot_calibration_curves(prob_dict, y_test, suffix=mode_suffix)
    except Exception as exc:
        logger.warning("Calibration curves failed: %s", exc)

    # ── 15. SHAP Explainability ─────────────────────────────────────
    logger.info("── Step 14/17: SHAP explainability ──")
    for name, model in models.items():
        try:
            n_sample = min(SHAP_SAMPLE_SIZE, len(X_test))
            X_sample = X_test.sample(n_sample, random_state=RANDOM_SEED)
            sv, _ = shap_analysis(model, X_sample, name, suffix=mode_suffix)
            save_shap_values(sv, X_test.columns.tolist(), name, suffix=mode_suffix)
        except Exception as exc:
            logger.warning("SHAP failed for %s: %s", name, exc)

    # ── 16. Segmentation ────────────────────────────────────────────
    logger.info("── Step 15/17: Segmentation ──")
    try:
        seg_df = segment_customers(X_train)
        plot_segmentation(seg_df, suffix=mode_suffix)
    except Exception as exc:
        logger.warning("Segmentation failed: %s", exc)

    # ── 17. Statistical tests ───────────────────────────────────────
    logger.info("── Step 16/17: Statistical tests ──")
    try:
        stat_results = feature_distribution_tests(X_train, y_train)
        stat_results.to_csv(
            os.path.join(results_dir('statistical_tests'),
                          f'feature_tests{mode_suffix}.csv'),
            index=False,
        )
    except Exception as exc:
        logger.warning("Statistical tests failed: %s", exc)

    # ── Ablation study ──────────────────────────────────────────────
    ablation_df = None
    try:
        ablation_df = run_ablation(X_train, y_train)
        ablation_df.to_csv(
            os.path.join(results_dir('ablation'),
                          f'ablation_results{mode_suffix}.csv'),
            index=False,
        )
        plot_ablation_results(ablation_df, suffix=mode_suffix)
    except Exception as exc:
        logger.warning("Ablation study failed: %s", exc)

    # ── Behavioural insights ────────────────────────────────────────
    try:
        plot_behavioral_insights(X_test, y_test, suffix=mode_suffix)
    except Exception as exc:
        logger.warning("Behavioural insights plot failed: %s", exc)

    # ── Best model selection ────────────────────────────────────────
    best_model = None
    model_names = list(models.keys())
    best_model_name = model_names[0] if models else 'none'
    try:
        if len(eval_df) > 0:
            non_baseline = eval_df[
                ~eval_df['model'].str.contains('baseline', na=False)
            ]
            if len(non_baseline) > 0:
                best_row = non_baseline.sort_values(
                    'roc_auc', ascending=False
                ).iloc[0]
                best_model_name = str(best_row['model'])
                best_model = models.get(best_model_name)
                logger.validation(
                    "Best model | %s (ROC-AUC: %.4f)",
                    best_model_name, best_row['roc_auc'],
                )
    except Exception as exc:
        logger.warning("Best model selection failed: %s", exc)

    # ── Risk scoring (best model) ───────────────────────────────────
    if best_model is not None:
        try:
            y_proba_best = best_model.predict_proba(X_test)[:, 1]
            risk_df = generate_risk_table(
                X_test.index, y_proba_best, best_model_name,
            )
            save_risk_scores(risk_df, best_model_name, suffix=mode_suffix)
        except Exception as exc:
            logger.warning("Risk scoring failed: %s", exc)

    # ── Failure analysis ────────────────────────────────────────────
    if best_model is not None:
        try:
            y_pred_best = best_model.predict(X_test)
            y_proba_best = best_model.predict_proba(X_test)[:, 1]
            err = analyze_errors(X_test, y_test, y_pred_best, y_proba_best)
            pd.DataFrame(err).transpose().to_csv(
                os.path.join(results_dir('failure_analysis'),
                              f'error_groups{mode_suffix}.csv'),
            )
            fp_mask = (y_test == 0) & (y_pred_best == 1)
            fn_mask = (y_test == 1) & (y_pred_best == 0)
            fp_feats = X_test[fp_mask] if fp_mask.sum() > 0 else None
            fn_feats = X_test[fn_mask] if fn_mask.sum() > 0 else None
            comp = behavioral_comparison(fp_feats, fn_feats)
            if not comp.empty:
                comp.to_csv(
                    os.path.join(results_dir('failure_analysis'),
                                  f'fp_fn_comparison{mode_suffix}.csv'),
                )
        except Exception as exc:
            logger.warning("Failure analysis failed: %s", exc)

    # ── Output validation (Layer 3) ────────────────────────────────
    logger.info("── Layer 3: Output validation ──")
    output_report = validate_outputs(
        results_dir=results_dir(),
        figures_dir=figures_dir(),
        eval_df=eval_df,
        y_proba=prob_dict,
        dataset_name=dataset,
    )

    # ── Cross-dataset master results ────────────────────────────────
    dominant_group = _compute_dominant_group(ablation_df, best_model_name)
    logger.validation("Master | Dominant feature group: %s", dominant_group)

    try:
        append_to_master_results(
            dataset_name=dataset,
            ecosystem_type=ecosystem_type,
            churn_rate=float(imb_test['churn_rate']),
            imbalance_ratio=float(imb_test['imbalance_ratio']),
            dominant_feature_group=dominant_group,
            metrics=eval_df,
        )
    except Exception as exc:
        logger.warning("Cross-dataset master results update failed: %s", exc)

    # ── Cross-dataset validation (Layer 4) ──────────────────────────
    logger.info("── Layer 4: Cross-dataset validation ──")
    try:
        master_path = master_results_path()
        validate_cross_dataset_behavior(master_path)
    except Exception as exc:
        logger.warning("Cross-dataset validation failed: %s", exc)

    # ── Experiment tracking ─────────────────────────────────────────
    validation_reports = {
        'schema': schema_report,
        'outputs': output_report,
    }
    try:
        meta = log_experiment(
            metrics_summary=eval_df,
            train_cutoff=train_cutoff,
            test_cutoff=test_cutoff,
            model_names=model_names,
            best_model=best_model_name,
            extra_info={
                'dataset': dataset,
                'ecosystem_type': ecosystem_type,
                'churn_window_days': churn_window,
                'available_feature_groups': ','.join(available_groups),
                'disabled_feature_groups': ','.join(
                    set(schema_report.get('disabled_feature_groups', []))
                ),
                'imbalance_ratio': float(imb_test['imbalance_ratio']),
                'dominant_feature_group': dominant_group,
                'pipeline_duration_sec': round(
                    (datetime.datetime.utcnow() - start_time).total_seconds(), 2
                ),
            },
            validation_reports=validation_reports,
            behavioral_stats=behavioral_report,
        )
        save_experiment_metadata(meta, suffix=mode_suffix)
        logger.validation(
            "Experiment | Metadata saved — %d keys", len(meta),
        )
    except Exception as exc:
        logger.warning("Experiment tracking failed: %s", exc)

    # ── Sensitivity analysis (optional) ─────────────────────────────
    if sensitivity or SENSITIVITY_ENABLED:
        try:
            from src.sensitivity import run_sensitivity_analysis
            logger.info("── Sensitivity analysis ──")
            run_sensitivity_analysis(dataset)
        except Exception as exc:
            logger.warning("Sensitivity analysis failed: %s", exc)

    # ── Pipeline complete ───────────────────────────────────────────
    elapsed = datetime.datetime.utcnow() - start_time
    logger.info("=" * 60)
    logger.validation(
        "PIPELINE | Dataset: %s | Ecosystem: %s | Time: %s | "
        "Churn: %.1f%% | Best model: %s",
        dataset, ecosystem_type, elapsed,
        imb_test['churn_rate'] * 100, best_model_name,
    )
    logger.info("=" * 60)

    pipeline_meta = {
        'dataset': dataset,
        'ecosystem_type': ecosystem_type,
        'duration_seconds': elapsed.total_seconds(),
        'churn_rate': float(imb_test['churn_rate']),
        'imbalance_ratio': float(imb_test['imbalance_ratio']),
        'best_model': best_model_name,
        'schema_errors': len(schema_report.get('errors', [])),
        'schema_warnings': len(schema_report.get('warnings', [])),
        'behavioral_warnings': len(behavioral_report.get('warnings', [])),
        'output_files_missing': len(output_report.get('files_missing', [])),
        'dominant_feature_group': dominant_group,
        'mode': 'smote' if use_smote else 'original',
        'metrics': eval_df,
    }
    if collect_calibration:
        pipeline_meta['prob_dict'] = prob_dict
        pipeline_meta['y_test'] = y_test
    return pipeline_meta


def _evaluate_predictions(
    y_test: pd.Series, y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    model_name: str,
) -> tuple:
    """Evaluate predictions directly without a model object.

    Used for baseline models that don't have a .predict() API.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, brier_score_loss,
        average_precision_score,
    )
    from src.evaluation import _expected_calibration_error

    n_pos = int(y_test.sum())
    n_neg = int((1 - y_test).sum())
    metrics = {
        'model': model_name, 'n_test': len(y_test),
        'n_pos': n_pos, 'n_neg': n_neg,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0.0),
        'recall': recall_score(y_test, y_pred, zero_division=0.0),
        'f1': f1_score(y_test, y_pred, zero_division=0.0),
    }
    if y_proba is not None:
        for metric, fn in [
            ('roc_auc', lambda: roc_auc_score(y_test, y_proba)),
            ('avg_precision', lambda: average_precision_score(y_test, y_proba)),
            ('brier_score', lambda: brier_score_loss(y_test, y_proba)),
            ('calibration_error',
             lambda: _expected_calibration_error(y_test, y_proba)),
        ]:
            try:
                metrics[metric] = fn()
            except Exception:
                metrics[metric] = np.nan

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    metrics['tp'] = int(tp)

    return metrics, cm, y_proba


# ─────────────────────────────────────────────────────────────────────
# SMOTE COMPARISON
# ─────────────────────────────────────────────────────────────────────
def _extract_model_metric(metrics_df: Optional[pd.DataFrame],
                          model: str, metric: str) -> Optional[float]:
    """Pull a single metric value for a model from an evaluation table."""
    if metrics_df is None or metrics_df.empty:
        return None
    rows = metrics_df[metrics_df.get('model', pd.Series(dtype=str)).astype(str) == model]
    if rows.empty:
        return None
    val = rows.iloc[0].get(metric)
    if val is None:
        return None
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None
    if np.isnan(val):
        return None
    return val


def _plot_smote_comparison_figure(summary: pd.DataFrame) -> None:
    """Side-by-side bar chart of Original vs SMOTE mean ROC-AUC per dataset."""
    from src.config import RESULTS_DIR
    d = ensure_dir(os.path.join(RESULTS_DIR, 'cross_dataset'))
    path = os.path.join(d, 'smote_comparison_figure.png')

    fig, ax = plt.subplots(figsize=(max(8, len(summary) * 1.1), 6))
    x = np.arange(len(summary))
    width = 0.38
    ax.bar(x - width / 2, summary['Original_AUC'], width,
           label='Original', color='#4C72B0')
    ax.bar(x + width / 2, summary['SMOTE_AUC'], width,
           label='SMOTE', color='#DD8452')
    ax.set_xticks(x)
    ax.set_xticklabels(summary['Dataset'], rotation=45, ha='right')
    ax.set_ylabel('Mean ROC-AUC across models')
    ax.set_title('SMOTE vs Original — ROC-AUC per Dataset')
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("SMOTE comparison figure saved to %s", path)


def _plot_calibration_comparison(
    dataset: str, orig_prob_dict: dict, orig_y: pd.Series,
    smote_prob_dict: dict, smote_y: pd.Series,
) -> None:
    """Overlay calibration curves (Original vs SMOTE) for a dataset.

    For every model shared by both modes, the original and SMOTE calibration
    curves are overlaid so the effect of SMOTE on probability calibration can
    be inspected per dataset.
    """
    from src.config import FIGURES_DIR
    d = ensure_dir(os.path.join(FIGURES_DIR, 'calibration'))
    path = os.path.join(d, f'{dataset}_calibration_comparison.png')
    if not orig_prob_dict or not smote_prob_dict:
        logger.warning("Calibration comparison skipped for %s — no probabilities",
                       dataset)
        return

    common = set(orig_prob_dict.keys()) & set(smote_prob_dict.keys())
    common = [m for m in common if m not in ('random_baseline', 'majority_class')]
    if not common:
        logger.warning("Calibration comparison skipped for %s — no common models",
                       dataset)
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in sorted(common):
        try:
            ot, op = calibration_curve(
                np.asarray(orig_y), np.asarray(orig_prob_dict[name]),
                n_bins=CALIBRATION_N_BINS, strategy='uniform',
            )
            st, sp = calibration_curve(
                np.asarray(smote_y), np.asarray(smote_prob_dict[name]),
                n_bins=CALIBRATION_N_BINS, strategy='uniform',
            )
        except Exception as exc:
            logger.warning("Calibration comparison failed for %s: %s", name, exc)
            continue
        ax.plot(op, ot, marker='o', label=f'{name} (Original)', color='#4C72B0')
        ax.plot(sp, st, marker='s', linestyle='--', label=f'{name} (SMOTE)',
                color='#DD8452')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'{dataset} — Calibration Comparison: Original vs SMOTE')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("Calibration comparison for %s saved to %s", dataset, path)


def run_smote_comparison(
    datasets: Optional[List[str]] = None,
    include_baselines: bool = False,
) -> Dict[str, Any]:
    """Run the pipeline in original and SMOTE modes for every dataset and
    produce cross-dataset comparison artefacts.

    Generated outputs:
        1. results/cross_dataset/smote_comparison_all_datasets.csv
        2. results/cross_dataset/smote_comparison_figure.png
        3. figures/calibration/{dataset}_calibration_comparison.png

    Parameters
    ----------
    datasets : list of str, optional
        Datasets to compare. Defaults to all registered datasets.
    include_baselines : bool
        If True, include baseline models in the comparison table.

    Returns
    -------
    dict keyed by dataset with per-mode pipeline metadata.
    """
    from src.datasets import list_datasets

    if datasets is None:
        datasets = list_datasets()

    results: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    for ds in datasets:
        logger.info("=" * 60)
        logger.info("SMOTE comparison — dataset: %s", ds)
        logger.info("=" * 60)
        try:
            meta_orig = run_pipeline(
                dataset=ds, use_smote=False, collect_calibration=True,
            )
        except Exception as exc:
            logger.warning("Original mode failed for %s: %s", ds, exc)
            meta_orig = None
        try:
            meta_smote = run_pipeline(
                dataset=ds, use_smote=True, collect_calibration=True,
            )
        except Exception as exc:
            logger.warning("SMOTE mode failed for %s: %s", ds, exc)
            meta_smote = None

        results[ds] = {'original': meta_orig, 'smote': meta_smote}

        if meta_orig is None or meta_smote is None:
            logger.warning("SMOTE comparison incomplete for %s — skipping rows",
                           ds)
            continue

        orig_m = meta_orig.get('metrics')
        smote_m = meta_smote.get('metrics')
        model_names = set()
        if orig_m is not None:
            model_names |= set(orig_m['model'].astype(str).tolist())
        if smote_m is not None:
            model_names |= set(smote_m['model'].astype(str).tolist())
        model_names = {
            m for m in model_names
            if include_baselines or 'baseline' not in m
        }

        for model in sorted(model_names):
            o_auc = _extract_model_metric(orig_m, model, 'roc_auc')
            s_auc = _extract_model_metric(smote_m, model, 'roc_auc')
            o_f1 = _extract_model_metric(orig_m, model, 'f1')
            s_f1 = _extract_model_metric(smote_m, model, 'f1')
            o_brier = _extract_model_metric(orig_m, model, 'brier_score')
            s_brier = _extract_model_metric(smote_m, model, 'brier_score')

            def _delta(o, s):
                if o is None or s is None:
                    return np.nan
                return s - o

            rows.append({
                'Dataset': ds,
                'Model': model,
                'Original_AUC': o_auc if o_auc is not None else np.nan,
                'SMOTE_AUC': s_auc if s_auc is not None else np.nan,
                'Original_F1': o_f1 if o_f1 is not None else np.nan,
                'SMOTE_F1': s_f1 if s_f1 is not None else np.nan,
                'Original_Brier': o_brier if o_brier is not None else np.nan,
                'SMOTE_Brier': s_brier if s_brier is not None else np.nan,
                'AUC_Change': _delta(o_auc, s_auc),
                'F1_Change': _delta(o_f1, s_f1),
                'Brier_Change': _delta(o_brier, s_brier),
            })

        # ── Calibration comparison figure (per dataset) ─────────────
        orig_pd = meta_orig.get('prob_dict', {})
        smote_pd = meta_smote.get('prob_dict', {})
        orig_y = meta_orig.get('y_test')
        smote_y = meta_smote.get('y_test')
        if orig_pd and smote_pd and orig_y is not None and smote_y is not None:
            try:
                _plot_calibration_comparison(
                    ds, orig_pd, orig_y, smote_pd, smote_y,
                )
            except Exception as exc:
                logger.warning("Calibration comparison plot failed for %s: %s",
                               ds, exc)

    # ── Cross-dataset comparison table ──────────────────────────────
    from src.config import RESULTS_DIR
    d = ensure_dir(os.path.join(RESULTS_DIR, 'cross_dataset'))
    cmp_df = pd.DataFrame(rows)
    cmp_path = os.path.join(d, 'smote_comparison_all_datasets.csv')
    cmp_df.to_csv(cmp_path, index=False)
    logger.validation("SMOTE comparison table saved: %s (%d rows)",
                      cmp_path, len(cmp_df))

    # ── Figure: mean AUC per dataset ────────────────────────────────
    try:
        if not cmp_df.empty:
            avg = (
                cmp_df.groupby('Dataset', as_index=False)[
                    ['Original_AUC', 'SMOTE_AUC']
                ].mean()
            )
            _plot_smote_comparison_figure(avg)
    except Exception as exc:
        logger.warning("SMOTE comparison figure failed: %s", exc)

    return results


if __name__ == '__main__':
    import sys
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else "olist"
    sensitivity_arg = '--sensitivity' in sys.argv
    smote_compare = '--smote-comparison' in sys.argv
    if smote_compare:
        run_smote_comparison()
    else:
        run_pipeline(dataset=dataset_arg, sensitivity=sensitivity_arg)
