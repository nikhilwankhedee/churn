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
  8. Model training (LR, RF, XGBoost, LightGBM, SVM) + baselines
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
    run_pipeline(dataset="credit_card")
    run_pipeline(dataset="lastfm")
"""
import datetime
import hashlib
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts

from src.ablation import run_ablation
from src.baselines import majority_class_baseline, random_baseline
from src.calibration import plot_calibration_curves
from src.churn_labeling import (
    create_churn_labels,
    get_train_test_cutoffs,
    stratified_native_split,
)
from src.config import (
    FIGURE_SUBDIRS,
    FIGURES_DIR,
    MODELS_DIR,
    PREDICTION_WINDOW_DAYS,
    PROCESSED_DIR,
    RANDOM_SEED,
    RESULT_SUBDIRS,
    RESULTS_DIR,
    SENSITIVITY_ENABLED,
    SHAP_SAMPLE_SIZE,
    SMOTE_K_NEIGHBORS,
    TRAIN_SPLIT_QUANTILE,
)
from src.data_quality import generate_data_quality_report
from src.datasets import get_dataset, get_ecosystem_type
from src.evaluation import (
    compute_imbalance_metrics,
    evaluate_model,
    get_pr_data,
    threshold_analysis,
)
from src.experiment_tracker import log_experiment
from src.explainability import shap_analysis
from src.exports import (
    append_to_master_results,
    save_data_quality_report,
    save_evaluation_table,
    save_experiment_artifacts,
    save_experiment_metadata,
    save_models,
    save_processed_data,
    save_risk_scores,
    save_shap_values,
)
from src.failure_analysis import analyze_errors, behavioral_comparison
from src.feature_engineering import engineer_features
from src.modeling import AVAILABLE_MODELS, MODEL_ORDER, train_models
from src.risk_scoring import generate_risk_table
from src.segmentation import segment_customers
from src.statistical_tests import feature_distribution_tests
from src.utils import (
    ensure_dir,
    get_logger,
    set_seed,
    timeit,
)
from src.validators import (
    validate_cross_dataset_behavior,
    validate_outputs,
)
from src.visualization import (
    plot_ablation_results,
    plot_behavioral_insights,
    plot_churn_distribution,
    plot_confusion_matrices,
    plot_correlation_heatmap,
    plot_delivery_delay_distribution,
    plot_feature_importance,
    plot_pr_curves,
    plot_roc_curves,
    plot_segmentation,
    plot_threshold_analysis,
)

logger = get_logger(__name__)
set_seed(RANDOM_SEED)


def _create_directories() -> None:
    for d in [FIGURES_DIR, RESULTS_DIR, MODELS_DIR, PROCESSED_DIR]:
        ensure_dir(d)
    for sub in FIGURE_SUBDIRS:
        ensure_dir(os.path.join(FIGURES_DIR, sub))
    for sub in RESULT_SUBDIRS:
        ensure_dir(os.path.join(RESULTS_DIR, sub))


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
    data_dir: Optional[str] = None,
    use_smote: bool = False,
    model_names: Optional[List[str]] = None,
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full behavioural churn prediction pipeline for a dataset.

    Stateless experiment design (Section 8 of the experiment spec): the
    SMOTE condition and model subset are explicit parameters — they never
    mutate global configuration.

    Parameters
    ----------
    dataset : str
        One of: olist, rees46, retailrocket, online_retail_ii, instacart,
        telco, credit_card, lastfm, kkbox.
    sensitivity : bool
        If True, also run sensitivity analysis for this dataset.
    churn_window_override : int, optional
        Override the default churn window for this run.
    data_dir : str, optional
        Explicit directory containing raw data files. When provided, the
        adapter loads data from this directory regardless of environment.
    use_smote : bool, default False
        If True, apply SMOTE to the training split only (never the
        validation/test split).
    model_names : list of str, optional
        Subset of models to train.  None → all five available models.
    results_dir : str, optional
        Base directory for the isolated per-condition outputs
        (``{results_dir}/{without|with_smote}/{dataset}/``).
        None → central RESULTS_DIR.

    Returns
    -------
    dict of pipeline metadata (metrics, validation reports, timing).
    """
    start_time = datetime.datetime.utcnow()
    logger.info("=" * 60)
    logger.info("Behavioural Churn Prediction Pipeline — dataset: %s", dataset)
    logger.info("SMOTE: %s | Models: %s",
                use_smote, ','.join(model_names or AVAILABLE_MODELS))
    logger.info("=" * 60)

    # ── 0. Load dataset adapter ──────────────────────────────────────
    adapter = get_dataset(dataset, data_dir=data_dir)
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
    logger.validation("Config | Temporal data: %s", adapter.has_temporal_data)

    condition = 'with_smote' if use_smote else 'without_smote'
    exp_base_dir = os.path.join(results_dir or RESULTS_DIR, condition, dataset)

    _create_directories()
    ensure_dir(exp_base_dir)
    logger.validation("Config | Output directories created")

    # ── 1. Load raw data ────────────────────────────────────────────
    logger.info("── Step 1/17: Load raw data ──")
    df = adapter.load_raw_data()
    dq_report = generate_data_quality_report(df)
    save_data_quality_report(dq_report, base_dir=exp_base_dir)
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

    # ── 5. Train/test split ─────────────────────────────────────────
    logger.info("── Step 4/17: Train/test split ──")
    # Native-label datasets without a genuine event timeline (credit_card,
    # telco) cannot use temporal cutoffs: the raw data has no real
    # timestamps, and a synthetic event_time must never be used as a
    # workaround.  They get a customer-level stratified 70/30 split instead,
    # and the snapshot filter is disabled so features are never truncated
    # against a fake timestamp.  SMOTE still runs later, strictly on the
    # training split.  Temporal datasets (olist, retailrocket, rees46,
    # instacart, online_retail_ii, lastfm, kkbox) keep the exact same
    # temporal methodology as before.
    non_temporal_native = (
        adapter.uses_native_churn_label and not adapter.has_temporal_data)
    if non_temporal_native:
        labels_all = adapter.get_native_churn_labels(
            df, df['event_time'].max())
        train_cust_ids, test_cust_ids, train_labels_df, test_labels_df = (
            stratified_native_split(labels_all))
        train_cutoff = test_cutoff = df['event_time'].max()
        logger.validation(
            "Split | Stratified native — train: %d, test: %d customers",
            len(train_cust_ids), len(test_cust_ids),
        )
    elif adapter.uses_native_churn_label:
        # Temporal native-label dataset (e.g. kkbox): quantile cutoff,
        # adapter extracts train/test labels per cutoff.
        max_date = df['event_time'].max()
        train_cutoff = df['event_time'].quantile(TRAIN_SPLIT_QUANTILE)
        test_cutoff = max_date
        logger.validation(
            "Cutoffs | Native label — train: %s, test: %s",
            train_cutoff.date(), test_cutoff.date(),
        )
    else:
        # Behavioural (inactivity-based) temporal path — unchanged.
        train_cutoff, test_cutoff = get_train_test_cutoffs(
            df, TRAIN_SPLIT_QUANTILE, churn_window,
        )

    # ── 6. Training labels ──────────────────────────────────────────
    logger.info("── Step 5/17: Train labels ──")
    if not non_temporal_native:
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
        filter_by_snapshot=adapter.has_temporal_data,
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
    if not non_temporal_native:
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
        filter_by_snapshot=adapter.has_temporal_data,
    )
    if test_features.empty:
        raise RuntimeError("Test feature matrix is empty")
    test_labels_df = test_labels_df.set_index('customer_id')
    test_labels_df = test_labels_df.loc[
        test_features.index.intersection(test_labels_df.index)
    ]
    test_features = test_features.loc[test_labels_df.index]

    # ── Leakage guard: identifier / target columns must not leak ────
    for name, feat in [('train', train_features), ('test', test_features)]:
        leaked_target = 'churn' in feat.columns
        leaked_id = 'customer_id' in feat.columns
        if leaked_target or leaked_id:
            raise RuntimeError(
                f"Leakage detected in {name} feature matrix — "
                f"target_col={leaked_target}, id_col={leaked_id}"
            )
        logger.validation(
            "Leakage | %s features contain no target/identifier columns",
            name,
        )

    # Align train/test columns
    for c in set(train_features.columns) - set(test_features.columns):
        test_features[c] = 0.0
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
                         base_dir=exp_base_dir)

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

    # ── Single-class guard ───────────────────────────────────────────
    # Both classes are required to train meaningful models and to compute
    # ROC-AUC / PR-AUC.  A run whose train or test labels collapse to one
    # class is a failed experiment — never a "success" with NaN metrics
    # (which previously crashed downstream publication aggregation).
    n_classes_train = int(y_train.nunique())
    n_classes_test = int(y_test.nunique())
    if n_classes_train < 2 or n_classes_test < 2:
        raise RuntimeError(
            f"Churn labels are single-class — train has {n_classes_train} "
            f"class(es) (churn {imb_train['churn_rate'] * 100:.1f}%), test "
            f"has {n_classes_test} class(es) (churn "
            f"{imb_test['churn_rate'] * 100:.1f}%). Models and ROC-AUC "
            f"require both classes, so this run is marked FAILED."
        )
    if imb_train['imbalance_ratio'] > 10:
        logger.warning(
            "IMBALANCE | High ratio (%.1f) — metrics may be dominated "
            "by majority class", imb_train['imbalance_ratio'],
        )

    stratify_y = y_train if y_train.nunique() >= 2 else None
    X_tr, X_val, y_tr, y_val = tts(
        X_train, y_train, test_size=0.1,
        random_state=RANDOM_SEED, stratify=stratify_y,
    )

    # ── 10b. Stateless SMOTE (train split only — never test) ────────
    # SMOTE is applied to X_tr/y_tr ONLY.  X_val/y_val and X_test/y_test
    # are never resampled (Sections 6-7 of the experiment spec).
    if use_smote:
        logger.info("── Applying SMOTE to training split only ──")
        from src.resamplers import get_resampler
        resampler = get_resampler('smote')
        resample_result = resampler.resample(
            X_tr, y_tr,
            random_state=RANDOM_SEED,
            k_neighbors=SMOTE_K_NEIGHBORS,
        )
        X_tr = resample_result.X_resampled
        y_tr = resample_result.y_resampled
        logger.info(
            "Resampling complete (smote) — %d → %d samples (+%d synthetic)",
            resample_result.n_original, resample_result.n_resampled,
            resample_result.n_synthetic,
        )
        assert len(X_tr) == len(y_tr), "SMOTE: X/y length mismatch"

    # ── Test identity fingerprint (for cross-condition verification) ─
    test_ids_hash = hashlib.sha256(
        np.asarray(sorted([str(x) for x in X_test.index])).tobytes()
    ).hexdigest()
    test_y_hash = hashlib.sha256(
        np.asarray(y_test.values, dtype=float).tobytes()
    ).hexdigest()
    logger.validation(
        "Identity | Test set — %d customers, id-hash=%s, y-hash=%s",
        len(X_test), test_ids_hash[:16], test_y_hash[:16],
    )

    # ── Train/test customer overlap (measured, never ignored) ────────
    # Temporal datasets legitimately share customers between the train and
    # test windows (a user active before the train cutoff can still be
    # active after it).  The overlap is reported explicitly so it is a
    # documented property of the split, not a hidden defect.
    train_ids = set(train_features.index)
    test_ids = set(X_test.index)
    n_overlap = len(train_ids & test_ids)
    overlap_ratio = (n_overlap / len(test_ids)) if len(test_ids) else 0.0
    logger.validation(
        "Overlap | %d/%d test customers also in train (%.1f%%) — "
        "expected for temporal splits, documented not ignored",
        n_overlap, len(test_ids), overlap_ratio * 100,
    )

    n_classes = y_train.nunique()
    if n_classes < 2:
        logger.warning(
            "Skipping model training — only %d class(es) in training data "
            "(churn rate %.1f%%). Models require at least 2 classes.",
            n_classes, imb_train['churn_rate'] * 100,
        )
        models = {}
    else:
        models = train_models(
            X_tr, y_tr, X_val, y_val, model_names=model_names,
        )
        save_models(models, base_dir=exp_base_dir)

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
                plot_threshold_analysis(tdf, name)
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
                        FIGURES_DIR, 'model_evaluation',
                        f'{name}_importance.png',
                    ),
                )
            except Exception as exc:
                logger.warning("Feature importance plot failed for %s: %s",
                                name, exc)

    eval_df = pd.DataFrame(metrics_list)
    save_evaluation_table(eval_df, base_dir=exp_base_dir)
    logger.validation("Metrics | %d model evaluations saved", len(eval_df))

    # ── 11b. Isolated experiment artifacts (Section 30) ─────────────
    save_experiment_artifacts(
        dataset=dataset,
        use_smote=use_smote,
        metrics_df=eval_df,
        y_test=y_test,
        prob_dict=prob_dict,
        meta={
            'condition': condition,
            'ecosystem_type': ecosystem_type,
            'use_smote': use_smote,
            'model_names': list(model_names or MODEL_ORDER),
            'train_cutoff': str(train_cutoff),
            'test_cutoff': str(test_cutoff),
            'churn_window_days': churn_window,
            'test_ids_hash': test_ids_hash,
            'test_y_hash': test_y_hash,
        },
        results_dir=results_dir,
    )
    logger.validation(
        "Artifacts | %s/%s saved (metrics, predictions, comparison, "
        "report, metadata)", condition, dataset,
    )

    # ── 13. Visualisations ──────────────────────────────────────────
    logger.info("── Step 12/17: Visualisations ──")
    viz_steps = [
        (plot_roc_curves, "ROC", [prob_dict, y_test]),
        (plot_confusion_matrices, "Confusion matrix", [cm_dict]),
        (plot_pr_curves, "PR", [pr_data, y_test]),
        (plot_churn_distribution, "Churn distribution", [y_test]),
        (plot_correlation_heatmap, "Correlation heatmap", [X_test]),
        (plot_delivery_delay_distribution, "Delivery delay", [X_test]),
    ]
    for fn, desc, args in viz_steps:
        try:
            fn(*args)
        except Exception as exc:
            logger.warning("Visualisation '%s' failed: %s", desc, exc)

    # ── 14. Calibration ─────────────────────────────────────────────
    logger.info("── Step 13/17: Calibration ──")
    try:
        plot_calibration_curves(prob_dict, y_test)
    except Exception as exc:
        logger.warning("Calibration curves failed: %s", exc)

    # ── 15. SHAP Explainability ─────────────────────────────────────
    logger.info("── Step 14/17: SHAP explainability ──")
    for name, model in models.items():
        try:
            n_sample = min(SHAP_SAMPLE_SIZE, len(X_test))
            X_sample = X_test.sample(n_sample, random_state=RANDOM_SEED)
            sv, _ = shap_analysis(model, X_sample, name)
            save_shap_values(sv, X_test.columns.tolist(), name,
                             base_dir=exp_base_dir)
        except Exception as exc:
            logger.warning("SHAP failed for %s: %s", name, exc)

    # ── 16. Segmentation ────────────────────────────────────────────
    logger.info("── Step 15/17: Segmentation ──")
    try:
        seg_df = segment_customers(X_train)
        plot_segmentation(seg_df)
    except Exception as exc:
        logger.warning("Segmentation failed: %s", exc)

    # ── 17. Statistical tests ───────────────────────────────────────
    logger.info("── Step 16/17: Statistical tests ──")
    try:
        stat_results = feature_distribution_tests(X_train, y_train)
        stat_results.to_csv(
            os.path.join(exp_base_dir, 'statistical_tests',
                          'feature_tests.csv'),
            index=False,
        )
    except Exception as exc:
        logger.warning("Statistical tests failed: %s", exc)

    # ── Ablation study ──────────────────────────────────────────────
    ablation_df = None
    try:
        ablation_df = run_ablation(X_train, y_train)
        if ablation_df is not None and not ablation_df.empty:
            ablation_df.to_csv(
                os.path.join(exp_base_dir, 'ablation', 'ablation_results.csv'),
                index=False,
            )
            plot_ablation_results(ablation_df)
    except Exception as exc:
        logger.warning("Ablation study failed: %s", exc)

    # ── Behavioural insights ────────────────────────────────────────
    try:
        plot_behavioral_insights(X_test, y_test)
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
            save_risk_scores(risk_df, best_model_name,
                             base_dir=exp_base_dir)
        except Exception as exc:
            logger.warning("Risk scoring failed: %s", exc)

    # ── Failure analysis ────────────────────────────────────────────
    if best_model is not None:
        try:
            y_pred_best = best_model.predict(X_test)
            y_proba_best = best_model.predict_proba(X_test)[:, 1]
            err = analyze_errors(X_test, y_test, y_pred_best, y_proba_best)
            pd.DataFrame(err).transpose().to_csv(
                os.path.join(exp_base_dir, 'failure_analysis',
                              'error_groups.csv'),
            )
            fp_mask = (y_test == 0) & (y_pred_best == 1)
            fn_mask = (y_test == 1) & (y_pred_best == 0)
            fp_feats = X_test[fp_mask] if fp_mask.sum() > 0 else None
            fn_feats = X_test[fn_mask] if fn_mask.sum() > 0 else None
            comp = behavioral_comparison(fp_feats, fn_feats)
            if not comp.empty:
                comp.to_csv(
                    os.path.join(exp_base_dir, 'failure_analysis',
                                  'fp_fn_comparison.csv'),
                )
        except Exception as exc:
            logger.warning("Failure analysis failed: %s", exc)

    # ── Output validation (Layer 3) ────────────────────────────────
    logger.info("── Layer 3: Output validation ──")
    output_report = validate_outputs(
        results_dir=RESULTS_DIR,
        figures_dir=FIGURES_DIR,
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
        master_path = os.path.join(
            RESULTS_DIR, 'cross_dataset', 'master_results.csv',
        )
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
                'condition': condition,
                'use_smote': use_smote,
                'model_names': ','.join(model_names or MODEL_ORDER),
                'test_ids_hash': test_ids_hash,
                'test_y_hash': test_y_hash,
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
        save_experiment_metadata(meta, base_dir=exp_base_dir)
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
        'condition': condition,
        'use_smote': use_smote,
        'duration_seconds': elapsed.total_seconds(),
        'churn_rate': float(imb_test['churn_rate']),
        'imbalance_ratio': float(imb_test['imbalance_ratio']),
        'best_model': best_model_name,
        'test_ids_hash': test_ids_hash,
        'test_y_hash': test_y_hash,
        'train_test_overlap': int(n_overlap),
        'train_test_overlap_ratio': float(overlap_ratio),
        'schema_errors': len(schema_report.get('errors', [])),
        'schema_warnings': len(schema_report.get('warnings', [])),
        'behavioral_warnings': len(behavioral_report.get('warnings', [])),
        'output_files_missing': len(output_report.get('files_missing', [])),
        'dominant_feature_group': dominant_group,
    }
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
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
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


def main(
    dataset: str = "olist",
    sensitivity: bool = False,
    use_smote: bool = False,
    model_names: Optional[List[str]] = None,
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience entry point — called by the notebook.

    Delegates to run_pipeline() with the default dataset.
    This function exists to preserve the notebook import:
        from src.pipeline import main
    """
    return run_pipeline(
        dataset=dataset,
        sensitivity=sensitivity,
        use_smote=use_smote,
        model_names=model_names,
        results_dir=results_dir,
    )


if __name__ == '__main__':
    import sys
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else "olist"
    sensitivity_arg = '--sensitivity' in sys.argv
    use_smote_arg = '--use-smote' in sys.argv
    run_pipeline(
        dataset=dataset_arg,
        sensitivity=sensitivity_arg,
        use_smote=use_smote_arg,
    )
