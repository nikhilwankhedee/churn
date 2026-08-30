"""
Model training with balanced class handling and deterministic behaviour.

Trains three classifiers:
  1. Logistic Regression  (LBFGS, balanced class_weight)
  2. Random Forest         (balanced subsample)
  3. XGBoost               (scale_pos_weight, early stopping)
  4. LightGBM              (is_unbalance)
  5. SVM                   (calibrated LinearSVC)

All random seeds are fixed.  XGBoost early stopping is optional and
requires separate validation data.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from src.config import (
    RANDOM_SEED,
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    SVM_PARAMS,
)
from src.utils import get_logger, set_seed

logger = get_logger(__name__)
set_seed(RANDOM_SEED)

try:
    import lightgbm as lgb
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    _LIGHTGBM_AVAILABLE = False


def _compute_scale_pos_weight(y: pd.Series) -> float:
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    if pos == 0 or neg == 0:
        return 1.0
    return neg / pos


def _build_xgb_kwargs() -> dict:
    kwargs = dict(XGBOOST_PARAMS)
    kwargs.pop('scale_pos_weight', None)
    kwargs.pop('use_label_encoder', None)
    return kwargs


def _without_class_weights(params: dict) -> dict:
    params = dict(params)
    if 'class_weight' in params:
        params['class_weight'] = None
    if 'is_unbalance' in params:
        params['is_unbalance'] = False
    return params


def _svm_training_subset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dataset_name: Optional[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    max_rows = None
    if dataset_name == 'retailrocket':
        max_rows = 10_000
    elif dataset_name == 'lastfm':
        max_rows = 5_000

    if max_rows is None or len(X_train) <= max_rows:
        return X_train, y_train

    tmp = X_train.copy()
    tmp['_target'] = y_train.values
    frac = max_rows / len(tmp)
    sampled = (
        tmp.groupby('_target', group_keys=False)
        .apply(lambda x: x.sample(
            n=max(1, int(round(len(x) * frac))),
            random_state=RANDOM_SEED,
        ))
        .sample(frac=1.0, random_state=RANDOM_SEED)
    )
    y_sub = sampled['_target'].astype(y_train.dtype)
    X_sub = sampled.drop(columns=['_target'])
    logger.info(
        "SVM training subset for %s: %d → %d rows",
        dataset_name, len(X_train), len(X_sub),
    )
    return X_sub, y_sub


def train_models(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    dataset_name: Optional[str] = None,
    use_smote: bool = False,
) -> Dict[str, object]:
    models: Dict[str, object] = {}
    scale_pos = _compute_scale_pos_weight(y_train)
    n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    logger.info("Training distribution — neg: %d, pos: %d, ratio: %.2f",
                 n_neg, n_pos, scale_pos)

    # ── Logistic Regression ────────────────────────────────────────
    logger.info("Training LogisticRegression …")
    lr_params = (
        _without_class_weights(LOGISTIC_REGRESSION_PARAMS)
        if use_smote else LOGISTIC_REGRESSION_PARAMS
    )
    lr = LogisticRegression(**lr_params)
    lr.fit(X_train, y_train)
    models['logistic_regression'] = lr

    # ── Random Forest ──────────────────────────────────────────────
    logger.info("Training RandomForest …")
    rf_params = (
        _without_class_weights(RANDOM_FOREST_PARAMS)
        if use_smote else RANDOM_FOREST_PARAMS
    )
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)
    models['random_forest'] = rf

    # ── XGBoost ────────────────────────────────────────────────────
    logger.info("Training XGBoost …")
    xgb_params = _build_xgb_kwargs()
    xgb_params['scale_pos_weight'] = 1.0 if use_smote else scale_pos

    if X_val is not None and y_val is not None:
        xgb_params['early_stopping_rounds'] = 10
        model = XGBClassifier(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)

    models['xgboost'] = model

    # ── LightGBM ───────────────────────────────────────────────────
    if _LIGHTGBM_AVAILABLE:
        logger.info("Training LightGBM …")
        lgb_params = (
            _without_class_weights(LIGHTGBM_PARAMS)
            if use_smote else dict(LIGHTGBM_PARAMS)
        )
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train)
        models['lightgbm'] = lgb_model
    else:
        logger.warning("LightGBM not installed — skipping lightgbm model")

    # ── SVM ────────────────────────────────────────────────────────
    logger.info("Training calibrated LinearSVC …")
    svm_params = _without_class_weights(SVM_PARAMS) if use_smote else dict(SVM_PARAMS)
    X_svm, y_svm = _svm_training_subset(X_train, y_train, dataset_name)
    svm_base = LinearSVC(**svm_params)
    try:
        svm_model = CalibratedClassifierCV(estimator=svm_base)
    except TypeError:
        svm_model = CalibratedClassifierCV(base_estimator=svm_base)
    svm_model.fit(X_svm, y_svm)
    models['svm'] = svm_model

    logger.info("All %d models trained successfully", len(models))
    return models
