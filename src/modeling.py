"""
Model training with balanced class handling and deterministic behaviour.

Trains five classifiers:
  1. Logistic Regression  (LBFGS, balanced class_weight)
  2. Random Forest         (balanced subsample)
  3. XGBoost               (scale_pos_weight, early stopping)
  4. LightGBM              (balanced class_weight)
  5. SVM                   (with optional stratified subsampling)

All random seeds are fixed.  XGBoost early stopping is optional and
requires separate validation data.  SVM uses a stratified subset when
the training set exceeds SVM_SUBSET_SIZE to maintain reasonable runtime.
"""
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import (
    RANDOM_SEED,
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    SVM_PARAMS,
    SVM_SUBSET_SIZE,
)
from src.utils import get_logger, set_seed

logger = get_logger(__name__)
set_seed(RANDOM_SEED)

try:
    from lightgbm import LGBMClassifier
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False
    logger.warning("lightgbm not installed — LightGBM classifier disabled")

# Canonical training order used by the final experiment matrix.
MODEL_ORDER: List[str] = [
    'logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'svm',
]

#: Set of model names that can currently be trained in this environment.
AVAILABLE_MODELS: List[str] = list(MODEL_ORDER)
if not _LIGHTGBM_AVAILABLE and 'lightgbm' in AVAILABLE_MODELS:
    AVAILABLE_MODELS.remove('lightgbm')


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


def _build_lgbm_kwargs(scale_pos: float) -> dict:
    kwargs = dict(LIGHTGBM_PARAMS)
    kwargs.pop('class_weight', None)
    kwargs['is_unbalance'] = True
    return kwargs


def _stratified_subset(
    X: pd.DataFrame, y: pd.Series, max_size: int, seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create a stratified subset preserving class proportions.

    Always logs the original/sampled population, the subsampling method,
    and the seed (Sections 28 of the experiment spec).
    """
    n_original = len(X)
    if len(X) <= max_size:
        logger.info(
            "SVM subset: applied NO — population %d <= limit %d "
            "(method=stratified_keep_all, seed=%d)",
            n_original, max_size, seed,
        )
        return X, y

    frac = max_size / n_original
    combined = pd.DataFrame(X).assign(_label=y.values)
    sampled = (
        combined
        .groupby('_label', group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=seed))
    )
    X_sub = sampled.drop(columns=['_label'])
    y_sub = sampled['_label']
    n_subset = len(X_sub)
    ratio = n_subset / n_original
    logger.info(
        "SVM subset: applied YES — %d → %d samples (ratio: %.2f, "
        "method=grouped_stratified_sample, seed=%d)",
        n_original, n_subset, ratio, seed,
    )
    return X_sub, y_sub


def _fit_timed(model: object, X: pd.DataFrame, y: pd.Series, **fit_kwargs) -> object:
    """Fit a model and attach a monotonic training-time attribute."""
    start = time.perf_counter()
    model.fit(X, y, **fit_kwargs)
    model._train_time_sec = float(time.perf_counter() - start)
    return model


def train_models(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    model_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Train the requested classifiers (default: all five).

    Each trained model carries a ``_train_time_sec`` attribute with its
    wall-clock training time.  LightGBM is mandatory — if requested but
    unavailable, a RuntimeError is raised (never a silent skip).

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val : optional validation set (used for early stopping)
    model_names : list of str, optional
        Subset of MODEL_ORDER to train.  None → all five.
    """
    if model_names is None:
        model_names = list(MODEL_ORDER)

    unknown = [m for m in model_names if m not in MODEL_ORDER]
    if unknown:
        raise ValueError(f"Unknown model name(s): {unknown}")

    models: Dict[str, object] = {}
    scale_pos = _compute_scale_pos_weight(y_train)
    n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    logger.info("Training distribution — neg: %d, pos: %d, ratio: %.2f",
                 n_neg, n_pos, scale_pos)

    for name in model_names:
        # ── Logistic Regression ──────────────────────────────────────
        if name == 'logistic_regression':
            logger.info("Training LogisticRegression …")
            lr = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
            models[name] = _fit_timed(lr, X_train, y_train)

        # ── Random Forest ────────────────────────────────────────────
        elif name == 'random_forest':
            logger.info("Training RandomForest …")
            rf = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
            models[name] = _fit_timed(rf, X_train, y_train)

        # ── XGBoost ──────────────────────────────────────────────────
        elif name == 'xgboost':
            logger.info("Training XGBoost …")
            xgb_params = _build_xgb_kwargs()
            xgb_params['scale_pos_weight'] = scale_pos
            if X_val is not None and y_val is not None:
                xgb_params['early_stopping_rounds'] = 10
                model = XGBClassifier(**xgb_params)
                model = _fit_timed(
                    model, X_train, y_train,
                    eval_set=[(X_val, y_val)], verbose=False,
                )
                models[name] = model
            else:
                model = XGBClassifier(**xgb_params)
                models[name] = _fit_timed(model, X_train, y_train)

        # ── LightGBM (mandatory — no silent skip) ────────────────────
        elif name == 'lightgbm':
            if not _LIGHTGBM_AVAILABLE:
                raise RuntimeError(
                    "LightGBM requested but not installed. "
                    "Install with: pip install lightgbm"
                )
            logger.info("Training LightGBM …")
            lgbm_params = _build_lgbm_kwargs(scale_pos)
            lgbm_model = LGBMClassifier(**lgbm_params)
            if X_val is not None and y_val is not None:
                lgbm_model = _fit_timed(
                    lgbm_model, X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        __import__('lightgbm').early_stopping(10, verbose=False),
                        __import__('lightgbm').log_evaluation(-1),
                    ],
                )
                models[name] = lgbm_model
            else:
                models[name] = _fit_timed(lgbm_model, X_train, y_train)

        # ── SVM ──────────────────────────────────────────────────────
        elif name == 'svm':
            logger.info("Training SVM …")
            X_svm, y_svm = _stratified_subset(
                X_train, y_train, SVM_SUBSET_SIZE, RANDOM_SEED,
            )
            svm_params = dict(SVM_PARAMS)
            svm_model = SVC(**svm_params)
            models[name] = _fit_timed(svm_model, X_svm, y_svm)

    logger.info("All %d requested models trained successfully", len(models))
    return models
