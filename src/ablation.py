"""
Ablation study: measure ROC-AUC impact of removing each feature group
via stratified k-fold cross-validation.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import (
    FEATURE_GROUPS, RANDOM_SEED, LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS, XGBOOST_PARAMS, LIGHTGBM_PARAMS, ABLATION_N_SPLITS,
    ABLATION_MAX_ROWS, ABLATION_ESTIMATORS,
)
from src.utils import get_logger

logger = get_logger(__name__)

try:
    import lightgbm as lgb
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    _LIGHTGBM_AVAILABLE = False


def _model_factory(name: str):
    if name == 'logistic_regression':
        return LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
    if name == 'random_forest':
        params = dict(RANDOM_FOREST_PARAMS)
        params['n_estimators'] = ABLATION_ESTIMATORS
        return RandomForestClassifier(**params)
    if name == 'xgboost':
        params = dict(XGBOOST_PARAMS)
        params.pop('scale_pos_weight', None)
        params['n_estimators'] = ABLATION_ESTIMATORS
        return XGBClassifier(**params)
    if name == 'lightgbm':
        if not _LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm is not installed")
        params = dict(LIGHTGBM_PARAMS)
        params['n_estimators'] = ABLATION_ESTIMATORS
        return lgb.LGBMClassifier(**params)
    raise ValueError(f"Unknown model: {name}")


def run_ablation(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=ABLATION_N_SPLITS, shuffle=True,
                          random_state=RANDOM_SEED)
    all_features = list(X.columns)
    model_names = ['logistic_regression', 'random_forest', 'xgboost']
    if _LIGHTGBM_AVAILABLE:
        model_names.append('lightgbm')
    records = []

    # Bound the CV cost on very large cohorts: the study is unchanged, it just
    # runs on a seeded subsample (same precedent as SVM / segmentation caps).
    if len(X) > ABLATION_MAX_ROWS:
        _X = X.sample(n=ABLATION_MAX_ROWS, random_state=RANDOM_SEED)
        _y = y.loc[_X.index]
        logger.info("Ablation on %d-row seeded subsample (full cohort: %d)",
                    ABLATION_MAX_ROWS, len(X))
    else:
        _X, _y = X, y

    for model_name in model_names:
        logger.info("Ablation — %s", model_name)

        clf_all = _model_factory(model_name)
        try:
            scores = cross_val_score(clf_all, _X, _y, cv=cv, scoring='roc_auc',
                                     n_jobs=1)
            records.append({
                'model': model_name,
                'feature_set': 'all_features',
                'mean_roc_auc': float(np.mean(scores)),
                'std_roc_auc': float(np.std(scores)),
            })
        except Exception as exc:
            logger.warning("Ablation failed for %s all features: %s",
                            model_name, exc)

        for grp_name, grp_feats in FEATURE_GROUPS.items():
            remaining = [f for f in all_features if f not in grp_feats]
            if len(remaining) < 2:
                continue
            clf = _model_factory(model_name)
            try:
                scores = cross_val_score(clf, _X[remaining], _y, cv=cv,
                                         scoring='roc_auc', n_jobs=1)
                records.append({
                    'model': model_name,
                    'feature_set': f'without_{grp_name}',
                    'mean_roc_auc': float(np.mean(scores)),
                    'std_roc_auc': float(np.std(scores)),
                })
            except Exception as exc:
                logger.debug("Ablation %s / %s failed: %s",
                              model_name, grp_name, exc)

    result = pd.DataFrame(records)
    logger.info("Ablation complete — %d records", len(result))
    return result
