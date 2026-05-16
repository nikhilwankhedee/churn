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
    RANDOM_FOREST_PARAMS, XGBOOST_PARAMS, ABLATION_N_SPLITS,
)
from src.utils import get_logger

logger = get_logger(__name__)


def _model_factory(name: str):
    if name == 'logistic_regression':
        return LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
    if name == 'random_forest':
        return RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    if name == 'xgboost':
        params = dict(XGBOOST_PARAMS)
        params.pop('scale_pos_weight', None)
        return XGBClassifier(**params)
    raise ValueError(f"Unknown model: {name}")


def run_ablation(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    if y.nunique() < 2:
        logger.warning("Ablation skipped — only %d class(es) in labels", y.nunique())
        return pd.DataFrame()

    cv = StratifiedKFold(n_splits=ABLATION_N_SPLITS, shuffle=True,
                          random_state=RANDOM_SEED)
    all_features = list(X.columns)
    model_names = ['logistic_regression', 'random_forest', 'xgboost']
    records = []

    for model_name in model_names:
        logger.info("Ablation — %s", model_name)

        clf_all = _model_factory(model_name)
        try:
            scores = cross_val_score(clf_all, X, y, cv=cv, scoring='roc_auc',
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
                scores = cross_val_score(clf, X[remaining], y, cv=cv,
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
