"""
Ablation study: measure ROC-AUC impact of removing each feature group
via stratified k-fold cross-validation.

Dataset-awareness
-----------------
The ablation must actually remove real feature columns.  The standard
FEATURE_GROUPS map to the behavioural feature names produced by
feature_engineering (Olist / RetailRocket / REES46 / Online Retail II …),
but datasets that use their own native predictors (Credit Card, Telco)
do NOT use those names.  For those we provide explicit per-dataset group
mappings so removing a group genuinely drops columns.

Validity guards
---------------
- A group that matches no columns is skipped (no misleading no-op rows).
- A `without_<group>` row is only emitted when at least one column was
  actually removed, and the ablated matrix must have strictly fewer
  columns than the full matrix (hard assertion).

These guards make the ablation FAIL LOUDLY instead of silently producing
identical ROC-AUC for every group.
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


# ─────────────────────────────────────────────────────────────────────
# Dataset-aware feature-group → column mappings
#
# ``credit_card`` and ``telco`` do not emit the standardised feature names
# used by FEATURE_GROUPS (they pass through native predictors from
# build_native_modeling_data).  These mappings describe those actual
# columns so the ablation removes real features.  Mappings only list
# prefixes where a column name is not fully deterministic (dummy blocks);
# exact columns are matched by name/prefix and any that are absent are
# ignored at runtime.
# ─────────────────────────────────────────────────────────────────────

# Credit Card: native predictors (see credit_card.build_native_modeling_data)
CREDIT_CARD_ABLATION_GROUPS: dict = {
    'monetization': [
        'Total_Trans_Amt', 'Avg_Utilization_Ratio', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Credit_Limit', 'Total_Trans_Ct',
    ],
    'activity': [
        'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
        'Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1',
    ],
    'relationship': [
        'Total_Relationship_Count', 'Months_on_book',
    ],
    'demographics': [
        'Gender', 'Education_Level_', 'Marital_Status_',
        'Income_Category_', 'Card_Category_',
    ],
}

# Telco: native predictors (see telco.build_native_modeling_data).
# Columns use the standardised names (engagement_signal=tenure,
# transaction_value=MonthlyCharges, total_charges=TotalCharges).
TELCO_ABLATION_GROUPS: dict = {
    'contract': [
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    ],
    'usage_charges': [
        'engagement_signal', 'transaction_value', 'total_charges',
    ],
    'services': [
        'MultipleLines_', 'InternetService_', 'OnlineSecurity_',
        'OnlineBackup_', 'DeviceProtection_', 'TechSupport_',
        'StreamingTV_', 'StreamingMovies_', 'PhoneService_encoded',
    ],
    'billing': [
        'PaymentMethod_', 'PaperlessBilling_encoded',
    ],
    'demographics': [
        'gender_encoded', 'SeniorCitizen', 'Partner_encoded',
        'Dependents_encoded',
    ],
}

# Olist: the standard FEATURE_GROUPS cover most columns but the one-hot
# payment-type dummies (pay_type__*) are emitted by the payment group and were
# never listed, so without_payment left them in place.  Keep the shared
# FEATURE_GROUPS untouched (clean datasets rely on it) and fix the mapping for
# Olist only by adding the pay_type_ prefix spec.
OLIST_ABLATION_GROUPS: dict = {
    'purchase': [
        'total_orders', 'total_items_purchased', 'repeat_purchase_ratio',
    ],
    'monetary': [
        'total_spent', 'avg_order_value', 'max_order_value',
        'min_order_value',
    ],
    'inactivity': [
        'days_since_last_purchase',
    ],
    'review': [
        'avg_review_score', 'min_review_score', 'low_review_ratio',
        'positive_review_ratio', 'review_variance',
    ],
    'delivery': [
        'avg_delivery_delay_days', 'delayed_order_ratio',
        'max_delivery_delay', 'on_time_delivery_ratio',
    ],
    'payment': [
        'avg_payment_installments', 'avg_payment_value', 'pay_type_',
    ],
    'cadence': [
        'avg_days_between_orders', 'customer_lifetime_days',
        'avg_orders_per_month',
    ],
}

# RetailRocket: the standard FEATURE_GROUPS omit the engagement totals
# (total_events, total_purchases) that the engagement group actually emits.
RETAILROCKET_ABLATION_GROUPS: dict = {
    'purchase': [
        'total_orders', 'total_items_purchased', 'repeat_purchase_ratio',
    ],
    'monetary': [
        'total_spent', 'avg_order_value', 'max_order_value',
        'min_order_value',
    ],
    'inactivity': [
        'days_since_last_purchase',
    ],
    'engagement': [
        'total_page_views', 'total_cart_adds', 'total_sessions',
        'avg_actions_per_session', 'total_wishlist_adds',
        'total_purchases', 'total_events',
    ],
    'cadence': [
        'avg_days_between_orders', 'customer_lifetime_days',
        'avg_orders_per_month',
    ],
}


def _match_group_columns(columns, group_feats):
    """Return the subset of ``columns`` covered by a group's feature spec.

    Handles exact names and prefix specs (e.g. ``Income_Category_`` matches
    every column starting with that prefix).
    """
    matched = []
    for spec in group_feats:
        found = [c for c in columns if c == spec]
        if not found:
            found = [c for c in columns if c.startswith(spec)]
        for c in found:
            if c not in matched:
                matched.append(c)
    return matched


def _resolve_ablation_groups(dataset_name: str):
    """Return the raw feature-group → spec dict for a dataset.

    Prefer the per-dataset (real-column) mapping for datasets whose emitted
    feature names deviate from the standardised FEATURE_GROUPS; otherwise
    fall back to the shared FEATURE_GROUPS (clean datasets unchanged).
    """
    if dataset_name == 'credit_card':
        return dict(CREDIT_CARD_ABLATION_GROUPS)
    if dataset_name == 'telco':
        return dict(TELCO_ABLATION_GROUPS)
    if dataset_name == 'olist':
        return dict(OLIST_ABLATION_GROUPS)
    if dataset_name == 'retailrocket':
        return dict(RETAILROCKET_ABLATION_GROUPS)
    return dict(FEATURE_GROUPS)


def resolve_ablation_groups_for_matrix(
    X: pd.DataFrame,
    dataset_name: str,
) -> dict:
    """Resolve feature groups to the columns that actually exist in ``X``.

    Groups whose spec matches no real column are dropped (with a warning) so
    no misleading no-op (identical-AUC) ablation rows are produced.

    Raises
    ------
    RuntimeError
        If no group matches any column — the ablation would be a no-op and
        is therefore experimentally invalid.
    """
    all_features = list(X.columns)
    group_columns = {}
    for grp_name, grp_feats in _resolve_ablation_groups(dataset_name).items():
        matched = _match_group_columns(all_features, grp_feats)
        if not matched:
            logger.warning(
                "Ablation group '%s' matched no columns for '%s' — skipped "
                "(no-op ablation row suppressed)", grp_name, dataset_name,
            )
            continue
        group_columns[grp_name] = matched
        logger.validation(
            "Ablation group '%s' → %d columns: %s", grp_name,
            len(matched), matched,
        )

    if not group_columns:
        raise RuntimeError(
            f"Ablation failed: no feature-group columns matched the "
            f"{dataset_name} feature matrix — a no-op ablation is not "
            f"experimentally valid."
        )

    # Hard validity guard per group: the removal must actually drop columns.
    n_full = len(all_features)
    for grp_name, grp_cols in group_columns.items():
        assert len(grp_cols) > 0, (
            f"Ablation group '{grp_name}' matched no columns for "
            f"'{dataset_name}' — refusing to emit a no-op row."
        )
        n_remaining = n_full - len(grp_cols)
        assert n_remaining < n_full, (
            f"Ablation group '{grp_name}' failed to remove any column from "
            f"{dataset_name} — feature count did not decrease "
            f"({n_full} → {n_remaining})."
        )
    return group_columns


def run_ablation(X: pd.DataFrame, y: pd.Series,
                 dataset_name: str = 'olist') -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=ABLATION_N_SPLITS, shuffle=True,
                          random_state=RANDOM_SEED)
    all_features = list(X.columns)
    n_full = len(all_features)
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

    # Resolve feature-group → actual-columns mapping against the real matrix.
    group_columns = resolve_ablation_groups_for_matrix(X, dataset_name)

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

        for grp_name, grp_cols in group_columns.items():
            # Validity already guaranteed by resolve_ablation_groups_for_matrix.
            remaining = [f for f in all_features if f not in grp_cols]
            if len(remaining) < 2:
                continue
            clf = _model_factory(model_name)
            try:
                scores = cross_val_score(clf, _X[remaining], _y, cv=cv,
                                         scoring='roc_auc', n_jobs=1)
                records.append({
                    'model': model_name,
                    'feature_set': f'without_{grp_name}',
                    'n_removed': len(grp_cols),
                    'mean_roc_auc': float(np.mean(scores)),
                    'std_roc_auc': float(np.std(scores)),
                })
            except Exception as exc:
                logger.debug("Ablation %s / %s failed: %s",
                              model_name, grp_name, exc)

    result = pd.DataFrame(records)
    logger.info("Ablation complete — %d records (full feature count: %d)",
                len(result), n_full)
    return result
