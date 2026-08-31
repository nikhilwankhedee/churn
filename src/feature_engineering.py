"""
Customer-level behavioural feature engineering on the standardised schema.

All features are computed from events strictly BEFORE a snapshot date.
No data from or after the snapshot is ever used — zero leakage.

Feature groups are modular and independently activated:
    purchase, monetary, inactivity, review, delivery, payment,
    engagement, cadence

Each group function checks whether the required standardised columns exist
in the input DataFrame.  If not, it logs the gap and returns an empty
(placeholder) result.  This allows the same code to work across datasets
with heterogeneous signal availability.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Callable

from src.config import PAYMENT_DUMMY_PREFIX, STANDARD_FEATURE_GROUPS
from src.utils import get_logger, assert_no_nans

logger = get_logger(__name__)

# Standardised column names expected by the feature engineering functions
STD_CUSTOMER_ID = 'customer_id'
STD_EVENT_TIME = 'event_time'
STD_TRANSACTION_VALUE = 'transaction_value'
STD_EVENT_TYPE = 'event_type'
STD_REVIEW_SCORE = 'review_score'
STD_PAYMENT_TYPE = 'payment_type'
STD_DELIVERY_DELAY = 'delivery_delay'
STD_ENGAGEMENT_SIGNAL = 'engagement_signal'
STD_SESSION_ID = 'session_id'
STD_PRODUCT_ID = 'product_id'

PURCHASE_EVENT_TYPES = {'purchase', 'transaction', 'order'}


# ── Group-level helpers ──────────────────────────────────────────────

def _check_columns(df: pd.DataFrame, required: List[str],
                   group_name: str) -> bool:
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.info(
            "Feature group '%s' not available — missing columns: %s",
            group_name, missing,
        )
        return False
    return True


def _empty_features(customer_ids: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({'customer_unique_id': customer_ids}).set_index(
        'customer_unique_id')


def _purchase_events(df: pd.DataFrame) -> pd.DataFrame:
    if STD_EVENT_TYPE in df.columns:
        return df[df[STD_EVENT_TYPE].isin(PURCHASE_EVENT_TYPES)].copy()
    # If no event_type column, assume all rows are purchase events
    return df.copy()


# ── Feature group: purchase ──────────────────────────────────────────

def _engineer_purchase(
    df: pd.DataFrame, snapshot_date: pd.Timestamp,
    customer_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    if not _check_columns(df, [STD_CUSTOMER_ID, STD_EVENT_TIME], 'purchase'):
        return pd.DataFrame()

    purchases = _purchase_events(df)
    if purchases.empty:
        logger.warning("No purchase events found — purchase features empty")
        return pd.DataFrame()

    if customer_ids is not None:
        purchases = purchases[purchases[STD_CUSTOMER_ID].isin(customer_ids)]

    grp = purchases.groupby(STD_CUSTOMER_ID)
    result = grp.agg(
        total_orders=(STD_EVENT_TIME, 'nunique') if STD_EVENT_TIME in purchases
                      else (STD_CUSTOMER_ID, 'count'),
        total_items_purchased=(STD_PRODUCT_ID, 'count')
                              if STD_PRODUCT_ID in purchases else (STD_EVENT_TIME, 'count'),
        first_purchase=(STD_EVENT_TIME, 'min'),
        last_purchase=(STD_EVENT_TIME, 'max'),
    )

    # Ensure column names regardless of aggregation source
    result = result.rename(columns={
        'total_orders': 'total_orders',
    }, errors='ignore')
    if 'total_orders' not in result.columns:
        result['total_orders'] = result.get(STD_EVENT_TIME, len(result))

    result['repeat_purchase_ratio'] = (
        result['total_orders'] > 1
    ).astype(np.float64)

    return result


# ── Feature group: monetary ──────────────────────────────────────────

def _engineer_monetary(
    df: pd.DataFrame, snapshot_date: pd.Timestamp,
    customer_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    if not _check_columns(df, [STD_CUSTOMER_ID, STD_TRANSACTION_VALUE],
                          'monetary'):
        return pd.DataFrame()

    purchases = _purchase_events(df)
    if purchases.empty:
        return pd.DataFrame()

    if customer_ids is not None:
        purchases = purchases[purchases[STD_CUSTOMER_ID].isin(customer_ids)]

    order_values = purchases.groupby(
        [STD_CUSTOMER_ID, STD_EVENT_TIME], as_index=False
    ).agg(
        order_value=(STD_TRANSACTION_VALUE, 'sum'),
    )
    monetary = order_values.groupby(STD_CUSTOMER_ID).agg(
        total_spent=('order_value', 'sum'),
        avg_order_value=('order_value', 'mean'),
        max_order_value=('order_value', 'max'),
        min_order_value=('order_value', 'min'),
    ).fillna(0.0)

    return monetary


# ── Feature group: inactivity ────────────────────────────────────────

def _engineer_inactivity(
    df: pd.DataFrame, snapshot_date: pd.Timestamp,
    customer_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    if not _check_columns(df, [STD_CUSTOMER_ID, STD_EVENT_TIME], 'inactivity'):
        return pd.DataFrame()

    purchases = _purchase_events(df)
    if purchases.empty:
        return pd.DataFrame()

    if customer_ids is not None:
        purchases = purchases[purchases[STD_CUSTOMER_ID].isin(customer_ids)]

    last_purchase = purchases.groupby(STD_CUSTOMER_ID)[STD_EVENT_TIME].max()
    days_since = (snapshot_date - last_purchase).dt.days.clip(lower=0)

    result = pd.DataFrame(
        {'days_since_last_purchase': days_since},
        index=last_purchase.index,
    )
    return result


# ── Feature group: review ────────────────────────────────────────────

def _engineer_review(
    df: pd.DataFrame, snapshot_date: pd.Timestamp,
    customer_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    if not _check_columns(df, [STD_CUSTOMER_ID, STD_REVIEW_SCORE], 'review'):
        return pd.DataFrame()

    has_review = df[df[STD_REVIEW_SCORE].notna() & (df[STD_REVIEW_SCORE] > 0)]
    if has_review.empty:
        logger.info("No non-zero review scores — review features unavailable")
        return pd.DataFrame()

    if customer_ids is not None:
        has_review = has_review[has_review[STD_CUSTOMER_ID].isin(customer_ids)]

    grp = has_review.groupby(STD_CUSTOMER_ID)
    review = grp.agg(
        avg_review_score=(STD_REVIEW_SCORE, 'mean'),
        min_review_score=(STD_REVIEW_SCORE, 'min'),
        review_variance=(STD_REVIEW_SCORE, 'var'),
    )
    review['low_review_ratio'] = grp[STD_REVIEW_SCORE].apply(
        lambda x: (x <= 3).mean() if len(x) > 0 else 0.0,
    )
    review['positive_review_ratio'] = grp[STD_REVIEW_SCORE].apply(
        lambda x: (x >= 4).mean() if len(x) > 0 else 0.0,
    )
    review = review.fillna(0.0)
    review['review_variance'] = review['review_variance'].fillna(0.0)

    return review


# ── Feature group: delivery ──────────────────────────────────────────

def _engineer_delivery(
    df: pd.DataFrame, snapshot_date: pd.Timestamp,
    customer_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    if not _check_columns(df, [STD_CUSTOMER_ID, STD_DELIVERY_DELAY],
                          'delivery'):
        return pd.DataFrame()

    purchases = _purchase_events(df)
    if purchases.empty:
        return pd.DataFrame()

    if customer_ids is not None:
        purchases = purchases[purchases[STD_CUSTOMER_ID].isin(customer_ids)]

    delay = purchases[STD_DELIVERY_DELAY].fillna(0).astype(float)
    purchases = purchases.copy()
    purchases['_delay'] = delay

    grp = purchases.groupby(STD_CUSTOMER_ID)
    delivery = grp.agg(
        avg_delivery_delay_days=('_delay', 'mean'),
        max_delivery_delay=('_delay', 'max'),
        delayed_order_ratio=('_delay', lambda x: (x > 0).mean()),
        on_time_delivery_ratio=('_delay', lambda x: (x <= 0).mean()),
    ).fillna(0.0)

    return delivery


# ── Feature group: payment ───────────────────────────────────────────

def _engineer_payment(
    df: pd.DataFrame, snapshot_date: pd.Timestamp,
    customer_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    if not _check_columns(df, [STD_CUSTOMER_ID, STD_PAYMENT_TYPE],
                          'payment'):
        return pd.DataFrame()

    purchases = _purchase_events(df)
    if purchases.empty:
        return pd.DataFrame()

    if customer_ids is not None:
        purchases = purchases[purchases[STD_CUSTOMER_ID].isin(customer_ids)]

    pay_hist = purchases.drop_duplicates(
        subset=[STD_CUSTOMER_ID, STD_EVENT_TIME]
    )
    grp = pay_hist.groupby(STD_CUSTOMER_ID)

    payment = pd.DataFrame(index=pay_hist[STD_CUSTOMER_ID].unique())
    payment.index.name = STD_CUSTOMER_ID

    if STD_TRANSACTION_VALUE in purchases.columns:
        pay_val = grp.agg(
            avg_payment_value=(STD_TRANSACTION_VALUE, 'mean'),
        ).fillna(0.0)
        payment['avg_payment_value'] = pay_val['avg_payment_value']
    else:
        payment['avg_payment_value'] = 0.0

    # Preferred payment type
    if STD_PAYMENT_TYPE in pay_hist.columns:
        pref = grp[STD_PAYMENT_TYPE].apply(
            lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown',
        )
        dummies = pd.get_dummies(
            pref, prefix=PAYMENT_DUMMY_PREFIX,
        ).astype(np.float64)
        payment = pd.concat([payment, dummies], axis=1)

    payment = payment.fillna(0.0)

    # Add avg_payment_installments placeholder (only Olist has this)
    payment['avg_payment_installments'] = 0.0

    return payment


# ── Feature group: engagement ────────────────────────────────────────

def _engineer_engagement(
    df: pd.DataFrame, snapshot_date: pd.Timestamp,
    customer_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    if not _check_columns(df, [STD_CUSTOMER_ID, STD_EVENT_TYPE],
                          'engagement'):
        return pd.DataFrame()

    events = df.copy()
    if customer_ids is not None:
        events = events[events[STD_CUSTOMER_ID].isin(customer_ids)]

    grp = events.groupby(STD_CUSTOMER_ID)

    engagement = pd.DataFrame(index=events[STD_CUSTOMER_ID].unique())
    engagement.index.name = STD_CUSTOMER_ID

    # Total events by type
    if STD_EVENT_TYPE in events.columns:
        type_counts = events.groupby(
            [STD_CUSTOMER_ID, STD_EVENT_TYPE]
        ).size().unstack(fill_value=0)

        engagement['total_page_views'] = type_counts.get('view', 0)
        engagement['total_cart_adds'] = type_counts.get('cart_add', 0)
        engagement['total_purchases'] = type_counts.get('purchase', 0)
        engagement['total_wishlist_adds'] = type_counts.get('wishlist', 0)

        # Total engagement events (all types)
        engagement['total_events'] = type_counts.sum(axis=1)

    # Session-level metrics
    if STD_SESSION_ID in events.columns and STD_EVENT_TYPE in events.columns:
        sessions = events.drop_duplicates(
            subset=[STD_CUSTOMER_ID, STD_SESSION_ID]
        )
        session_counts = sessions.groupby(STD_CUSTOMER_ID).size()
        engagement['total_sessions'] = session_counts

        actions_per_session = (
            engagement['total_events'] / engagement['total_sessions'].clip(lower=1)
        )
        engagement['avg_actions_per_session'] = actions_per_session.fillna(0.0)
    else:
        engagement['total_sessions'] = 0
        engagement['avg_actions_per_session'] = 0.0

    if STD_ENGAGEMENT_SIGNAL in events.columns:
        eng_signal = grp[STD_ENGAGEMENT_SIGNAL].mean()
        engagement['avg_engagement_signal'] = eng_signal.fillna(0.0)

    engagement = engagement.fillna(0.0)
    return engagement


# ── Feature group: cadence ───────────────────────────────────────────

def _engineer_cadence(
    df: pd.DataFrame, snapshot_date: pd.Timestamp,
    customer_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    if not _check_columns(df, [STD_CUSTOMER_ID, STD_EVENT_TIME], 'cadence'):
        return pd.DataFrame()

    purchases = _purchase_events(df)
    if purchases.empty:
        return pd.DataFrame()

    if customer_ids is not None:
        purchases = purchases[purchases[STD_CUSTOMER_ID].isin(customer_ids)]

    grp = purchases.groupby(STD_CUSTOMER_ID)
    timeline = grp.agg(
        first_purchase=(STD_EVENT_TIME, 'min'),
        last_purchase=(STD_EVENT_TIME, 'max'),
        n_orders=(STD_EVENT_TIME, 'nunique'),
    )

    lifetime = (timeline['last_purchase'] - timeline['first_purchase']).dt.days
    cadence = pd.DataFrame(index=timeline.index)

    cadence['customer_lifetime_days'] = lifetime.clip(lower=0)
    cadence['avg_days_between_orders'] = np.clip(
        np.where(
            timeline['n_orders'] > 1,
            cadence['customer_lifetime_days'] / (timeline['n_orders'] - 1),
            0.0,
        ),
        a_min=0, a_max=None,
    )

    lifetime_months = cadence['customer_lifetime_days'] / 30.44
    cadence['avg_orders_per_month'] = np.where(
        lifetime_months > 0,
        timeline['n_orders'] / lifetime_months,
        0.0,
    )

    return cadence


# ── Registry of feature group builders ───────────────────────────────

FEATURE_GROUP_BUILDERS: Dict[str, Callable] = {
    'purchase': _engineer_purchase,
    'monetary': _engineer_monetary,
    'inactivity': _engineer_inactivity,
    'review': _engineer_review,
    'delivery': _engineer_delivery,
    'payment': _engineer_payment,
    'engagement': _engineer_engagement,
    'cadence': _engineer_cadence,
}


# ── Main entry point ─────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    customer_ids: Optional[List[str]] = None,
    available_groups: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build a customer × feature matrix using only events < snapshot_date.

    Parameters
    ----------
    df : pd.DataFrame
        Data with standardised schema columns (customer_id, event_time, …).
    snapshot_date : pd.Timestamp
        Only events strictly before this date are used.
    customer_ids : list of str, optional
        If provided, only features for these customers are computed.
    available_groups : list of str, optional
        Which feature groups to build.  If None, all groups are attempted.

    Returns
    -------
    pd.DataFrame with customers as index, feature names as columns.
    """
    hist = df[df[STD_EVENT_TIME] < snapshot_date].copy()
    n_hist = len(hist)
    logger.info(
        "Engineering features up to %s (historical rows: %d, groups: %s)",
        snapshot_date.date(), n_hist,
        available_groups or 'all',
    )

    assert (hist[STD_EVENT_TIME] < snapshot_date).all(), \
        "Future timestamps in historical slice!"

    if customer_ids is not None:
        customer_set = set(customer_ids)
        hist = hist[hist[STD_CUSTOMER_ID].isin(customer_set)].copy()

    if hist.empty:
        logger.warning("No historical data for snapshot %s", snapshot_date.date())
        return pd.DataFrame()

    if available_groups is None:
        available_groups = list(FEATURE_GROUP_BUILDERS.keys())

    # Build each active feature group independently
    group_results = {}
    for group in available_groups:
        if group in FEATURE_GROUP_BUILDERS:
            try:
                result = FEATURE_GROUP_BUILDERS[group](
                    hist, snapshot_date, customer_ids,
                )
                if result is not None and not result.empty:
                    group_results[group] = result
                    logger.debug("Feature group '%s': %d customers, %d features",
                                  group, result.shape[0], result.shape[1])
                else:
                    logger.info("Feature group '%s' produced no features", group)
            except Exception as exc:
                logger.warning(
                    "Feature group '%s' failed: %s", group, exc,
                )
        else:
            logger.warning("Unknown feature group '%s' — skipping", group)

    if not group_results:
        logger.error("No feature groups produced any features")
        return pd.DataFrame()

    # Merge all group results on customer_id index
    features = None
    for group_name, result_df in group_results.items():
        if features is None:
            features = result_df
        else:
            features = features.join(result_df, how='outer')

    if features is None or features.empty:
        return pd.DataFrame()

    # Drop helper columns
    features = features.drop(
        columns=['first_purchase', 'last_purchase', STD_EVENT_TIME,
                 STD_CUSTOMER_ID],
        errors='ignore',
    )

    features = features.fillna(0.0)
    assert_no_nans(features, "Feature matrix")
    logger.info("Feature matrix: %d customers × %d features",
                 features.shape[0], features.shape[1])
    return features
