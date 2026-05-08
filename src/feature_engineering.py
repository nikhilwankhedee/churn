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

# Adapters emit static per-customer attributes under this prefix.
STATIC_PREFIX = 'static_'

PURCHASE_EVENT_TYPES = {'purchase', 'transaction', 'order', 'subscription_event'}


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

    cadence['customer_lifetime_days'] = np.clip(lifetime, 0, None)
    cadence['avg_days_between_orders'] = np.clip(
        np.where(
            timeline['n_orders'] > 1,
            cadence['customer_lifetime_days'] / (timeline['n_orders'] - 1),
            0.0,
        ), 0, None,
    )

    lifetime_months = cadence['customer_lifetime_days'] / 30.44
    cadence['avg_orders_per_month'] = np.where(
        lifetime_months > 0,
        timeline['n_orders'] / lifetime_months,
        0.0,
    )

    return cadence


# ── Feature group: static ────────────────────────────────────────────

def _engineer_static(
    df: pd.DataFrame, snapshot_date: pd.Timestamp,
    customer_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Per-customer static attributes (e.g. Telco tenure, charges, dummies).

    Columns are identified by the 'static_' prefix and aggregated with
    first() per customer (they are constant within a customer).
    """
    if STD_CUSTOMER_ID not in df.columns:
        return pd.DataFrame()
    static_cols = [c for c in df.columns if c.startswith(STATIC_PREFIX)]
    if not static_cols:
        logger.info(
            "Feature group 'static' — no '%s*' columns present", STATIC_PREFIX,
        )
        return pd.DataFrame()

    events = df.copy()
    if customer_ids is not None:
        events = events[events[STD_CUSTOMER_ID].isin(customer_ids)]
    if events.empty:
        return pd.DataFrame()

    static = events.groupby(STD_CUSTOMER_ID)[static_cols].first()
    static = static.astype(np.float64, errors='ignore')
    return static


# ── Feature group: listening (e.g. Last.fm) ──────────────────────────

def _engineer_listening(
    df: pd.DataFrame, snapshot_date: pd.Timestamp,
    customer_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Listening behaviour features: volume, diversity, cadence, recency.

    Expected standardised columns: customer_id, event_time, product_id
    (artist), track, event_type='purchase' (listening events mapped to
    purchase so the generic event machinery can be reused).
    """
    if not _check_columns(df, [STD_CUSTOMER_ID, STD_EVENT_TIME], 'listening'):
        return pd.DataFrame()

    events = df.copy()
    if customer_ids is not None:
        events = events[events[STD_CUSTOMER_ID].isin(customer_ids)]
    if events.empty:
        return pd.DataFrame()

    events['_day'] = events[STD_EVENT_TIME].dt.normalize()
    grp = events.groupby(STD_CUSTOMER_ID)

    listening = pd.DataFrame(index=events[STD_CUSTOMER_ID].unique())
    listening.index.name = STD_CUSTOMER_ID

    total = events.groupby(STD_CUSTOMER_ID).size()
    listening['total_listens'] = total

    if STD_PRODUCT_ID in events.columns:
        listening['unique_artists'] = grp[STD_PRODUCT_ID].nunique()
    if 'track' in events.columns:
        listening['unique_tracks'] = grp['track'].nunique()

    listening['active_days'] = grp['_day'].nunique()
    active_days = listening['active_days'].clip(lower=1)
    listening['avg_listens_per_day'] = listening['total_listens'] / active_days

    last_listen = grp[STD_EVENT_TIME].max()
    listening['days_since_last_listen'] = (
        snapshot_date - last_listen
    ).dt.days.clip(lower=0)

    first_listen = grp[STD_EVENT_TIME].min()
    span_days = (last_listen - first_listen).dt.days.clip(lower=0) + 1
    listening['listening_frequency'] = listening['total_listens'] / span_days

    # Max gap between consecutive listening sessions (days)
    sorted_events = events.sort_values([STD_CUSTOMER_ID, STD_EVENT_TIME])
    sorted_events['_prev_time'] = sorted_events.groupby(STD_CUSTOMER_ID)[
        STD_EVENT_TIME
    ].shift(1)
    sorted_events['_gap_days'] = (
        sorted_events[STD_EVENT_TIME] - sorted_events['_prev_time']
    ).dt.days
    gap_max = sorted_events.groupby(STD_CUSTOMER_ID)['_gap_days'].max()
    listening['max_gap_between_sessions'] = gap_max.fillna(0.0)

    div = listening['unique_artists'] / listening['total_listens'].clip(lower=1)
    listening['artist_diversity_ratio'] = div

    listening = listening.fillna(0.0)
    return listening


# ── Feature group: kkbox ─────────────────────────────────────────────

def _engineer_kkbox(
    df: pd.DataFrame, snapshot_date: pd.Timestamp,
    customer_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """KKBox subscription + listening activity features.

    The KKBox adapter emits transaction events (event_type='purchase',
    kkbox_* pricing columns) and listening events (event_type='listen',
    kkbox_num_25/50/75/985/100, kkbox_num_unq, kkbox_total_secs).
    """
    if not _check_columns(df, [STD_CUSTOMER_ID, STD_EVENT_TIME], 'kkbox'):
        return pd.DataFrame()

    events = df.copy()
    if customer_ids is not None:
        events = events[events[STD_CUSTOMER_ID].isin(customer_ids)]
    if events.empty:
        return pd.DataFrame()

    feat = pd.DataFrame(index=events[STD_CUSTOMER_ID].unique())
    feat.index.name = STD_CUSTOMER_ID

    if STD_EVENT_TYPE in events.columns:
        trans = events[events[STD_EVENT_TYPE] == 'purchase']
    else:
        trans = events

    # ── Subscription / transaction features ──────────────────────────
    if not trans.empty:
        tgrp = trans.groupby(STD_CUSTOMER_ID)
        feat['total_transactions'] = trans.groupby(STD_CUSTOMER_ID).size()
        if 'kkbox_actual_amount_paid' in events.columns:
            feat['total_msno_paid'] = tgrp['kkbox_actual_amount_paid'].sum()
        else:
            feat['total_msno_paid'] = 0.0
        if 'kkbox_is_auto_renew' in events.columns:
            feat['is_auto_renew'] = tgrp['kkbox_is_auto_renew'].mean()
        else:
            feat['is_auto_renew'] = 0.0
        if 'kkbox_plan_list_price' in events.columns:
            feat['avg_plan_list_price'] = tgrp['kkbox_plan_list_price'].mean()
        else:
            feat['avg_plan_list_price'] = 0.0
        if 'kkbox_plan_id' in events.columns:
            feat['n_unique_plans'] = tgrp['kkbox_plan_id'].nunique()
        if 'kkbox_payment_method_id' in events.columns:
            feat['n_unique_payment_methods'] = tgrp[
                'kkbox_payment_method_id'
            ].nunique()
        if 'kkbox_payment_plan_days' in events.columns:
            feat['n_unique_payment_plan_days'] = tgrp[
                'kkbox_payment_plan_days'
            ].nunique()
            feat['avg_payment_plan_days'] = tgrp[
                'kkbox_payment_plan_days'
            ].mean()

    # ── Listening log features ───────────────────────────────────────
    if STD_EVENT_TYPE in events.columns:
        logs = events[events[STD_EVENT_TYPE] == 'listen'].copy()
    else:
        logs = pd.DataFrame()
    if not logs.empty:
        logs['_day'] = logs[STD_EVENT_TIME].dt.normalize()
        feat['total_log_days'] = logs.groupby(STD_CUSTOMER_ID)['_day'].nunique()
        feat['active_log_days'] = feat['total_log_days']

        for col in ['kkbox_num_25', 'kkbox_num_50', 'kkbox_num_75',
                    'kkbox_num_985', 'kkbox_num_100', 'kkbox_num_unq',
                    'kkbox_total_secs']:
            if col in logs.columns:
                feat[col.replace('kkbox_', 'total_')] = logs.groupby(
                    STD_CUSTOMER_ID
                )[col].sum()

        if 'kkbox_total_secs' in logs.columns:
            feat['total_seconds'] = logs.groupby(STD_CUSTOMER_ID)[
                'kkbox_total_secs'
            ].sum()

        num_cols = [c for c in ['kkbox_num_25', 'kkbox_num_50', 'kkbox_num_75',
                                'kkbox_num_985', 'kkbox_num_100', 'kkbox_num_unq']
                    if c in logs.columns]
        if num_cols:
            feat['avg_num_per_day'] = (
                logs.groupby(STD_CUSTOMER_ID)[num_cols].sum().sum(axis=1)
                / feat['active_log_days'].clip(lower=1)
            )
        if 'kkbox_total_secs' in logs.columns:
            feat['avg_seconds_per_day'] = (
                feat['total_seconds'] / feat['active_log_days'].clip(lower=1)
            )

        last_log = logs.groupby(STD_CUSTOMER_ID)[STD_EVENT_TIME].max()
        feat['days_since_last_listen'] = (
            snapshot_date - last_log
        ).dt.days.clip(lower=0)

    feat = feat.fillna(0.0)
    return feat


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
    'static': _engineer_static,
    'listening': _engineer_listening,
    'kkbox': _engineer_kkbox,
}


# ── Main entry point ─────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    customer_ids: Optional[List[str]] = None,
    available_groups: Optional[List[str]] = None,
    filter_by_snapshot: bool = True,
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
    filter_by_snapshot : bool, default True
        If False, skip the ``event_time < snapshot`` filter.  Used by
        datasets whose ``event_time`` is synthetic (no genuine temporal
        dimension) but whose features are per-customer static attributes
        (e.g. Telco).  Setting this False never widens the temporal window
        for datasets with real timestamps — callers must only disable it
        when ``adapter.has_temporal_data`` is False.

    Returns
    -------
    pd.DataFrame with customers as index, feature names as columns.
    """
    if filter_by_snapshot:
        hist = df[df[STD_EVENT_TIME] < snapshot_date].copy()
    else:
        hist = df.copy()
    n_hist = len(hist)
    logger.info(
        "Engineering features up to %s (historical rows: %d, groups: %s)",
        snapshot_date.date(), n_hist,
        available_groups or 'all',
    )

    if filter_by_snapshot:
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
