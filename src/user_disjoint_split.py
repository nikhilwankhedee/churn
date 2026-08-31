"""
User-disjoint temporal train/test split.

Eliminates the train/test **user overlap** that the plain global-cutoff
temporal path suffers from on event-stream datasets (Olist, RetailRocket),
where the same customer can appear in both the training and the test
feature matrices with overlapping churn-label windows.

Design (late-arrival test cohort)
---------------------------------
Two cohorts are kept disjoint *by construction* because they occupy
non-overlapping ranges of the calendar:

    T0 ──── [train users: first_event < C] ──── C ──[(C, C+W] train label]──
        ... ── b_test ──[(b_test, b_test+W] test label]── Tmax

    * train users : first_event < C
        features = events < C
        label    = churned iff no event in (C, C+W]
    * test users  : first_event >= C            (a strictly later arrival set)
        features = events < b_test
        label    = churned iff no event in (b_test, b_test+W]

A user cannot have ``first_event < C`` and ``first_event >= C`` at the same
time, so TRAIN_USERS ∩ TEST_USERS = ∅ for every run.

Causality is preserved per cohort (features are always computed from events
strictly before the cohort's own label window), and the test cohort is
strictly later in calendar time than the training cohort.

Users whose timeline is too short to yield a valid label window (no event in
the label window **and** no event before their snapshot to build features) are
excluded and reported rather than silently dropped.

All validity invariants are enforced with hard assertions — the split fails
loudly instead of starting an expensive, invalid model run.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any

from src.config import TRAIN_SPLIT_QUANTILE, RANDOM_SEED
from src.utils import get_logger

logger = get_logger(__name__)

STD_CUSTOMER_ID = 'customer_id'
STD_EVENT_TIME = 'event_time'


def _report_user_disjoint_split(
    n_raw: int, n_train: int, n_test: int, n_excluded: int,
    c_train, b_test, churn_train: float, churn_test: float,
) -> None:
    """Log a structured summary of the split (fail-loudly friendly)."""
    logger.info("=" * 60)
    logger.info("User-disjoint temporal split")
    logger.info("=" * 60)
    logger.validation("UDSplit | raw users        : %d", n_raw)
    logger.validation("UDSplit | train users      : %d", n_train)
    logger.validation("UDSplit | test users       : %d", n_test)
    logger.validation("UDSplit | excluded users   : %d", n_excluded)
    if n_raw > 0:
        logger.validation(
            "UDSplit | train %% of raw  : %.1f%%", 100.0 * n_train / n_raw,
        )
        logger.validation(
            "UDSplit | test %% of raw   : %.1f%%", 100.0 * n_test / n_raw,
        )
        logger.validation(
            "UDSplit | excluded %%      : %.1f%%", 100.0 * n_excluded / n_raw,
        )
    logger.validation("UDSplit | train cutoff (C) : %s", c_train)
    logger.validation("UDSplit | test snapshot (B): %s", b_test)
    logger.validation("UDSplit | train churn      : %.2f%%", churn_train * 100)
    logger.validation("UDSplit | test churn       : %.2f%%", churn_test * 100)


def compute_user_disjoint_cohorts(
    df: pd.DataFrame,
    churn_window_days: int,
    train_split_quantile: float = TRAIN_SPLIT_QUANTILE,
) -> dict:
    """Compute user-disjoint temporal cohorts WITHOUT engineering features.

    Cheap validation path used by the pre-run validation gate (:mod:`src.preflight`)
    and the authoritative builder (:func:`build_user_disjoint_modeling_data`), so
    counts, cutoffs and churn rates are derived from a single implementation.

    Returns a dict with: c_train, b_test, W, n_raw, n_train, n_test,
    n_excluded, train_events, test_events, train_label_events,
    test_label_events, churn_train, churn_test, train_users, test_users.
    """
    required = {STD_CUSTOMER_ID, STD_EVENT_TIME}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"user_disjoint_split missing columns: {missing}"
        )

    events = df.dropna(subset=[STD_EVENT_TIME, STD_CUSTOMER_ID]).copy()
    if events.empty:
        raise ValueError("No events available for user-disjoint split")

    W = pd.Timedelta(days=churn_window_days)
    Tmax = events[STD_EVENT_TIME].max()

    # Reserve the final `W` for the test label window so test labels are not
    # right-censored.  The test snapshot sits W days before the end.
    b_test = Tmax - W

    # Training observation cutoff: event-time quantile, capped so the train
    # label window (C, C+W] finishes at or before the test snapshot.
    c_raw = events[STD_EVENT_TIME].quantile(train_split_quantile)
    c_train = min(c_raw, b_test - W)
    if c_train >= b_test:
        # Corner case: window too large relative to span — force separation.
        c_train = b_test - W

    # ── Assign users to cohorts by first-event time ────────────────────
    first_event = (
        events.groupby(STD_CUSTOMER_ID)[STD_EVENT_TIME].min()
    )

    # Train cohort: first event strictly before C (active in training era).
    train_users = first_event[first_event < c_train].index

    # Test cohort: strictly later arrivals (first event >= C) that also have
    # history before the test snapshot (first_event < b_test) so features exist.
    test_mask = (first_event >= c_train) & (first_event < b_test)
    test_users = first_event[test_mask].index

    # Excluded: users that cannot yield a valid cohort (reported, not silent).
    # Hash-based set membership — np.isin on object dtype degenerates to an
    # O(n*m) Python loop that hangs on large event datasets (e.g. RetailRocket
    # ~1.1M x ~1.4M ids), so keep ids in a set instead.
    all_users = events[STD_CUSTOMER_ID].unique()
    kept = set(np.asarray(train_users, object))
    kept.update(np.asarray(test_users, object))
    excluded = np.asarray([u for u in all_users if u not in kept], dtype=object)

    train_users = pd.Index(train_users, name=STD_CUSTOMER_ID)
    test_users = pd.Index(test_users, name=STD_CUSTOMER_ID)
    excluded = pd.Index(excluded, name=STD_CUSTOMER_ID)

    def _labels(users, snapshot):
        """churn = 1 iff the user has no event in (snapshot, snapshot+W]."""
        if len(users) == 0:
            return pd.Series(dtype=float)
        window_end = snapshot + W
        had_events_in_window = events[
            (events[STD_EVENT_TIME] > snapshot)
            & (events[STD_EVENT_TIME] <= window_end)
        ][STD_CUSTOMER_ID].unique()
        active = set(had_events_in_window)
        return pd.Series(
            {uid: (0 if uid in active else 1) for uid in users},
            dtype=float,
        )

    train_labels = _labels(train_users, c_train)
    test_labels = _labels(test_users, b_test)

    def _count_events(users, before, after=None):
        m = events[STD_CUSTOMER_ID].isin(users)
        if before is not None:
            m &= events[STD_EVENT_TIME] < before
        if after is not None:
            m &= events[STD_EVENT_TIME] > after
        return int(m.sum())

    return {
        'c_train': c_train,
        'b_test': b_test,
        'W': W,
        'n_raw': int(len(all_users)),
        'n_train': int(len(train_users)),
        'n_test': int(len(test_users)),
        'n_excluded': int(len(excluded)),
        'train_events': _count_events(train_users, c_train),
        'test_events': _count_events(test_users, b_test),
        'train_label_events': _count_events(train_users, None, c_train),
        'test_label_events': _count_events(test_users, None, b_test),
        'churn_train': float(train_labels.mean()),
        'churn_test': float(test_labels.mean()),
        'train_users': train_users,
        'test_users': test_users,
        'excluded_users': excluded,
    }


def build_user_disjoint_modeling_data(
    df: pd.DataFrame,
    churn_window_days: int,
    train_split_quantile: float = TRAIN_SPLIT_QUANTILE,
    feature_groups: Optional[list] = None,
    min_train_users: int = 100,
    min_test_users: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build user-disjoint train/test features + labels.

    Parameters
    ----------
    df : pd.DataFrame
        Standardised schema (customer_id, event_time, plus feature columns).
    churn_window_days : int
        Churn (inactivity) window in days.
    train_split_quantile : float
        Quantile of the event timeline used to place the training observation
        cutoff C (capped so the training label window fits before the test
        snapshot).
    feature_groups : list of str, optional
        Feature groups to engineer (delegated to engineer_features).
    min_train_users / min_test_users : int
        Minimum cohort sizes; the split raises if either is not met.

    Returns
    -------
    (X_train, X_test, y_train, y_test) with user ids as index.
    """
    from src.feature_engineering import engineer_features

    cohorts = compute_user_disjoint_cohorts(
        df,
        churn_window_days=churn_window_days,
        train_split_quantile=train_split_quantile,
    )
    events = df.dropna(subset=[STD_EVENT_TIME, STD_CUSTOMER_ID]).copy()
    c_train = cohorts['c_train']
    b_test = cohorts['b_test']
    train_users = cohorts['train_users']
    test_users = cohorts['test_users']
    excluded = cohorts['excluded_users']

    # ── Per-cohort feature engineering (events strictly before snapshot) ─
    train_features = engineer_features(
        events, c_train,
        customer_ids=train_users.tolist(),
        available_groups=feature_groups,
    )
    test_features = engineer_features(
        events, b_test,
        customer_ids=test_users.tolist(),
        available_groups=feature_groups,
    )

    if train_features.empty:
        raise RuntimeError(
            "User-disjoint split: train feature matrix empty at cutoff "
            f"{c_train}"
        )
    if test_features.empty:
        raise RuntimeError(
            "User-disjoint split: test feature matrix empty at snapshot "
            f"{b_test}"
        )

    # ── Churn labels from each cohort's own future window ───────────────
    def _labels(users, snapshot):
        if len(users) == 0:
            return pd.DataFrame(
                {STD_CUSTOMER_ID: [], 'churn': []},
            ).set_index(STD_CUSTOMER_ID)
        window_end = snapshot + cohorts['W']
        had_events_in_window = events[
            (events[STD_EVENT_TIME] > snapshot)
            & (events[STD_EVENT_TIME] <= window_end)
        ][STD_CUSTOMER_ID].unique()
        active = set(had_events_in_window)
        rows = {
            uid: (0 if uid in active else 1)
            for uid in users
        }
        return pd.DataFrame(
            {STD_CUSTOMER_ID: list(rows.keys()),
             'churn': list(rows.values())},
        ).set_index(STD_CUSTOMER_ID)

    train_labels = _labels(train_users, c_train)
    test_labels = _labels(test_users, b_test)

    # Align features/labels by user id.
    train_labels = train_labels.loc[
        train_features.index.intersection(train_labels.index)
    ]
    train_features = train_features.loc[train_labels.index]
    test_labels = test_labels.loc[
        test_features.index.intersection(test_labels.index)
    ]
    test_features = test_features.loc[test_labels.index]

    # ── Hard validity assertions (fail loudly) ──────────────────────────
    train_id_set = set(train_features.index)
    test_id_set = set(test_features.index)
    assert not (train_id_set & test_id_set), (
        f"User-disjoint split violated: {len(train_id_set & test_id_set)} "
        f"users appear in BOTH train and test."
    )
    assert len(train_features) >= min_train_users, (
        f"Train cohort too small ({len(train_features)} < {min_train_users})"
    )
    assert len(test_features) >= min_test_users, (
        f"Test cohort too small ({len(test_features)} < {min_test_users})"
    )
    churn_train = train_labels['churn'].mean()
    churn_test = test_labels['churn'].mean()
    assert 0.0 < churn_train < 1.0, (
        f"Degenerate train churn rate {churn_train:.3f} — no discriminant signal"
    )
    assert 0.0 < churn_test < 1.0, (
        f"Degenerate test churn rate {churn_test:.3f} — no discriminant signal"
    )

    _report_user_disjoint_split(
        n_raw=int(events[STD_CUSTOMER_ID].nunique()),
        n_train=len(train_features),
        n_test=len(test_features),
        n_excluded=int(len(excluded)),
        c_train=c_train,
        b_test=b_test,
        churn_train=float(churn_train),
        churn_test=float(churn_test),
    )

    return (
        train_features,
        test_features,
        train_labels,
        test_labels,
    )
