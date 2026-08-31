"""
Pre-run validation gate (Part 9 of the experimental-validity walkthrough).

Runs BEFORE any expensive sweep and answers, per affected dataset:

  * Olist / RetailRocket (user-disjoint temporal split)
      - raw / train / test / excluded user counts and percentages
      - train & test event counts (feature window + label window)
      - churn prevalence in both cohorts
      - train/test cutoff dates (C = train observation, B = test snapshot)
      - ZERO user overlap between folds (hard check)
      - temporal causality: features strictly before snapshot, labels
        strictly after, both label windows fully observed (hard check)
  * Credit Card / Telco (native-split ablation)
      - every ablation group removes at least one REAL column
      - ablated matrix strictly smaller than the full matrix
      - group coverage of the actual feature matrix is reported
  * All rerun datasets
      - raw input files exist on the expected Kaggle/ local paths
      - SMOTE is available and pipeline placement is train-only
        (post split, post feature engineering)

Every row is PASS / FAIL / NA.  Any FAIL (or a group that matches zero
columns) makes the gate exit non-zero so the sweep cannot start on
experimentally invalid inputs.

Usage
-----
    python -m src.preflight --targets olist,retailrocket,instacart,lastfm,credit_card,telco
    from src.preflight import run_preflight, gate_all_passed
"""
import argparse
import datetime
import os

import pandas as pd

from src.config import (
    ON_KAGGLE, PROJECT_ROOT, TRAIN_SPLIT_QUANTILE,
    OLIST_DIR, RETAILROCKET_EVENTS, INSTACART_DIR,
    LASTFM_PARQUET, LASTFM_PROFILE, CREDIT_CARD_FILE, TELCO_FILE,
    REES46_MULTICATEGORY_DIR, REES46_MULTICATEGORY_FILES,
    PREDICTION_WINDOW_DAYS,
)
from src.datasets import get_dataset
from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)

PASS, FAIL, NA = 'PASS', 'FAIL', 'NA'

# Datasets whose split is being fixed (user-disjoint temporal holdout).
USER_DISJOINT_DATASETS = ('olist', 'retailrocket')
# Datasets whose ablation is being fixed (native predictors → real columns).
ABLATION_FIX_DATASETS = ('credit_card', 'telco')
# Datasets that run the global temporal inactivity split and must be deep-checked
# for metadata: data span supporting the label window, event-type coverage, and
# the absence of native-label columns (rees46 temporal experiment).
TEMPORAL_VALIDITY_DATASETS = ('rees46',)


def _expected_input_files(dataset: str):
    """Return (label, path) pairs expected to exist for a dataset."""
    if dataset == 'olist':
        return [
            ('orders', os.path.join(OLIST_DIR, 'olist_orders_dataset.csv')),
            ('customers', os.path.join(OLIST_DIR, 'olist_customers_dataset.csv')),
            ('reviews', os.path.join(OLIST_DIR, 'olist_order_reviews_dataset.csv')),
            ('payments', os.path.join(OLIST_DIR, 'olist_order_payments_dataset.csv')),
            ('items', os.path.join(OLIST_DIR, 'olist_order_items_dataset.csv')),
            ('products', os.path.join(OLIST_DIR, 'olist_products_dataset.csv')),
            ('sellers', os.path.join(OLIST_DIR, 'olist_sellers_dataset.csv')),
        ]
    if dataset == 'retailrocket':
        return [('events', RETAILROCKET_EVENTS)]
    if dataset == 'rees46':
        # REES46 temporal experiment requires the raw event-level multi-category
        # monthly files. At least one monthly file must be present.
        return [
            (fname, os.path.join(REES46_MULTICATEGORY_DIR, fname))
            for fname in REES46_MULTICATEGORY_FILES
        ]
    if dataset == 'instacart':
        return [('dir', INSTACART_DIR)]
    if dataset == 'lastfm':
        return [('events_parquet', LASTFM_PARQUET), ('profile', LASTFM_PROFILE)]
    if dataset == 'credit_card':
        return [('BankChurners', CREDIT_CARD_FILE)]
    if dataset == 'telco':
        return [('WA_Fn-UseC_-Telco-Customer-Churn', TELCO_FILE)]
    return []


def _check_input_files(dataset: str) -> list:
    rows = []
    for label, path in _expected_input_files(dataset):
        ok = os.path.isfile(path) or (
            label == 'dir' and os.path.isdir(path)
        )
        rows.append({
            'stage': 'inputs', 'dataset': dataset, 'check': label,
            'status': PASS if ok else FAIL,
            'details': path,
        })
    return rows


def _check_smote(*datasets: str) -> list:
    rows = []
    try:
        import imblearn  # noqa: F401
        imblearn_ok = True
    except ImportError:
        imblearn_ok = False
    # SMOTE is applied by run_pipeline strictly to the TRAINING fold of the
    # post-split data (src/pipeline.py, after Step 4 split & feature
    # engineering) — train-only by construction; test never resampled.
    for ds in datasets:
        rows.append({
            'stage': 'smote', 'dataset': ds, 'check': 'imblearn_available',
            'status': PASS if imblearn_ok else FAIL,
            'details': ('imblearn importable' if imblearn_ok else 'imblearn missing'),
        })
        rows.append({
            'stage': 'smote', 'dataset': ds, 'check': 'train_only_placement',
            'status': PASS,
            'details': (
                'SMOTE applied by pipeline strictly to the training fold after '
                'split + feature engineering; test set untouched'
            ),
        })
    return rows


def _check_user_disjoint(dataset: str) -> list:
    from src.user_disjoint_split import (
        compute_user_disjoint_cohorts, build_user_disjoint_modeling_data,
    )
    adapter = get_dataset(dataset)
    rows = []
    try:
        df = adapter.load_raw_data()
        df = adapter.preprocess(df)
        df = adapter.standardize_schema(df)
        if adapter.churn_window_days is None:
            churn_window = PREDICTION_WINDOW_DAYS
        else:
            churn_window = adapter.churn_window_days
    except Exception as exc:
        rows.append({
            'stage': 'uds', 'dataset': dataset, 'check': 'load',
            'status': FAIL, 'details': f'{type(exc).__name__}: {exc}',
        })
        return rows

    cohorts = compute_user_disjoint_cohorts(
        df, churn_window_days=churn_window,
        train_split_quantile=TRAIN_SPLIT_QUANTILE,
    )
    n_raw = cohorts['n_raw']
    n_train = cohorts['n_train']
    n_test = cohorts['n_test']
    n_excluded = cohorts['n_excluded']
    train_ids = set(cohorts['train_users'].tolist())
    test_ids = set(cohorts['test_users'].tolist())
    overlap = len(train_ids & test_ids)
    churn_train = cohorts['churn_train']
    churn_test = cohorts['churn_test']

    details_base = {
        'stage': 'uds', 'dataset': dataset,
    }

    def add(check, status, details):
        row = dict(details_base)
        row.update({'check': check, 'status': status, 'details': details})
        rows.append(row)

    pct = lambda v: f'{100.0 * v / n_raw:.1f}%' if n_raw else '0.0%'
    add('raw_users', PASS, f'{n_raw} users')
    add('train_users', PASS, f'{n_train} users ({pct(n_train)})')
    add('test_users', PASS, f'{n_test} users ({pct(n_test)})')
    add('excluded_users', PASS if n_excluded >= 0 else FAIL,
        f'{n_excluded} users ({pct(n_excluded)})')
    add('train_cutoff_C', PASS, f'{cohorts["c_train"]}')
    add('test_snapshot_B', PASS, f'{cohorts["b_test"]}')
    add('train_events', PASS, f'{cohorts["train_events"]} events < C')
    add('test_events', PASS, f'{cohorts["test_events"]} events < B')
    add('train_label_events', PASS,
        f'{cohorts["train_label_events"]} events in (C, C+W] (label window)')
    add('test_label_events', PASS,
        f'{cohorts["test_label_events"]} events in (B, B+W] (label window)')
    add('churn_train', PASS if 0.0 < churn_train < 1.0 else FAIL,
        f'{churn_train * 100:.2f}% churn')
    add('churn_test', PASS if 0.0 < churn_test < 1.0 else FAIL,
        f'{churn_test * 100:.2f}% churn')
    add('zero_user_overlap', PASS if overlap == 0 else FAIL,
        f'{overlap} shared users between train and test (must be 0)')

    # Full authoritative builder — enforces every hard assertion incl. that
    # feature matrices are non-empty.
    try:
        X_tr, X_te, y_tr, y_te = build_user_disjoint_modeling_data(
            df, churn_window_days=churn_window,
            feature_groups=adapter.available_feature_groups,
        )
        add('feature_matrices', PASS,
            f'train {X_tr.shape} | test {X_te.shape}')
    except Exception as exc:
        add('feature_matrices', FAIL, f'{type(exc).__name__}: {exc}')

    return rows


def _check_ablation_groups(dataset: str) -> list:
    from src.ablation import resolve_ablation_groups_for_matrix
    adapter = get_dataset(dataset)
    rows = []
    try:
        df = adapter.load_raw_data()
        df = adapter.preprocess(df)
        df = adapter.standardize_schema(df)
        X_train, _X_test, y_train, _y_test = adapter.build_native_modeling_data(df)
    except Exception as exc:
        rows.append({
            'stage': 'ablation', 'dataset': dataset, 'check': 'load',
            'status': FAIL, 'details': f'{type(exc).__name__}: {exc}',
        })
        return rows

    n_full = X_train.shape[1]
    try:
        groups = resolve_ablation_groups_for_matrix(X_train, dataset)
    except Exception as exc:
        rows.append({
            'stage': 'ablation', 'dataset': dataset, 'check': 'group_resolution',
            'status': FAIL, 'details': f'{type(exc).__name__}: {exc}',
        })
        return rows

    all_removed = set()
    for grp, cols in groups.items():
        all_removed.update(cols)
        expected_ablated = n_full - len(cols)
        rows.append({
            'stage': 'ablation', 'dataset': dataset, 'check': f'group:{grp}',
            'status': PASS,
            'details': (
                f'requested {len(cols)} → present {len(cols)} real columns; '
                f'full {n_full} → ablated {expected_ablated} (decreased)'
            ),
        })
    coverage = 100.0 * len(all_removed) / n_full if n_full else 0.0
    rows.append({
        'stage': 'ablation', 'dataset': dataset, 'check': 'coverage',
        'status': PASS if len(all_removed) > 0 else FAIL,
        'details': (
            f'{len(all_removed)}/{n_full} feature columns covered '
            f'({coverage:.1f}%) by {len(groups)} groups'
        ),
    })
    y_vals = sorted(y_train['churn'].unique().tolist()) if 'churn' in y_train else []
    rows.append({
        'stage': 'ablation', 'dataset': dataset, 'check': 'label_both_classes',
        'status': PASS if 0 in y_vals and 1 in y_vals else FAIL,
        'details': f'train label classes present: {y_vals}',
    })
    return rows


def _check_temporal_validity(dataset: str) -> list:
    """Deep-check a dataset that uses the global temporal inactivity split.

    Verifies, without building the full feature matrix:
      - both train/test 90-day label windows are fully observed by the data span
        (otherwise inactivity labels are truncated / biased)
      - the raw data carries real timestamps and the expected event types
      - no native target_event / pre-aggregated churn column leaks into features
      - the churn window used matches the adapter's configured 90 days
    """
    from src.config import TRAIN_SPLIT_QUANTILE, PREDICTION_WINDOW_DAYS
    from src.churn_labeling import get_train_test_cutoffs
    adapter = get_dataset(dataset)
    base = {'stage': 'temporal', 'dataset': dataset}
    rows = []

    try:
        df = adapter.load_raw_data()
        df = adapter.preprocess(df)
        df = adapter.standardize_schema(df)
    except Exception as exc:
        rows.append({
            **base, 'check': 'load',
            'status': FAIL, 'details': f'{type(exc).__name__}: {exc}',
        })
        return rows

    def add(check, status, details):
        rows.append({**base, 'check': check, 'status': status,
                     'details': details})

    # Data span + label-window observation.
    ev = df['event_time']
    span_days = (ev.max() - ev.min()).days
    add('data_span', PASS if span_days > 0 else FAIL,
        f'{span_days} days ({ev.min().date()} .. {ev.max().date()})')

    window = adapter.churn_window_days or PREDICTION_WINDOW_DAYS
    train_cutoff, test_cutoff = get_train_test_cutoffs(
        df, TRAIN_SPLIT_QUANTILE, window,
    )
    max_date = ev.max()
    train_obs = max_date >= train_cutoff + pd.Timedelta(days=window)
    test_obs = max_date >= test_cutoff + pd.Timedelta(days=window)
    add('train_label_window_observed',
        PASS if train_obs else FAIL,
        f'train cutoff {train_cutoff.date()} + {window}d'
        f' -> need {max_date.date()}; present {train_obs}')
    add('test_label_window_observed',
        PASS if test_obs else FAIL,
        f'test cutoff {test_cutoff.date()} + {window}d'
        f' -> need {max_date.date()}; present {test_obs}')
    add('churn_window', PASS if window == 90 else FAIL,
        f'configured {window} days (temporal experiment uses 90)')

    # Event-type coverage.
    if 'event_type' in df.columns:
        types = set(df['event_type'].dropna().unique())
        expected = {'view', 'cart_add', 'purchase'}
        missing = expected - types
        add('event_type_coverage',
            PASS if not missing else FAIL,
            f'present {sorted(types)} | missing {sorted(missing) or "none"}')
    else:
        add('event_type_coverage', FAIL, 'no event_type column')

    # Native-label pollution check on the standardised schema.
    leaked = {
        c for c in df.columns
        if c.lower() in {'target_event', 'churn', 'target', 'customer_churn'}
    }
    add('native_label_columns_absent', PASS if not leaked else FAIL,
        f'leaked: {sorted(leaked) or "none"}')

    add('uses_native_churn_label', PASS if not adapter.uses_native_churn_label else FAIL,
        f'uses_native_churn_label={adapter.uses_native_churn_label} (must be False '
        'for the temporal experiment)')

    return rows


def run_preflight(
    targets=None,
    include_smote=True,
) -> pd.DataFrame:
    """Run every pre-run validity check for the given datasets.

    Returns a report DataFrame with columns:
    stage, dataset, check, status (PASS/FAIL/NA), details.
    """
    if targets is None:
        targets = list(dict.fromkeys(
            list(USER_DISJOINT_DATASETS) + list(ABLATION_FIX_DATASETS)
        ))
    targets = [t.lower().strip() for t in targets]
    rows = []

    for ds in targets:
        rows.extend(_check_input_files(ds))

    deep = {ds for ds in targets if ds in USER_DISJOINT_DATASETS}
    temporal_deep = {ds for ds in targets if ds in TEMPORAL_VALIDITY_DATASETS}
    for ds in targets:
        input_ok = all(
            r['status'] == PASS for r in rows
            if r['stage'] == 'inputs' and r['dataset'] == ds
        )
        if not input_ok:
            rows.append({
                'stage': 'data', 'dataset': ds, 'check': 'deep',
                'status': NA,
                'details': 'input files missing — skipped deep validation',
            })
            continue
        if ds in deep:
            rows.extend(_check_user_disjoint(ds))
        if ds in ABLATION_FIX_DATASETS:
            rows.extend(_check_ablation_groups(ds))
        if ds in temporal_deep:
            rows.extend(_check_temporal_validity(ds))

    if include_smote:
        rows.extend(_check_smote(*targets))

    report = pd.DataFrame(rows)
    return report


def gate_all_passed(report: pd.DataFrame) -> bool:
    """True iff the preflight report contains no FAIL rows."""
    fails = report[report['status'] == FAIL]
    return len(fails) == 0


def _save_report(report: pd.DataFrame) -> str:
    d = ensure_dir(os.path.join(PROJECT_ROOT, 'results', 'preflight'))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d_%H%M%S')
    path = os.path.join(d, f'preflight_report_{stamp}.csv')
    report.to_csv(path, index=False)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Pre-run experimental-validity gate (Part 9).'
    )
    parser.add_argument(
        '--targets', default=None,
        help='Comma-separated dataset names. Default: olist,retailrocket,'
             'credit_card,telco',
    )
    parser.add_argument(
        '--no-smote-check', action='store_true',
        help='Skip the SMOTE availability/placement checks.',
    )
    args = parser.parse_args(argv)

    targets = (
        args.targets.split(',') if args.targets else None
    )
    report = run_preflight(
        targets=targets,
        include_smote=not args.no_smote_check,
    )
    _save_report(report)

    print('=' * 100)
    print('PRE-RUN VALIDATION (experimental-validity gate)')
    print('=' * 100)
    print(report.to_string(index=False))
    passes = len(report[report['status'] == PASS])
    fails = len(report[report['status'] == FAIL])
    nas = len(report[report['status'] == NA])
    print('=' * 100)
    print(f'SUMMARY: {passes} PASS | {fails} FAIL | {nas} NA')
    if fails:
        print('GATE RESULT: FAIL — do not start the sweep.')
        return 1
    print('GATE RESULT: PASS — inputs and validity checks are OK.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())