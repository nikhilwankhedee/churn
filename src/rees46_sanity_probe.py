"""
REES46 temporal sanity probe — run ON KAGGLE with the multi-category dataset
attached (mkechinov/ecommerce-behavior-data-from-multi-category-store).

This is the Phase-7 sanity test.  It does NOT run the full pipeline.  It
verifies, on the real ~19M-event raw dataset, everything needed to confirm the
temporal inactivity experiment is valid BEFORE the expensive full sweep:

  1. every expected monthly file exists and loads
  2. columns match the expected 9-col multi-category schema
  3. event_type coverage (view / cart / remove_from_cart / purchase)
  4. event_time parses to UTC datetimes; null/invalid dropped
  5. user_id / price integrity
  6. data span vs the 90-day label window (train & test fully observed)
  7. global temporal cutoffs (train=70% quantile, test=max-90d)
  8. churn rates at both cutoffs (should be a healthy 2-98% band, NOT ~0 or ~1)
  9. feature matrix builds on a sample and the temporal-validity assertions pass

Usage (on Kaggle, after attaching the dataset):
    !python src/rees46_sanity_probe.py
or:
    from src.rees46_sanity_probe import sanity_probe; sanity_probe()
"""
import os
import time

import numpy as np
import pandas as pd

from src.config import (
    ON_KAGGLE, TRAIN_SPLIT_QUANTILE,
    REES46_MULTICATEGORY_FILES,
)
from src.churn_labeling import get_train_test_cutoffs, create_churn_labels
from src.feature_engineering import engineer_features
from src.datasets.rees46 import REES46Adapter


def _chunked_event_count(path: str, chunksize: int = 1_000_000) -> int:
    """Count rows in a large CSV using chunked reads (memory-safe)."""
    n = 0
    for chunk in pd.read_csv(path, usecols=['event_time'], chunksize=chunksize):
        n += len(chunk)
    return n


def sanity_probe() -> dict:
    t0 = time.time()
    adapter = REES46Adapter()
    report: dict = {
        "dataset": "rees46",
        "phase": "sanity_probe",
        "on_kaggle": ON_KAGGLE,
    }

    print("=" * 80)
    print("REES46 TEMPORAL SANITY PROBE")
    print("=" * 80)

    # ── 1. Input files ────────────────────────────────────────────────
    data_dir = adapter._multicategory_dir()
    present = []
    missing = []
    for fname in REES46_MULTICATEGORY_FILES:
        path = os.path.join(data_dir, fname)
        if os.path.isfile(path):
            present.append(fname)
        else:
            missing.append(fname)
    report["monthly_files_present"] = present
    report["monthly_files_missing"] = missing
    report["data_dir"] = data_dir
    print(f"\n[1] Monthly files present ({data_dir}): {present}")
    if missing:
        print(f"    MISSING: {missing}")

    # ── 2. Load via adapter ───────────────────────────────────────────
    df = adapter.load_raw_data()
    report["n_events_raw"] = int(len(df))
    report["n_cols_raw"] = int(df.shape[1])
    report["raw_columns"] = list(df.columns)
    print(f"[2] Raw: {len(df):,} events x {df.shape[1]} cols")
    print(f"    columns: {list(df.columns)}")

    # ── 3. Event-type coverage (raw) ──────────────────────────────────
    evt_raw = df["event_type"].value_counts().to_dict()
    report["event_type_counts_raw"] = evt_raw
    print(f"[3] raw event_type counts: {evt_raw}")

    # ── 4. Preprocess + schema ───────────────────────────────────────
    df = adapter.preprocess(df)
    report["n_events_preprocessed"] = int(len(df))
    df = adapter.standardize_schema(df)
    evt_map = df["event_type"].value_counts().to_dict()
    report["event_type_counts_mapped"] = evt_map
    print(f"[4] preprocessed: {len(df):,} events; mapped event types: {evt_map}")

    # ── 5. Time span + label-window feasibility ───────────────────────
    ev = df["event_time"]
    span = (ev.max() - ev.min()).days
    report["first_event"] = str(ev.min())
    report["last_event"] = str(ev.max())
    report["span_days"] = int(span)
    window = adapter.churn_window_days
    train_cutoff, test_cutoff = get_train_test_cutoffs(
        df, TRAIN_SPLIT_QUANTILE, window,
    )
    report["train_cutoff"] = str(train_cutoff)
    report["test_cutoff"] = str(test_cutoff)
    train_obs = ev.max() >= train_cutoff + pd.Timedelta(days=window)
    test_obs = ev.max() >= test_cutoff + pd.Timedelta(days=window)
    report["train_label_window_observed"] = bool(train_obs)
    report["test_label_window_observed"] = bool(test_obs)
    print(f"[5] span {span}d ({ev.min()} .. {ev.max()})")
    print(f"    train cutoff {train_cutoff} | +{window}d observed: {train_obs}")
    print(f"    test  cutoff {test_cutoff} | +{window}d observed: {test_obs}")

    # ── 6. Churn geometry at both cutoffs ─────────────────────────────
    train_labels = create_churn_labels(
        df, train_cutoff, prediction_window_days=window,
    )
    test_labels = create_churn_labels(
        df, test_cutoff, prediction_window_days=window,
    )
    ct = float(train_labels["churn"].mean())
    ck = float(test_labels["churn"].mean())
    report["churn_rate_train"] = ct
    report["churn_rate_test"] = ck
    report["n_train_customers"] = int(len(train_labels))
    report["n_test_customers"] = int(len(test_labels))
    print(f"[6] train churn {ct*100:.2f}% ({len(train_labels):,} cust)")
    print(f"    test  churn {ck*100:.2f}% ({len(test_labels):,} cust)")
    if ct <= 0.01 or ct >= 0.99 or ck <= 0.01 or ck >= 0.99:
        print("    WARNING: extreme churn rate — label geometry suspect")

    # ── 7. Feature matrix on a bounded sample + assertions ────────────
    # The full pipeline engineers on all customers; here we subset to keep the
    # probe fast, then run the same hard assertions on the standardised data.
    train_labels = train_labels.set_index("customer_id")
    test_labels = test_labels.set_index("customer_id")
    grps = adapter.available_feature_groups
    max_cust = 50_000
    tr_ids = train_labels.index[:max_cust].tolist()
    te_ids = test_labels.index[:max_cust].tolist()
    tr = engineer_features(df, train_cutoff, customer_ids=tr_ids,
                           available_groups=grps)
    te = engineer_features(df, test_cutoff, customer_ids=te_ids,
                           available_groups=grps)
    for c in set(tr.columns) - set(te.columns):
        te[c] = 0.0
    for c in set(te.columns) - set(tr.columns):
        tr[c] = 0.0
    te = te[tr.columns]
    trl = train_labels.loc[tr.index.intersection(train_labels.index)]
    tr = tr.loc[trl.index]
    tel = test_labels.loc[te.index.intersection(test_labels.index)]
    te = te.loc[tel.index]
    report["sample_feature_shape_train"] = list(tr.shape)
    report["sample_feature_shape_test"] = list(te.shape)
    report["n_features"] = int(tr.shape[1])
    report["feature_columns"] = list(tr.columns)
    print(f"[7] sample feature train {tr.shape} | test {te.shape}")

    checks = adapter.assert_temporal_validity(
        df, train_cutoff, test_cutoff, trl, tel, tr, te,
    )
    report["validity_checks"] = checks
    print("[8] temporal validity assertions PASSED:")
    for c in checks:
        print(f"    - {c}")

    report["duration_sec"] = round(time.time() - t0, 2)
    print(f"\nSanity probe complete in {report['duration_sec']}s.")
    return report


if __name__ == "__main__":
    sanity_probe()
