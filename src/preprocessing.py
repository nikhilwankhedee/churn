"""
Data preprocessing: timestamp parsing, validity checks, outlier handling,
and basic cleaning.  All operations are deterministic and idempotent.
"""
import pandas as pd
import numpy as np
from typing import List

from src.config import TIMESTAMP_MIN, TIMESTAMP_MAX, OUTLIER_PRICING_PERCENTILE
from src.utils import get_logger

logger = get_logger(__name__)

DATE_COLS: List[str] = [
    'order_purchase_timestamp', 'order_approved_at',
    'order_delivered_carrier_date', 'order_delivered_customer_date',
    'order_estimated_delivery_date',
]

ALL_DATE_COLS: List[str] = list(set(DATE_COLS + [
    'shipping_limit_date', 'review_creation_date', 'review_answer_timestamp',
]))


def preprocess_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ALL_DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    before = len(df)
    df = df.dropna(subset=['order_purchase_timestamp'])
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with null purchase timestamp", dropped)

    before = len(df)
    ts_min = pd.Timestamp(TIMESTAMP_MIN)
    ts_max = pd.Timestamp(TIMESTAMP_MAX)
    valid = (df['order_purchase_timestamp'] >= ts_min) & (
        df['order_purchase_timestamp'] <= ts_max
    )
    df = df[valid].copy()
    filtered = before - len(df)
    if filtered:
        logger.info("Filtered %d rows outside [%s, %s]", filtered, TIMESTAMP_MIN, TIMESTAMP_MAX)

    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    orig_rows = len(df)

    for col in ['price', 'freight_value']:
        if col in df.columns:
            df = df[df[col] >= 0].copy()

    for col in ['price', 'freight_value']:
        if col in df.columns:
            cap = df[col].quantile(OUTLIER_PRICING_PERCENTILE)
            if cap > 0:
                df[col] = df[col].clip(upper=cap)

    if 'review_score' in df.columns:
        med = df['review_score'].median()
        if pd.isna(med):
            med = 5.0
        df['review_score'] = df['review_score'].fillna(med)

    dropped = orig_rows - len(df)
    if dropped:
        logger.info("Basic clean removed %d rows (negative price/freight)", dropped)

    return df


def engineer_missings(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['payment_installments', 'payment_value', 'price', 'freight_value']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df
