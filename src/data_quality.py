"""
Automated data quality reporting for the merged Olist dataset.
"""
import pandas as pd
import numpy as np
from src.utils import get_logger

logger = get_logger(__name__)


def generate_data_quality_report(df: pd.DataFrame) -> dict:
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'duplicate_rows': int(df.duplicated().sum()),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    report['columns_with_missing'] = len(missing)
    if not missing.empty:
        report['missing_detail'] = missing.to_dict()

    if 'order_purchase_timestamp' in df.columns:
        ts = df['order_purchase_timestamp'].dropna()
        if not ts.empty:
            report['date_range'] = f"{ts.min()} — {ts.max()}"

    if ('order_delivered_customer_date' in df.columns
            and 'order_estimated_delivery_date' in df.columns):
        neg_delivery = (
            df['order_delivered_customer_date'] < df['order_estimated_delivery_date']
        ).sum()
        report['early_deliveries'] = int(neg_delivery)

    for col in ['price', 'freight_value']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            iqr = Q3 - Q1
            lo = Q1 - 1.5 * iqr
            hi = Q3 + 1.5 * iqr
            n_out = ((df[col] < lo) | (df[col] > hi)).sum()
            report[f'{col}_outliers_iqr'] = int(n_out)

    if 'review_score' in df.columns:
        report['review_score_missing'] = int(df['review_score'].isnull().sum())
        report['review_score_mean'] = round(df['review_score'].mean(), 2)

    n_unique_customers = df['customer_unique_id'].nunique() if 'customer_unique_id' in df.columns else 0
    report['unique_customers'] = int(n_unique_customers)

    logger.info("Data quality report generated — %d metrics", len(report))
    return report
