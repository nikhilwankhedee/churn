"""
Data loading and merging of Olist CSV files.
Handles missing files, schema validation, and proper aggregation
to prevent row-count explosions from one-to-many joins.
"""
import os
import pandas as pd
from typing import List, Optional

from src.config import (
    DATA_DIR, ORDERS_FILE, CUSTOMERS_FILE, REVIEWS_FILE,
    PAYMENTS_FILE, ITEMS_FILE, PRODUCTS_FILE, SELLERS_FILE,
)
from src.utils import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS: dict = {
    'orders': [
        'order_id', 'customer_id', 'order_purchase_timestamp',
        'order_approved_at', 'order_delivered_carrier_date',
        'order_delivered_customer_date', 'order_estimated_delivery_date',
    ],
    'customers': ['customer_id', 'customer_unique_id', 'customer_city',
                  'customer_state', 'customer_zip_code_prefix'],
    'reviews': ['order_id', 'review_score', 'review_creation_date',
                'review_answer_timestamp'],
    'payments': ['order_id', 'payment_sequential', 'payment_type',
                 'payment_installments', 'payment_value'],
    'items': ['order_id', 'order_item_id', 'product_id', 'seller_id',
              'shipping_limit_date', 'price', 'freight_value'],
    'products': ['product_id', 'product_category_name',
                 'product_weight_g', 'product_length_cm',
                 'product_height_cm', 'product_width_cm'],
    'sellers': ['seller_id', 'seller_city', 'seller_state',
                'seller_zip_code_prefix'],
}


def _validate_columns(df: pd.DataFrame, expected: List[str], name: str) -> None:
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def _safe_read_csv(filepath: str, expected_cols: List[str],
                   table_name: str, **kwargs) -> Optional[pd.DataFrame]:
    if not os.path.isfile(filepath):
        logger.warning("File not found: %s — skipping %s", filepath, table_name)
        return None
    try:
        df = pd.read_csv(filepath, **kwargs)
        _validate_columns(df, expected_cols, table_name)
        logger.info("Loaded %s: %d rows x %d cols", table_name, df.shape[0], df.shape[1])
        return df
    except Exception as exc:
        logger.error("Failed to load %s: %s", filepath, exc)
        return None


def load_raw_data() -> pd.DataFrame:
    orders = _safe_read_csv(
        os.path.join(DATA_DIR, ORDERS_FILE),
        REQUIRED_COLUMNS['orders'], 'orders',
    )
    customers = _safe_read_csv(
        os.path.join(DATA_DIR, CUSTOMERS_FILE),
        REQUIRED_COLUMNS['customers'], 'customers',
    )
    reviews = _safe_read_csv(
        os.path.join(DATA_DIR, REVIEWS_FILE),
        REQUIRED_COLUMNS['reviews'], 'reviews',
    )
    payments = _safe_read_csv(
        os.path.join(DATA_DIR, PAYMENTS_FILE),
        REQUIRED_COLUMNS['payments'], 'payments',
    )
    items = _safe_read_csv(
        os.path.join(DATA_DIR, ITEMS_FILE),
        REQUIRED_COLUMNS['items'], 'order_items',
    )
    products = _safe_read_csv(
        os.path.join(DATA_DIR, PRODUCTS_FILE),
        REQUIRED_COLUMNS['products'], 'products',
    )
    sellers = _safe_read_csv(
        os.path.join(DATA_DIR, SELLERS_FILE),
        REQUIRED_COLUMNS['sellers'], 'sellers',
    )

    if orders is None:
        raise FileNotFoundError(f"Required file {ORDERS_FILE} not found in {DATA_DIR}")

    orders = orders.copy()
    if customers is not None:
        orders = orders.merge(
            customers[['customer_id', 'customer_unique_id']],
            on='customer_id', how='left',
        )
    else:
        logger.warning("Customers table missing — using customer_id as unique_id")
        orders['customer_unique_id'] = orders['customer_id']

    if reviews is not None:
        reviews_dedup = (
            reviews.sort_values('review_creation_date')
            .drop_duplicates(subset='order_id', keep='first')
        )
        orders = orders.merge(
            reviews_dedup[['order_id', 'review_score']],
            on='order_id', how='left',
        )

    if payments is not None:
        pay_agg = payments.groupby('order_id', as_index=False).agg(
            payment_installments=('payment_installments', 'sum'),
            payment_value=('payment_value', 'sum'),
            payment_type=('payment_type',
                          lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown'),
            payment_sequential_count=('payment_sequential', 'count'),
        )
        orders = orders.merge(pay_agg, on='order_id', how='left')

    if items is not None:
        item_agg = items.groupby('order_id', as_index=False).agg(
            product_id=('product_id', lambda x: x.iloc[0] if not x.empty else 'unknown'),
            price=('price', 'sum'),
            freight_value=('freight_value', 'sum'),
            item_count=('order_item_id', 'count'),
        )
        orders = orders.merge(item_agg, on='order_id', how='left')

    if products is not None and 'product_id' in orders.columns:
        orders = orders.merge(
            products[['product_id', 'product_category_name']],
            on='product_id', how='left',
        )

    n_orders_before = orders['order_id'].nunique() if 'order_id' in orders.columns else 0
    orders = orders.drop_duplicates(subset='order_id', keep='first')
    n_orders_after = orders['order_id'].nunique()
    if n_orders_before > n_orders_after:
        logger.info("Deduplicated orders: %d → %d unique orders",
                     n_orders_before, n_orders_after)

    logger.info("Final merged dataset: %d rows x %d columns", orders.shape[0], orders.shape[1])
    return orders
