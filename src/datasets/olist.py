"""
Olist Brazilian E-Commerce dataset adapter.

Ecosystem type: transactional_marketplace
Churn window: 180 days of inactivity
Citation: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
"""
import os
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

from src.datasets.base import BaseDatasetAdapter
from src.config import (
    ORDERS_FILE, CUSTOMERS_FILE, REVIEWS_FILE,
    PAYMENTS_FILE, ITEMS_FILE, PRODUCTS_FILE, SELLERS_FILE,
    TIMESTAMP_MIN, TIMESTAMP_MAX, OUTLIER_PRICING_PERCENTILE,
)
from src.utils import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS: dict = {
    "orders": [
        "order_id", "customer_id", "order_purchase_timestamp",
        "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date",
    ],
    "customers": [
        "customer_id", "customer_unique_id", "customer_city",
        "customer_state", "customer_zip_code_prefix",
    ],
    "reviews": [
        "order_id", "review_score", "review_creation_date",
        "review_answer_timestamp",
    ],
    "payments": [
        "order_id", "payment_sequential", "payment_type",
        "payment_installments", "payment_value",
    ],
    "items": [
        "order_id", "order_item_id", "product_id", "seller_id",
        "shipping_limit_date", "price", "freight_value",
    ],
    "products": ["product_id", "product_category_name",
                  "product_weight_g", "product_length_cm",
                  "product_height_cm", "product_width_cm"],
    "sellers": ["seller_id", "seller_city", "seller_state",
                 "seller_zip_code_prefix"],
}


class OlistAdapter(BaseDatasetAdapter):
    @property
    def dataset_name(self) -> str:
        return "olist"

    @property
    def ecosystem_type(self) -> str:
        return "transactional_marketplace"

    @property
    def churn_window_days(self) -> Optional[int]:
        return 180

    @property
    def required_files(self) -> list:
        return [
            ORDERS_FILE, CUSTOMERS_FILE, REVIEWS_FILE,
            PAYMENTS_FILE, ITEMS_FILE, PRODUCTS_FILE, SELLERS_FILE,
        ]

    # ── Data loading ─────────────────────────────────────────────────

    def _validate_columns(self, df: pd.DataFrame, expected: list, name: str):
        missing = set(expected) - set(df.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")

    def _safe_read_csv(self, filepath: str, expected_cols: list,
                       table_name: str, **kwargs) -> Optional[pd.DataFrame]:
        if not os.path.isfile(filepath):
            logger.warning("File not found: %s — skipping %s", filepath, table_name)
            return None
        try:
            df = pd.read_csv(filepath, **kwargs)
            self._validate_columns(df, expected_cols, table_name)
            logger.info("Loaded %s: %d rows x %d cols",
                         table_name, df.shape[0], df.shape[1])
            return df
        except Exception as exc:
            logger.error("Failed to load %s: %s", filepath, exc)
            return None

    def load_raw_data(self) -> pd.DataFrame:
        orders = self._safe_read_csv(
            os.path.join(self.data_dir, ORDERS_FILE),
            REQUIRED_COLUMNS["orders"], "orders",
        )
        customers = self._safe_read_csv(
            os.path.join(self.data_dir, CUSTOMERS_FILE),
            REQUIRED_COLUMNS["customers"], "customers",
        )
        reviews = self._safe_read_csv(
            os.path.join(self.data_dir, REVIEWS_FILE),
            REQUIRED_COLUMNS["reviews"], "reviews",
        )
        payments = self._safe_read_csv(
            os.path.join(self.data_dir, PAYMENTS_FILE),
            REQUIRED_COLUMNS["payments"], "payments",
        )
        items = self._safe_read_csv(
            os.path.join(self.data_dir, ITEMS_FILE),
            REQUIRED_COLUMNS["items"], "order_items",
        )
        products = self._safe_read_csv(
            os.path.join(self.data_dir, PRODUCTS_FILE),
            REQUIRED_COLUMNS["products"], "products",
        )
        sellers = self._safe_read_csv(
            os.path.join(self.data_dir, SELLERS_FILE),
            REQUIRED_COLUMNS["sellers"], "sellers",
        )

        if orders is None:
            raise FileNotFoundError(
                f"Required file {ORDERS_FILE} not found in {self.data_dir}"
            )

        orders = orders.copy()
        if customers is not None:
            orders = orders.merge(
                customers[["customer_id", "customer_unique_id"]],
                on="customer_id", how="left",
            )
        else:
            logger.warning("Customers table missing — using customer_id as unique_id")
            orders["customer_unique_id"] = orders["customer_id"]

        if reviews is not None:
            reviews_dedup = (
                reviews.sort_values("review_creation_date")
                .drop_duplicates(subset="order_id", keep="first")
            )
            orders = orders.merge(
                reviews_dedup[["order_id", "review_score"]],
                on="order_id", how="left",
            )

        if payments is not None:
            pay_agg = payments.groupby("order_id", as_index=False).agg(
                payment_installments=("payment_installments", "sum"),
                payment_value=("payment_value", "sum"),
                payment_type=(
                    "payment_type",
                    lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown",
                ),
                payment_sequential_count=("payment_sequential", "count"),
            )
            orders = orders.merge(pay_agg, on="order_id", how="left")

        if items is not None:
            item_agg = items.groupby("order_id", as_index=False).agg(
                product_id=("product_id",
                            lambda x: x.iloc[0] if not x.empty else "unknown"),
                price=("price", "sum"),
                freight_value=("freight_value", "sum"),
                item_count=("order_item_id", "count"),
            )
            orders = orders.merge(item_agg, on="order_id", how="left")

        if products is not None and "product_id" in orders.columns:
            orders = orders.merge(
                products[["product_id", "product_category_name"]],
                on="product_id", how="left",
            )

        n_before = orders["order_id"].nunique() if "order_id" in orders.columns else 0
        orders = orders.drop_duplicates(subset="order_id", keep="first")
        n_after = orders["order_id"].nunique()
        if n_before > n_after:
            logger.info("Deduplicated orders: %d → %d unique orders", n_before, n_after)

        logger.info("Final merged dataset: %d rows x %d cols",
                     orders.shape[0], orders.shape[1])
        return orders

    # ── Preprocessing ────────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        date_cols = [
            "order_purchase_timestamp", "order_approved_at",
            "order_delivered_carrier_date", "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "shipping_limit_date", "review_creation_date",
            "review_answer_timestamp",
        ]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        before = len(df)
        df = df.dropna(subset=["order_purchase_timestamp"])
        dropped = before - len(df)
        if dropped:
            logger.info("Dropped %d rows with null purchase timestamp", dropped)

        before = len(df)
        ts_min = pd.Timestamp(TIMESTAMP_MIN)
        ts_max = pd.Timestamp(TIMESTAMP_MAX)
        valid = (df["order_purchase_timestamp"] >= ts_min) & (
            df["order_purchase_timestamp"] <= ts_max
        )
        df = df[valid].copy()
        filtered = before - len(df)
        if filtered:
            logger.info("Filtered %d rows outside [%s, %s]",
                         filtered, TIMESTAMP_MIN, TIMESTAMP_MAX)

        for col in ["price", "freight_value"]:
            if col in df.columns:
                df = df[df[col] >= 0].copy()

        for col in ["price", "freight_value"]:
            if col in df.columns:
                cap = df[col].quantile(OUTLIER_PRICING_PERCENTILE)
                if cap > 0 and not np.isnan(cap):
                    df[col] = df[col].clip(upper=cap)

        if "review_score" in df.columns:
            med = df["review_score"].median()
            if pd.isna(med):
                med = 5.0
            df["review_score"] = df["review_score"].fillna(med)

        for col in ["payment_installments", "payment_value", "price", "freight_value"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    # ── Schema standardisation ───────────────────────────────────────

    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        mapping = {
            "order_purchase_timestamp": "event_time",
            "payment_value": "transaction_value",
        }
        df = df.rename(columns=mapping, errors="ignore")

        # In olist, customer_id is order-level (unique per order).
        # customer_unique_id is the actual customer identifier.
        # We must use customer_unique_id for correct churn labeling.
        if "customer_unique_id" in df.columns:
            df = df.rename(columns={
                "customer_id": "order_customer_id",
                "customer_unique_id": "customer_id",
            })

        # Add event_type — all rows are purchases for Olist
        df["event_type"] = "purchase"

        # Derive delivery_delay
        if ("order_delivered_customer_date" in df.columns
                and "order_estimated_delivery_date" in df.columns):
            delay = (
                df["order_delivered_customer_date"]
                - df["order_estimated_delivery_date"]
            ).dt.days
            df["delivery_delay"] = delay.fillna(0).astype(float)
        else:
            df["delivery_delay"] = 0.0

        if "review_score" not in df.columns:
            df["review_score"] = 0.0

        if "payment_type" not in df.columns:
            df["payment_type"] = "unknown"

        logger.info(
            "Standardised schema — columns: %s", list(df.columns)
        )
        return df

    # ── Feature groups ───────────────────────────────────────────────

    @property
    def available_feature_groups(self) -> List[str]:
        return ["purchase", "monetary", "inactivity", "review",
                "delivery", "payment", "cadence"]

    # ── Metadata ─────────────────────────────────────────────────────

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": "olist",
            "ecosystem_type": "transactional_marketplace",
            "citation": (
                "Olist, Brazilian E-Commerce Public Dataset by Olist. "
                "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
            ),
            "source_url": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
            "n_customers_approx": 100_000,
            "n_orders_approx": 100_000,
            "churn_window_days": 180,
            "churn_justification": (
                "180 days chosen based on Olist's sparse repeat-purchase "
                "behaviour; median inter-purchase interval is ~60 days, "
                "so 3x median covers 95%+ of repurchase gap."
            ),
            "uses_native_churn_label": False,
            "available_feature_groups": self.available_feature_groups,
        }
