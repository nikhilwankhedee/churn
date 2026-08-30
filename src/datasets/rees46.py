"""
REES46 e-commerce dataset adapter.

Ecosystem type: transactional_marketplace
Churn window: 180 days of inactivity

REES46 provides behavioural event data (view, cart, purchase) with
timestamps, prices, and user/session identifiers.  Structurally similar
to Olist but with richer engagement signals.

Data source: https://www.kaggle.com/datasets/rees46/rees46-marketplace
"""
import os
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

from src.datasets.base import BaseDatasetAdapter
from src.config import ON_KAGGLE, REES46_FILE
from src.utils import get_logger

logger = get_logger(__name__)

EVENTS_FILE = "rees46_events.csv"
USERS_FILE = "rees46_users.csv"
ITEMS_FILE = "rees46_items.csv"


class REES46Adapter(BaseDatasetAdapter):
    @property
    def dataset_name(self) -> str:
        return "rees46"

    @property
    def ecosystem_type(self) -> str:
        return "transactional_marketplace"

    @property
    def churn_window_days(self) -> Optional[int]:
        return 90

    # ── Data loading ─────────────────────────────────────────────────

    def _safe_read_csv(self, filepath: str, table_name: str,
                       **kwargs) -> Optional[pd.DataFrame]:
        if not os.path.isfile(filepath):
            logger.warning("File not found: %s — skipping %s", filepath, table_name)
            return None
        try:
            df = pd.read_csv(filepath, **kwargs)
            logger.info("Loaded %s: %d rows x %d cols",
                         table_name, df.shape[0], df.shape[1])
            return df
        except Exception as exc:
            logger.error("Failed to load %s: %s", filepath, exc)
            return None

    def load_raw_data(self) -> pd.DataFrame:
        events_path = (
            REES46_FILE if ON_KAGGLE else os.path.join(self.data_dir, EVENTS_FILE)
        )
        events = self._safe_read_csv(
            events_path, "events",
            dtype={"user_id": str, "item_id": str, "event_type": str},
        )
        users = self._safe_read_csv(
            os.path.join(self.data_dir, USERS_FILE), "users",
        )
        items = self._safe_read_csv(
            os.path.join(self.data_dir, ITEMS_FILE), "items",
        )

        if events is None:
            raise FileNotFoundError(
                f"Required file {EVENTS_FILE} not found in {self.data_dir}"
            )

        events = events.copy()

        if "timestamp" in events.columns:
            events["timestamp"] = pd.to_datetime(
                events["timestamp"], unit="s", errors="coerce"
            )

        if "price" in events.columns:
            events["price"] = pd.to_numeric(events["price"], errors="coerce").fillna(0)

        if users is not None and "user_id" in users.columns:
            events = events.merge(
                users, on="user_id", how="left",
            )

        if items is not None and "item_id" in events.columns:
            events = events.merge(
                items, on="item_id", how="left",
            )

        logger.info("Final merged dataset: %d rows x %d cols",
                     events.shape[0], events.shape[1])
        return events

    # ── Preprocessing ────────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            before = len(df)
            df = df.dropna(subset=["timestamp"])
            dropped = before - len(df)
            if dropped:
                logger.info("Dropped %d rows with null timestamp", dropped)

        if "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
            df = df[df["price"] >= 0].copy()
            cap = df["price"].quantile(0.999)
            if cap > 0 and not np.isnan(cap):
                df["price"] = df["price"].clip(upper=cap)

        if "user_id" in df.columns:
            df = df.dropna(subset=["user_id"])

        return df

    # ── Schema standardisation ───────────────────────────────────────

    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        mapping = {
            "user_id": "customer_id",
            "timestamp": "event_time",
            "price": "transaction_value",
            "item_id": "product_id",
        }
        df = df.rename(columns=mapping, errors="ignore")

        if "event_type" not in df.columns:
            df["event_type"] = "purchase"

        if "review_score" not in df.columns:
            df["review_score"] = 0.0

        if "payment_type" not in df.columns:
            df["payment_type"] = "unknown"

        df["delivery_delay"] = 0.0

        if "session_id" not in df.columns:
            df["session_id"] = "unknown"

        logger.info(
            "Standardised schema — columns: %s", list(df.columns)
        )
        return df

    # ── Feature groups ───────────────────────────────────────────────

    @property
    def available_feature_groups(self) -> List[str]:
        return ["purchase", "monetary", "inactivity", "engagement",
                "cadence"]

    # ── Metadata ─────────────────────────────────────────────────────

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": "rees46",
            "ecosystem_type": "transactional_marketplace",
            "citation": (
                "REES46 Marketplace Dataset. "
                "https://www.kaggle.com/datasets/rees46/rees46-marketplace"
            ),
            "source_url": "https://www.kaggle.com/datasets/rees46/rees46-marketplace",
            "n_customers_approx": 500_000,
            "n_events_approx": 5_000_000,
            "churn_window_days": 90,
            "churn_justification": (
                "90 days — configured methodology for the REES46 customer "
                "modeling dataset."
            ),
            "uses_native_churn_label": False,
            "available_feature_groups": self.available_feature_groups,
        }
