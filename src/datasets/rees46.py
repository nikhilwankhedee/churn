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
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.datasets.base import BaseDatasetAdapter
from src.utils import get_logger

logger = get_logger(__name__)

EVENTS_FILE = "rees46_events.csv"
EVENTS_FILE_KAGGLE = "events.csv"
USERS_FILE = "rees46_users.csv"
USERS_FILE_KAGGLE = "users.csv"
ITEMS_FILE = "rees46_items.csv"
ITEMS_FILE_KAGGLE = "items.csv"
CUSTOMER_MODEL_FILE = "rees46_customer_model.csv"
CUSTOMER_MODEL_FILE_KAGGLE = "customer_model.csv"


class REES46Adapter(BaseDatasetAdapter):
    @property
    def dataset_name(self) -> str:
        return "rees46"

    @property
    def ecosystem_type(self) -> str:
        return "transactional_marketplace"

    @property
    def churn_window_days(self) -> Optional[int]:
        return 180

    @property
    def required_files(self) -> list:
        return [EVENTS_FILE]

    @property
    def alternate_filenames(self) -> Dict[str, List[str]]:
        """Kaggle ships these files under different names than the local
        build.  The resolver treats a required file as present if any
        alternate exists (e.g. ``events.csv`` for ``rees46_events.csv``)."""
        return {
            EVENTS_FILE: [EVENTS_FILE_KAGGLE],
            USERS_FILE: [USERS_FILE_KAGGLE],
            ITEMS_FILE: [ITEMS_FILE_KAGGLE],
        }

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

    def _resolve_file(self, *candidates: str) -> Optional[str]:
        """Return path of first existing file, or None."""
        for name in candidates:
            path = os.path.join(self.data_dir, name)
            if os.path.isfile(path):
                return path
        return None

    def _load_customer_model_format(self) -> Optional[pd.DataFrame]:
        """Load from rees46_customer_model.csv (single-file format)."""
        cm_path = self._resolve_file(CUSTOMER_MODEL_FILE, CUSTOMER_MODEL_FILE_KAGGLE)
        if cm_path is None:
            return None
        try:
            df = pd.read_csv(cm_path, nrows=5)
            logger.info("Customer model columns: %s", list(df.columns))
            # This format has pre-computed user-level features
            # We'll use it directly as the events source
            df = pd.read_csv(cm_path)
            return df
        except Exception as exc:
            logger.error("Failed to load customer model: %s", exc)
            return None

    def _load_customer_model_format(self) -> Optional[pd.DataFrame]:
        """Deprecated: customer_model has no per-event timestamps.

        Retained only to document why it must not be used as an events
        substitute for temporal churn analysis.  Raises FileNotFoundError
        instead of silently returning non-temporal data.
        """
        raise FileNotFoundError(
            f"customer_model format cannot be used for temporal churn: "
            f"it lacks per-event timestamps.  Provide the full events file "
            f"({EVENTS_FILE}/{EVENTS_FILE_KAGGLE}) instead."
        )

    def load_raw_data(self) -> pd.DataFrame:
        events_path = self._resolve_file(EVENTS_FILE, EVENTS_FILE_KAGGLE)

        if events_path is not None:
            events = self._safe_read_csv(
                events_path, "events",
                dtype={"user_id": str, "item_id": str, "event_type": str},
            )
        else:
            events = None

        if events is None:
            # Refuse to fall back to the user-level customer_model file: it has
            # no per-event timestamps, so it cannot support temporal churn.
            # Requiring it to stand in for events would silently produce
            # non-temporal (and unusable) data.
            raise FileNotFoundError(
                f"No events file found in {self.data_dir}. "
                f"Expected one of {EVENTS_FILE}/{EVENTS_FILE_KAGGLE} "
                f"(the full 'rees46/rees46-marketplace' events.csv). "
                f"A user-level {CUSTOMER_MODEL_FILE}/{CUSTOMER_MODEL_FILE_KAGGLE} "
                f"file alone is insufficient because it lacks per-event "
                f"timestamps."
            )

        users_path = self._resolve_file(USERS_FILE, USERS_FILE_KAGGLE)
        items_path = self._resolve_file(ITEMS_FILE, ITEMS_FILE_KAGGLE)

        users = self._safe_read_csv(users_path, "users") if users_path else None
        items = self._safe_read_csv(items_path, "items") if items_path else None

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

        if "engagement_signal" not in df.columns:
            if "event_type" in df.columns:
                df["engagement_signal"] = df["event_type"].map({
                    "purchase": 3, "cart_add": 2, "view": 1,
                }).fillna(0).astype(float)
            else:
                df["engagement_signal"] = 0.0

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
            "churn_window_days": 180,
            "churn_justification": (
                "180 days — matches Olist for cross-dataset comparability; "
                "REES46 is a similar marketplace with sparse repurchase."
            ),
            "uses_native_churn_label": False,
            "available_feature_groups": self.available_feature_groups,
        }
