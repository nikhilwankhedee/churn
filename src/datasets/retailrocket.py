"""
RetailRocket clickstream dataset adapter.

Ecosystem type: clickstream_commerce
Churn window: 30 days of inactivity

RetailRocket contains browsing events (view, addtocart, transaction)
from an e-commerce website over 4.5 months.  Session-level engagement
is directly observable, making this the strongest dataset for studying
engagement-driven churn.

Data source: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
"""
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.datasets.base import BaseDatasetAdapter
from src.utils import get_logger

logger = get_logger(__name__)

EVENTS_FILE = "retailrocket_events.csv"
EVENTS_FILE_KAGGLE = "events.csv"
ITEMS_FILE = "retailrocket_items.csv"
ITEMS_FILE_KAGGLE = "item_properties_part1.csv"
CATEGORY_FILE = "retailrocket_category_tree.csv"
CATEGORY_FILE_KAGGLE = "category_tree.csv"
VISITS_FILE = "retailrocket_visits.csv"  # optional session-level


class RetailRocketAdapter(BaseDatasetAdapter):
    @property
    def dataset_name(self) -> str:
        return "retailrocket"

    @property
    def ecosystem_type(self) -> str:
        return "clickstream_commerce"

    @property
    def churn_window_days(self) -> Optional[int]:
        return 30

    @property
    def required_files(self) -> list:
        return [EVENTS_FILE]

    @property
    def alternate_filenames(self) -> Dict[str, List[str]]:
        """Kaggle ships these files under different names than the local
        build.  The resolver treats a required file as present if any
        alternate exists (e.g. ``events.csv`` for ``retailrocket_events.csv``)."""
        return {
            EVENTS_FILE: [EVENTS_FILE_KAGGLE],
            ITEMS_FILE: [ITEMS_FILE_KAGGLE],
            CATEGORY_FILE: [CATEGORY_FILE_KAGGLE],
        }

    def _resolve_file(self, preferred: str, fallback: str) -> str:
        """Return whichever filename exists in data_dir."""
        primary = os.path.join(self.data_dir, preferred)
        if os.path.isfile(primary):
            return primary
        secondary = os.path.join(self.data_dir, fallback)
        if os.path.isfile(secondary):
            return secondary
        return primary  # return primary for error messaging

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

    def _deduplicate_items(self, items: pd.DataFrame) -> pd.DataFrame:
        """Collapse the Kaggle ``item_properties_part1.csv`` (long-format:
        one row per item×property) to one row per itemid so that merging with
        events does not blow up the row count via a cartesian product."""
        if "itemid" not in items.columns:
            logger.warning("Items table has no itemid column — skipping merge")
            return items
        n_before = len(items)
        duplicated = items.duplicated(subset=["itemid"])
        if not duplicated.any():
            return items
        if {"property", "value"}.issubset(items.columns):
            agg = items.groupby("itemid").agg(
                n_properties=("property", "size"),
                n_distinct_properties=("property", "nunique"),
            ).reset_index()
            logger.info(
                "Items table was long-format (%d rows) — collapsed to %d rows",
                n_before, len(agg),
            )
            return agg
        items = items.drop_duplicates(subset=["itemid"]).copy()
        logger.info(
            "Items table had duplicated itemid (%d rows) — collapsed to %d rows",
            n_before, len(items),
        )
        return items

    def load_raw_data(self) -> pd.DataFrame:
        events_path = self._resolve_file(EVENTS_FILE, EVENTS_FILE_KAGGLE)
        items_path = self._resolve_file(ITEMS_FILE, ITEMS_FILE_KAGGLE)

        events = self._safe_read_csv(
            events_path, "events",
            dtype={"visitorid": str, "itemid": str,
                   "event": str, "transactionid": str},
        )
        items = self._safe_read_csv(
            items_path, "items",
        )

        if events is None:
            raise FileNotFoundError(
                f"Neither {EVENTS_FILE} nor {EVENTS_FILE_KAGGLE} "
                f"found in {self.data_dir}"
            )

        events = events.copy()

        if "timestamp" in events.columns:
            events["timestamp"] = pd.to_datetime(
                events["timestamp"], unit="ms", errors="coerce"
            )

        if items is not None and "itemid" in events.columns:
            items = self._deduplicate_items(items)
            events = events.merge(items, on="itemid", how="left")

        logger.info("Final events dataset: %d rows x %d cols",
                     events.shape[0], events.shape[1])
        return events

    # ── Preprocessing ────────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "timestamp" in df.columns:
            before = len(df)
            df = df.dropna(subset=["timestamp"])
            dropped = before - len(df)
            if dropped:
                logger.info("Dropped %d rows with null timestamp", dropped)

        if "visitorid" in df.columns:
            df = df.dropna(subset=["visitorid"])

        if "event" in df.columns:
            df = df[df["event"].notna()].copy()

        return df

    # ── Schema standardisation ───────────────────────────────────────

    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        event_type_map = {
            "view": "view",
            "addtocart": "cart_add",
            "transaction": "purchase",
        }

        mapping = {
            "visitorid": "customer_id",
            "timestamp": "event_time",
            "itemid": "product_id",
            "transactionid": "session_id",
        }
        df = df.rename(columns=mapping, errors="ignore")

        if "event" in df.columns:
            df["event_type"] = df["event"].map(event_type_map).fillna("other")
        else:
            df["event_type"] = "purchase"

        # Monetary value only available for transaction events
        # Use transactionid presence as proxy; price info not in events table
        if "transaction_value" not in df.columns:
            df["transaction_value"] = 0.0

        if "review_score" not in df.columns:
            df["review_score"] = 0.0

        if "payment_type" not in df.columns:
            df["payment_type"] = "unknown"

        df["delivery_delay"] = 0.0

        logger.info(
            "Standardised schema — columns: %s, event types: %s",
            list(df.columns), df["event_type"].unique(),
        )
        return df

    # ── Feature groups ───────────────────────────────────────────────

    @property
    def available_feature_groups(self) -> List[str]:
        """RetailRocket has strong engagement observability but no legitimate
        price information → monetary group is disabled (Section 5)."""
        return ["purchase", "inactivity", "engagement",
                "cadence"]

    # ── Metadata ─────────────────────────────────────────────────────

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": "retailrocket",
            "ecosystem_type": "clickstream_commerce",
            "citation": (
                "RetailRocket, E-commerce Clickstream Data. "
                "https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset"
            ),
            "source_url": "https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset",
            "n_customers_approx": 1_400_000,
            "n_events_approx": 2_700_000,
            "churn_window_days": 30,
            "churn_justification": (
                "30 days — RetailRocket spans only ~4.5 months; a longer window "
                "would consume too much data.  Clickstream users churn faster "
                "than marketplace buyers."
            ),
            "uses_native_churn_label": False,
            "available_feature_groups": self.available_feature_groups,
        }
