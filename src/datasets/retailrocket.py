"""
RetailRocket clickstream dataset adapter.

Ecosystem type: clickstream_commerce
Churn window: 30 days of inactivity

RetailRocket contains browsing events (view, addtocart, transaction)
from an e-commerce website over 4.5 months.  Session-level engagement
is directly observable, making this the strongest dataset for studying
engagement-driven churn.

Data source: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

Memory-optimised for Kaggle (16 GB limit).  Key strategies:
  - Efficient dtypes on CSV load (int32, category)
  - Early filtering of item properties to only items in events
  - Chunked processing of ~11M-row property tables
  - Aggressive intermediate cleanup (del + gc.collect)
  - No unnecessary DataFrame copies
"""
import os
import gc
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

from src.datasets.base import BaseDatasetAdapter
from src.config import ON_KAGGLE, RETAILROCKET_EVENTS
from src.utils import get_logger

logger = get_logger(__name__)

EVENTS_FILE = "retailrocket_events.csv"
ITEMS_FILE = "retailrocket_items.csv"
CATEGORY_FILE = "retailrocket_category_tree.csv"
VISITS_FILE = "retailrocket_visits.csv"  # optional session-level

# Property chunk size — balances I/O throughput vs peak memory
_PROP_CHUNK_SIZE = 500_000

# Target upper-bound (bytes) for the wide property table after pivoting.
# Columns below this threshold are greedily dropped to stay under the cap.
_PROP_WIDE_TARGET_BYTES = 50 * 1024 * 1024  # 50 MB


# ── Module-level helpers (avoid re-creation on every call) ────────────

def _read_csv_optimised(filepath, table_name, usecols=None, dtypes=None,
                        relevant_items=None, **kwargs):
    """Read a CSV once with optimised dtypes, optionally filtering rows."""
    logger.info("Loading %s from %s ...", table_name, filepath)
    try:
        df = pd.read_csv(
            filepath,
            usecols=usecols,
            dtype=dtypes,
            **kwargs,
        )
    except Exception as exc:
        logger.error("Failed to read %s: %s", filepath, exc)
        return None

    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    logger.info("Loaded %s: %d rows x %d cols (%.1f MB)",
                table_name, len(df), len(df.columns), mem_mb)
    return df


def _optimise_string_columns(df):
    """Convert repeated object columns to category where beneficial."""
    for col in df.select_dtypes(include="object").columns:
        nunique = df[col].nunique()
        ratio = nunique / max(len(df), 1)
        if ratio < 0.5:
            before = df[col].memory_usage(deep=True)
            df[col] = df[col].astype("category")
            after = df[col].memory_usage(deep=True)
            logger.info("  category(%s): %d unique, saved %.1f MB",
                        col, nunique, (before - after) / 1024**2)
    return df


def _downcast_numerics(df):
    """Downcast int64/float64 to smaller types where safe."""
    for col in df.select_dtypes(include=["int64"]).columns:
        lo, hi = df[col].min(), df[col].max()
        for dtype in (np.int32, np.int16, np.int8):
            if lo >= np.iinfo(dtype).min and hi <= np.iinfo(dtype).max:
                before = df[col].nbytes
                df[col] = df[col].astype(dtype)
                saved = (before - df[col].nbytes) / 1024**2
                if saved > 0.01:
                    logger.info("  downcast %s: int64 -> %s (%.2f MB saved)",
                                col, dtype.__name__, saved)
                break
    for col in df.select_dtypes(include=["float64"]).columns:
        if df[col].max() <= np.finfo(np.float32).max:
            before = df[col].nbytes
            df[col] = df[col].astype(np.float32)
            saved = (before - df[col].nbytes) / 1024**2
            if saved > 0.01:
                logger.info("  downcast %s: float64 -> float32 (%.2f MB saved)",
                            col, saved)
    return df


def _optimise_df(df, label=""):
    """Apply all dtype optimisations to a DataFrame."""
    before = df.memory_usage(deep=True).sum() / 1024**2
    df = _downcast_numerics(df)
    df = _optimise_string_columns(df)
    after = df.memory_usage(deep=True).sum() / 1024**2
    logger.info("Memory optimisation [%s]: %.1f MB -> %.1f MB (%.1f MB saved)",
                label or "df", before, after, before - after)
    return df


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

    # ── Property file discovery ──────────────────────────────────────

    def _find_property_files(self):
        """Detect property files available in the data directory.

        Returns (prop_files, items_file_or_None).
        On Kaggle the dataset ships ``item_properties_part{1,2}.csv``
        (~11 M rows total).  Some pre-processed copies ship a single
        ``retailrocket_items.csv`` instead.
        """
        items_path = os.path.join(self.data_dir, ITEMS_FILE)
        prop_parts = []
        for name in ("item_properties_part1.csv", "item_properties_part2.csv"):
            p = os.path.join(self.data_dir, name)
            if os.path.isfile(p):
                prop_parts.append(p)
        return prop_parts, items_path if os.path.isfile(items_path) else None

    # ── Chunked property loading & aggregation ───────────────────────

    def _load_properties_chunked(self, prop_files, relevant_items):
        """Read property CSVs in chunks, keeping only the latest value
        per (itemid, property) for items that appear in events.

        Returns a wide-format DataFrame with one row per item and one
        column per property, or None on failure.
        """
        # {itemid: {property: (timestamp, value)}}
        latest: Dict[str, Dict[str, tuple]] = {}
        item_set = set(relevant_items)
        total_rows = 0

        for fi, filepath in enumerate(prop_files, 1):
            reader = pd.read_csv(
                filepath,
                usecols=["timestamp", "itemid", "property", "value"],
                dtype={"itemid": str, "property": str, "value": str},
                chunksize=_PROP_CHUNK_SIZE,
            )
            total_chunks = max(1, os.path.getsize(filepath) //
                               (_PROP_CHUNK_SIZE * 200))  # rough estimate
            for ci, chunk in enumerate(reader, 1):
                total_rows += len(chunk)
                # Filter to relevant items immediately
                mask = chunk["itemid"].isin(item_set)
                chunk = chunk.loc[mask]
                if chunk.empty:
                    if ci % 5 == 0:
                        logger.info("  [%d/%d] chunk %d — 0 relevant rows "
                                    "(skipped)", fi, len(prop_files), ci)
                    continue
                # Update latest per (itemid, property)
                for item_id, grp in chunk.groupby("itemid"):
                    item_d = latest.setdefault(item_id, {})
                    for _, row in grp.iterrows():
                        prop = row["property"]
                        existing = item_d.get(prop)
                        if existing is None or row["timestamp"] > existing[0]:
                            item_d[prop] = (row["timestamp"], row["value"])
                if ci % 5 == 0:
                    logger.info("  [%d/%d] chunk %d processed, "
                                "%d items tracked so far",
                                fi, len(prop_files), ci, len(latest))
                del chunk
                gc.collect()

        logger.info("Loaded %d raw property rows, tracking %d items",
                     total_rows, len(latest))
        if not latest:
            logger.warning("No property data survived filtering — "
                           "returning None")
            return None

        # ── Build value-only DataFrame ──────────────────────────────
        value_dicts = {}
        for item_id, props in latest.items():
            value_dicts[item_id] = {p: v for p, (_, v) in props.items()}
        del latest
        gc.collect()

        prop_df = pd.DataFrame.from_dict(value_dicts, orient="index")
        prop_df.index = prop_df.index.astype(object)  # match string itemid key
        prop_df.index.name = "itemid"
        logger.info("Property table: %d items x %d properties",
                     len(prop_df), len(prop_df.columns))
        return prop_df

    # ── Narrow down wide property table ──────────────────────────────

    @staticmethod
    def _narrow_property_table(prop_df):
        """Drop low-value columns and downcast numerics to meet the
        memory budget ``_PROP_WIDE_TARGET_BYTES``.
        """
        mem = prop_df.memory_usage(deep=True).sum()
        logger.info("Property table before narrowing: %.1f MB",
                     mem / 1024**2)

        # 1. Drop columns that are mostly null
        null_frac = prop_df.isnull().mean()
        high_null = null_frac[null_frac > 0.95].index.tolist()
        if high_null:
            logger.info("  Dropping %d columns > 95%% null", len(high_null))
            prop_df = prop_df.drop(columns=high_null)

        # 2. Downcast remaining columns
        prop_df = _downcast_numerics(prop_df)

        # 3. Drop columns with <= 1 unique value
        low_card = [c for c in prop_df.columns if prop_df[c].nunique() <= 1]
        if low_card:
            logger.info("  Dropping %d constant columns", len(low_card))
            prop_df = prop_df.drop(columns=low_card)

        # 4. If still above target, greedily drop the smallest-value columns
        mem = prop_df.memory_usage(deep=True).sum()
        if mem > _PROP_WIDE_TARGET_BYTES:
            logger.info("  Above target (%.1f MB), trimming columns...",
                        mem / 1024**2)
            col_savings = {}
            for c in prop_df.columns:
                col_savings[c] = prop_df[c].memory_usage(deep=True)
            sorted_cols = sorted(col_savings, key=col_savings.get)
            for c in sorted_cols:
                if mem <= _PROP_WIDE_TARGET_BYTES:
                    break
                mem -= col_savings.pop(c)
                prop_df = prop_df.drop(columns=[c])
            logger.info("  After trimming: %d columns, %.1f MB",
                        len(prop_df.columns), mem / 1024**2)

        prop_df = _downcast_numerics(prop_df)
        final = prop_df.memory_usage(deep=True).sum() / 1024**2
        logger.info("Property table after narrowing: %.1f MB", final)
        return prop_df

    # ── Data loading ─────────────────────────────────────────────────

    def _safe_read_csv(self, filepath: str, table_name: str,
                       **kwargs) -> Optional[pd.DataFrame]:
        if not os.path.isfile(filepath):
            logger.warning("File not found: %s — skipping %s",
                           filepath, table_name)
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
            RETAILROCKET_EVENTS if ON_KAGGLE
            else os.path.join(self.data_dir, EVENTS_FILE)
        )

        # ── Load events with optimised dtypes ───────────────────────
        events = _read_csv_optimised(
            events_path, "events",
            dtypes={"visitorid": str, "itemid": str,
                    "event": str, "transactionid": str},
        )
        if events is None:
            raise FileNotFoundError(
                f"Required file {EVENTS_FILE} not found at {events_path}"
            )

        mem_before = events.memory_usage(deep=True).sum() / 1024**2

        # ── Parse timestamps in-place (no copy) ─────────────────────
        if "timestamp" in events.columns:
            events["timestamp"] = pd.to_datetime(
                events["timestamp"], unit="ms", errors="coerce",
            )

        # ── Downcast string ID columns to category ──────────────────
        for col in ("visitorid", "itemid"):
            if col in events.columns:
                events[col] = events[col].astype("category")

        mem_after = events.memory_usage(deep=True).sum() / 1024**2
        logger.info("Events dtype optimisation: %.1f MB -> %.1f MB",
                     mem_before, mem_after)

        # ── Collect item IDs in events for early property filtering ──
        event_items = set(events["itemid"].cat.categories) \
            if events["itemid"].dtype.name == "category" \
            else set(events["itemid"].unique())
        logger.info("Unique item IDs in events: %d", len(event_items))

        # ── Load items and/or properties ─────────────────────────────
        prop_files, items_path = self._find_property_files()

        if items_path is not None:
            items = self._safe_read_csv(items_path, "items")
            if items is not None and "itemid" in events.columns:
                events = events.merge(items, on="itemid", how="left")
                del items
                gc.collect()
        elif prop_files:
            prop_df = self._load_properties_chunked(prop_files, event_items)
            if prop_df is not None:
                prop_df = self._narrow_property_table(prop_df)
                events = events.merge(
                    prop_df, left_on="itemid", right_index=True, how="left",
                )
                del prop_df
                gc.collect()
        else:
            logger.info("No item/property files found — events-only mode")

        # ── Drop events for items not in any item/property table ─────
        if "timestamp" in events.columns:
            n_before = len(events)
            # Only filter if we successfully loaded properties
            # (otherwise keep all events — they're already relevant)
            if prop_files and "itemid" in events.columns:
                events = events.dropna(subset=["itemid"])
                n_dropped = n_before - len(events)
                if n_dropped:
                    logger.info("Dropped %d events with unmatched itemids",
                                n_dropped)

        mem_final = events.memory_usage(deep=True).sum() / 1024**2
        logger.info("Final events dataset: %d rows x %d cols (%.1f MB)",
                     events.shape[0], events.shape[1], mem_final)
        return events

    # ── Preprocessing ────────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Avoid a full copy — operate in-place where possible
        if "timestamp" in df.columns:
            before = len(df)
            df.dropna(subset=["timestamp"], inplace=True)
            dropped = before - len(df)
            if dropped:
                logger.info("Dropped %d rows with null timestamp", dropped)

        if "visitorid" in df.columns:
            before = len(df)
            df.dropna(subset=["visitorid"], inplace=True)
            dropped = before - len(df)
            if dropped:
                logger.info("Dropped %d rows with null visitorid", dropped)

        if "event" in df.columns:
            before = len(df)
            df = df[df["event"].notna()]
            dropped = before - len(df)
            if dropped:
                logger.info("Dropped %d rows with null event", dropped)

        # Early deduplication: if two events share the same
        # (visitorid, itemid, event, timestamp), keep one.
        dedup_cols = [c for c in ("visitorid", "itemid", "event", "timestamp")
                      if c in df.columns]
        before = len(df)
        df.drop_duplicates(subset=dedup_cols, keep="first", inplace=True)
        dupes = before - len(df)
        if dupes:
            logger.info("Removed %d duplicate events", dupes)

        gc.collect()
        logger.info("After preprocessing: %d rows", len(df))
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
            df["event_type"] = (
                df["event"].map(event_type_map).fillna("other").astype("category")
            )
        else:
            df["event_type"] = pd.Categorical(["purchase"] * len(df))

        # Monetary value only available for transaction events
        if "transaction_value" not in df.columns:
            df["transaction_value"] = 0.0

        if "review_score" not in df.columns:
            df["review_score"] = 0.0

        if "payment_type" not in df.columns:
            df["payment_type"] = "unknown"

        df["delivery_delay"] = 0.0

        # Optimise the newly-created string column
        if df["payment_type"].dtype == object:
            df["payment_type"] = df["payment_type"].astype("category")

        logger.info(
            "Standardised schema — columns: %s, event types: %s",
            list(df.columns), df["event_type"].unique(),
        )
        return df

    # ── Feature groups ───────────────────────────────────────────────

    @property
    def available_feature_groups(self) -> List[str]:
        """RetailRocket has strong engagement observability."""
        return ["purchase", "monetary", "inactivity", "engagement",
                "cadence"]

    # ── User-disjoint temporal split ─────────────────────────────────

    @property
    def uses_user_disjoint_split(self) -> bool:
        """RetailRocket uses a user-disjoint temporal holdout to stop the
        same visitor appearing in both train and test (the plain global
        temporal cutoffs overlap ~86% of test users with train)."""
        return True

    def build_user_disjoint_modeling_data(self, df: pd.DataFrame) -> tuple:
        """Build user-disjoint train/test features + labels via the shared
        temporal helper (late-arrival test cohort)."""
        from src.user_disjoint_split import build_user_disjoint_modeling_data
        if self.churn_window_days is None:
            raise RuntimeError(
                "RetailRocket requires a churn window for UD split"
            )
        X_train, X_test, y_train, y_test = build_user_disjoint_modeling_data(
            df,
            churn_window_days=self.churn_window_days,
            feature_groups=self.available_feature_groups,
        )
        return X_train, X_test, y_train, y_test

    # ── Metadata ─────────────────────────────────────────────────────

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": "retailrocket",
            "ecosystem_type": "clickstream_commerce",
            "citation": (
                "RetailRocket, E-commerce Clickstream Data. "
                "https://www.kaggle.com/datasets/retailrocket/"
                "ecommerce-dataset"
            ),
            "source_url": (
                "https://www.kaggle.com/datasets/retailrocket/"
                "ecommerce-dataset"
            ),
            "n_customers_approx": 1_400_000,
            "n_events_approx": 2_700_000,
            "churn_window_days": 30,
            "churn_justification": (
                "30 days — RetailRocket spans only ~4.5 months; a longer "
                "window would consume too much data.  Clickstream users "
                "churn faster than marketplace buyers."
            ),
            "uses_native_churn_label": False,
            "available_feature_groups": self.available_feature_groups,
        }
