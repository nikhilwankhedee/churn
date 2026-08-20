"""
Instacart grocery dataset adapter.

Ecosystem type: habitual_retail
Churn window: 60 days of inactivity

Instacart is a habitual grocery delivery ecosystem with high repeat
purchase cadence.  Users order weekly/biweekly.  ~3M orders —
may require stratified sampling for memory safety.

Churn in this context means a prolonged gap in order activity; because
the cadence is high, a shorter window than marketplace datasets is justified.

Data source: https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis
"""
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.datasets.base import BaseDatasetAdapter
from src.utils import get_logger

logger = get_logger(__name__)

ORDERS_FILE = "instacart_orders.csv"
ORDERS_FILE_KAGGLE = "orders.csv"
PRODUCTS_FILE = "instacart_products.csv"
PRODUCTS_FILE_KAGGLE = "products.csv"
AISLES_FILE = "instacart_aisles.csv"
AISLES_FILE_KAGGLE = "aisles.csv"
DEPARTMENTS_FILE = "instacart_departments.csv"
DEPARTMENTS_FILE_KAGGLE = "departments.csv"
ORDER_PRODUCTS_PRIOR = "instacart_order_products__prior.csv"
ORDER_PRODUCTS_PRIOR_KAGGLE = "order_products__prior.csv"


class InstacartAdapter(BaseDatasetAdapter):
    @property
    def dataset_name(self) -> str:
        return "instacart"

    @property
    def ecosystem_type(self) -> str:
        return "habitual_retail"

    @property
    def churn_window_days(self) -> Optional[int]:
        return 60

    @property
    def required_files(self) -> list:
        return [ORDERS_FILE]

    @property
    def alternate_filenames(self) -> Dict[str, List[str]]:
        """Kaggle ships these files under their original names (e.g.
        ``orders.csv`` for ``instacart_orders.csv``).  The resolver treats a
        required file as present if any alternate exists."""
        return {
            ORDERS_FILE: [ORDERS_FILE_KAGGLE],
            PRODUCTS_FILE: [PRODUCTS_FILE_KAGGLE],
            AISLES_FILE: [AISLES_FILE_KAGGLE],
            DEPARTMENTS_FILE: [DEPARTMENTS_FILE_KAGGLE],
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

    def _sample_if_large(self, df: pd.DataFrame, max_rows: int = 1_000_000,
                         name: str = "data") -> pd.DataFrame:
        if len(df) > max_rows:
            original_len = len(df)
            frac = max_rows / len(df)
            df = df.sample(frac=frac, random_state=42)
            logger.warning(
                "%s large (%d rows) — sampled to %d rows for memory safety",
                name, original_len, len(df),
            )
        return df

    def _resolve_file(self, *candidates: str) -> Optional[str]:
        """Return path of first existing file, or None."""
        for name in candidates:
            path = os.path.join(self.data_dir, name)
            if os.path.isfile(path):
                return path
        return None

    def load_raw_data(self) -> pd.DataFrame:
        orders_path = self._resolve_file(ORDERS_FILE, ORDERS_FILE_KAGGLE)
        products_path = self._resolve_file(PRODUCTS_FILE, PRODUCTS_FILE_KAGGLE)
        aisles_path = self._resolve_file(AISLES_FILE, AISLES_FILE_KAGGLE)
        departments_path = self._resolve_file(DEPARTMENTS_FILE, DEPARTMENTS_FILE_KAGGLE)
        order_products_path = self._resolve_file(ORDER_PRODUCTS_PRIOR, ORDER_PRODUCTS_PRIOR_KAGGLE)

        orders = self._safe_read_csv(
            orders_path, "orders",
            dtype={"order_id": int, "user_id": str, "eval_set": str},
        ) if orders_path else None
        products = self._safe_read_csv(products_path, "products") if products_path else None
        aisles = self._safe_read_csv(aisles_path, "aisles") if aisles_path else None
        departments = self._safe_read_csv(departments_path, "departments") if departments_path else None
        order_products = self._safe_read_csv(
            order_products_path, "order_products__prior",
        ) if order_products_path else None

        if orders is None:
            raise FileNotFoundError(
                f"Required file {ORDERS_FILE}/{ORDERS_FILE_KAGGLE} "
                f"not found in {self.data_dir}"
            )

        orders = orders.copy()
        orders = self._sample_if_large(orders, max_rows=500_000, name="orders")

        if order_products is not None:
            order_products = self._sample_if_large(
                order_products, max_rows=2_000_000, name="order_products",
            )
            orders = orders.merge(order_products, on="order_id", how="left")
        else:
            logger.warning("Order products not available — no product details")

        if products is not None and "product_id" in orders.columns:
            orders = orders.merge(
                products, on="product_id", how="left",
            )

        if aisles is not None and "aisle_id" in orders.columns:
            orders = orders.merge(aisles, on="aisle_id", how="left")

        if departments is not None and "department_id" in orders.columns:
            orders = orders.merge(departments, on="department_id", how="left")

        logger.info("Final merged dataset: %d rows x %d cols",
                     orders.shape[0], orders.shape[1])
        return orders

    # ── Preprocessing ────────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Build a synthetic event_time from order_number and
        # days_since_prior_order.  Instacart ships no calendar timestamps,
        # so we reconstruct each user's timeline from inter-order gaps.
        #
        # Strategy — rank-based temporal distribution:
        #   1. Preserve each user's real inter-order spacing exactly.
        #   2. Distribute users across a fixed observation window so that
        #      users with longer histories start earlier and users with
        #      shorter histories start later.
        #   3. No user's final order coincides with another's (except by
        #      chance of identical spans AND rank, which is negligible).
        #
        # This avoids the previous approach's failure mode where anchoring
        # all users' final orders to a single horizon date collapsed the
        # temporal distribution and produced degenerate churn labels.
        if "days_since_prior_order" in df.columns and "order_number" in df.columns:
            df["days_since_prior_order"] = (
                pd.to_numeric(df["days_since_prior_order"], errors="coerce").fillna(0)
            )

            # Reconstruct the timeline at order level (product rows within an
            # order must not inflate the cumulative-day sum).
            if {"order_id", "user_id"}.issubset(df.columns):
                order_lvl = (
                    df[["user_id", "order_id", "days_since_prior_order"]]
                    .drop_duplicates(subset=["user_id", "order_id"])
                    .copy()
                )
            else:
                order_lvl = (
                    df[["user_id", "days_since_prior_order"]]
                    .drop_duplicates(subset=["user_id"])
                    .copy()
                )

            # Cumulative days from each user's first order.
            order_lvl["days_from_start"] = (
                order_lvl.groupby("user_id")["days_since_prior_order"]
                .cumsum().fillna(0)
            )

            # Per-user total span (days between first and last order).
            user_spans = order_lvl.groupby("user_id")["days_from_start"].max()
            user_spans.name = "user_span"

            # Rank users by span (ascending) — longer-lived users get earlier
            # start positions so their timelines fit within the window.
            user_ranks = user_spans.rank(method="first", ascending=True) - 1
            n_users = len(user_ranks)

            # Observation window: accommodate the longest user span plus a
            # margin proportional to the span distribution (p99 + 10%).
            p99_span = float(user_spans.quantile(0.99))
            observation_span = max(
                float(user_spans.max()) + 1,
                p99_span * 1.10 + 1,
            )
            timeline_end = pd.Timestamp("2017-03-21")
            timeline_start = timeline_end - pd.Timedelta(days=observation_span)

            # Place each user's first order within the feasible range.
            # Feasible range for user u: [0, observation_span - user_span[u]]
            # Rank-based placement distributes users deterministically across
            # this range, avoiding endpoint collisions.
            user_starts = pd.Series(
                (user_ranks / max(n_users - 1, 1))
                * (observation_span - user_spans),
                index=user_ranks.index,
            )
            user_starts_dict = user_starts.to_dict()

            # Assign event_time = timeline_start + start_offset + days_from_start
            order_lvl["event_time"] = order_lvl.apply(
                lambda r: timeline_start + pd.Timedelta(
                    days=user_starts_dict[r["user_id"]] + r["days_from_start"]
                ),
                axis=1,
            )

            if "order_id" in order_lvl.columns:
                df = df.drop(columns=["event_time"], errors="ignore").merge(
                    order_lvl[["user_id", "order_id", "event_time"]],
                    on=["user_id", "order_id"], how="left",
                )
            else:
                df["event_time"] = df["user_id"].map(
                    order_lvl.set_index("user_id")["event_time"]
                )
        else:
            df["event_time"] = pd.Timestamp("2017-03-21")

        if "user_id" in df.columns:
            df = df.dropna(subset=["user_id"])

        if "purchase_value" not in df.columns:
            df["purchase_value"] = 0.0

        return df

    # ── Schema standardisation ───────────────────────────────────────

    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        mapping = {
            "user_id": "customer_id",
            "purchase_value": "transaction_value",
            "product_id": "product_id",
            "order_id": "session_id",
            "reordered": "engagement_signal",
        }
        df = df.rename(columns=mapping, errors="ignore")

        df["event_type"] = "purchase"

        if "review_score" not in df.columns:
            df["review_score"] = 0.0

        if "payment_type" not in df.columns:
            df["payment_type"] = "unknown"

        df["delivery_delay"] = 0.0

        logger.info(
            "Standardised schema — columns: %s", list(df.columns)
        )
        return df

    # ── Feature groups ───────────────────────────────────────────────

    @property
    def available_feature_groups(self) -> List[str]:
        """Instacart provides no price information (purchase_value is
        unavailable) → monetary group is disabled (Section 5)."""
        return ["purchase", "inactivity", "cadence"]

    # ── Metadata ─────────────────────────────────────────────────────

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": "instacart",
            "ecosystem_type": "habitual_retail",
            "citation": (
                "Instacart Market Basket Analysis. "
                "https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis"
            ),
            "source_url": "https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis",
            "n_customers_approx": 200_000,
            "n_orders_approx": 3_000_000,
            "churn_window_days": 60,
            "churn_justification": (
                "60 days — Instacart has weekly/biweekly ordering cadence; "
                "60 days (~2 months) is ~4× median inter-purchase interval "
                "(~14 days), indicating genuine disengagement."
            ),
            "uses_native_churn_label": False,
            "available_feature_groups": self.available_feature_groups,
        }
