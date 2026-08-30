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
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

from src.datasets.base import BaseDatasetAdapter
from src.config import ON_KAGGLE, INSTACART_DIR, RANDOM_SEED
from src.utils import get_logger

logger = get_logger(__name__)

ORDERS_FILE = "orders.csv"
PRODUCTS_FILE = "products.csv"
AISLES_FILE = "aisles.csv"
DEPARTMENTS_FILE = "departments.csv"
ORDER_PRODUCTS_PRIOR = "order_products__prior.csv"


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
    def data_dir(self) -> str:
        return INSTACART_DIR if ON_KAGGLE else super().data_dir

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
            frac = max_rows / len(df)
            df = df.sample(frac=frac, random_state=42)
            logger.warning(
                "%s large (%d rows) — sampled to %d rows for memory safety",
                name, len(df), len(df),
            )
        return df

    @property
    def uses_user_relative_churn_label(self) -> bool:
        return True

    def load_raw_data(self) -> pd.DataFrame:
        orders = self._safe_read_csv(
            os.path.join(self.data_dir, ORDERS_FILE), "orders",
            dtype={"order_id": int, "user_id": str, "eval_set": str},
        )
        products = self._safe_read_csv(
            os.path.join(self.data_dir, PRODUCTS_FILE), "products",
        )
        aisles = self._safe_read_csv(
            os.path.join(self.data_dir, AISLES_FILE), "aisles",
        )
        departments = self._safe_read_csv(
            os.path.join(self.data_dir, DEPARTMENTS_FILE), "departments",
        )
        order_products = self._safe_read_csv(
            os.path.join(self.data_dir, ORDER_PRODUCTS_PRIOR),
            "order_products__prior",
        )

        if orders is None:
            raise FileNotFoundError(
                f"Required file {ORDERS_FILE} not found in {self.data_dir}"
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

        # Build synthetic event_time from order_number and days_since_prior_order
        # We approximate a timeline: assume most recent order = max_date
        if "days_since_prior_order" in df.columns and "order_number" in df.columns:
            df["days_since_prior_order"] = (
                pd.to_numeric(df["days_since_prior_order"], errors="coerce").fillna(0)
            )
            max_days = df["days_since_prior_order"].sum()
            df["days_from_end"] = (
                df.groupby("user_id")["days_since_prior_order"]
                .cumsum().fillna(0)
            )
            df["days_from_end"] = df.groupby("user_id")["days_from_end"].transform(
                lambda x: x.max() - x
            )
            df["event_time"] = pd.Timestamp("2017-03-21") - pd.to_timedelta(
                df["days_from_end"], unit="D"
            )
        else:
            df["event_time"] = pd.Timestamp("2017-03-21")

        if "user_id" in df.columns:
            df = df.dropna(subset=["user_id"])

        if "purchase_value" not in df.columns:
            df["purchase_value"] = 0.0

        return df

    def user_relative_split(
        self, orders_df: pd.DataFrame, train_ratio: float = 0.70,
    ) -> tuple:
        """Split each user's order history independently by order number."""
        train_orders = []
        test_orders = []
        id_col = "user_id" if "user_id" in orders_df.columns else "customer_id"

        for _, user_orders in orders_df.groupby(id_col):
            user_orders_sorted = user_orders.sort_values("order_number")
            n = len(user_orders_sorted)
            split_idx = max(1, int(n * train_ratio))
            train_orders.append(user_orders_sorted.iloc[:split_idx])
            test_orders.append(user_orders_sorted.iloc[split_idx:])

        if not train_orders:
            return pd.DataFrame(), pd.DataFrame()
        train_df = pd.concat(train_orders, ignore_index=True)
        test_df = (
            pd.concat(test_orders, ignore_index=True)
            if test_orders else pd.DataFrame(columns=orders_df.columns)
        )
        return train_df, test_df

    def build_user_relative_modeling_data(
        self, df: pd.DataFrame, train_ratio: float = 0.70,
    ) -> tuple:
        """Build user-level features and churn labels from personal cadence decay.

        Users with fewer than three unique orders are excluded.  A user is labeled
        churned when mean test-period days_since_prior_order exceeds 1.5 times
        the mean train-period days_since_prior_order.
        """
        from sklearn.model_selection import train_test_split
        from src.feature_engineering import engineer_features

        required = {"customer_id", "order_number", "days_since_prior_order"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Instacart user-relative split missing columns: {missing}")

        order_level = (
            df.sort_values(["customer_id", "order_number"])
            .drop_duplicates(subset=["customer_id", "order_number"])
            .copy()
        )
        counts = order_level.groupby("customer_id")["order_number"].nunique()
        eligible_users = counts[counts >= 3].index
        order_level = order_level[order_level["customer_id"].isin(eligible_users)].copy()
        event_rows = df[df["customer_id"].isin(eligible_users)].copy()

        label_rows = []
        feature_rows = []
        for user_id, user_orders in order_level.groupby("customer_id"):
            user_orders = user_orders.sort_values("order_number")
            split_idx = max(1, int(len(user_orders) * train_ratio))
            train_orders = user_orders.iloc[:split_idx]
            test_orders = user_orders.iloc[split_idx:]
            if test_orders.empty:
                continue

            train_gap = pd.to_numeric(
                train_orders["days_since_prior_order"], errors="coerce",
            ).replace(0, np.nan).dropna()
            test_gap = pd.to_numeric(
                test_orders["days_since_prior_order"], errors="coerce",
            ).replace(0, np.nan).dropna()
            if train_gap.empty or test_gap.empty:
                continue

            train_mean = train_gap.mean()
            test_mean = test_gap.mean()
            churn = int(test_mean > 1.5 * train_mean)
            label_rows.append({"customer_id": user_id, "churn": churn})
            feature_rows.append(
                event_rows[
                    (event_rows["customer_id"] == user_id)
                    & (event_rows["order_number"].isin(train_orders["order_number"]))
                ]
            )

        if not label_rows or not feature_rows:
            raise RuntimeError("No eligible Instacart users for user-relative modeling")

        labels = pd.DataFrame(label_rows).drop_duplicates("customer_id")
        feature_source = pd.concat(feature_rows, ignore_index=True)
        snapshot = feature_source["event_time"].max() + pd.Timedelta(days=1)
        features = engineer_features(
            feature_source,
            snapshot,
            customer_ids=labels["customer_id"].tolist(),
            available_groups=self.available_feature_groups,
        )
        labels = labels.set_index("customer_id").loc[features.index]
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels["churn"],
            test_size=0.30,
            random_state=RANDOM_SEED,
            stratify=labels["churn"],
        )
        return X_train, X_test, y_train.to_frame("churn"), y_test.to_frame("churn")

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
        return ["purchase", "monetary", "inactivity", "cadence"]

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
