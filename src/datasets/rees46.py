"""
REES46 e-commerce dataset adapter.

Ecosystem type: transactional_marketplace
Churn definition: native label (target_event) — no transaction in the
future period.

The Kaggle version of this dataset ("e-commerce-churn-dataset-rees46")
ships a PRE-AGGREGATED per-customer model (rees46_customer_model.csv,
one row per customer, 276 columns) rather than raw behavioural events.
REES46 therefore consumes the customer model directly:

  - customer identifier : customer_id (first column fallback)
  - churn label         : target_event (native)
  - split               : 70/30 stratified on the label — no temporal
                          split is possible (no event timestamps)
  - features            : every numeric column except the identifier
                          and the label column(s)

Data source: https://www.kaggle.com/datasets/fridrichmrtn/e-commerce-churn-dataset-rees46
"""
import os
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from src.datasets.base import BaseDatasetAdapter
from src.config import DATA_DIR, ON_KAGGLE, REES46_FILE, RANDOM_SEED
from src.utils import get_logger

logger = get_logger(__name__)


LOCAL_REES46_FILE = "rees46_customer_model.csv"


class REES46Adapter(BaseDatasetAdapter):
    @property
    def dataset_name(self) -> str:
        return "rees46"

    @property
    def ecosystem_type(self) -> str:
        return "transactional_marketplace"

    @property
    def churn_window_days(self) -> Optional[int]:
        return None

    @property
    def uses_native_churn_label(self) -> bool:
        return True

    # ── Data loading ─────────────────────────────────────────────────

    def load_raw_data(self) -> pd.DataFrame:
        filepath = (
            REES46_FILE if ON_KAGGLE
            else os.path.join(self.data_dir, LOCAL_REES46_FILE)
        )
        df = pd.read_csv(filepath)
        logger.info("REES46 columns (first 20): %s", list(df.columns[:20]))
        logger.info("REES46 shape: %d rows x %d cols", df.shape[0], df.shape[1])
        return df

    # ── Preprocessing ────────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        id_col = "customer_id" if "customer_id" in df.columns else df.columns[0]
        df = df.dropna(subset=[id_col])
        logger.info("REES46 rows after dropping null identifiers: %d", len(df))
        return df

    # ── Schema standardisation ───────────────────────────────────────

    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        if "customer_id" not in df.columns:
            id_col = [
                c for c in df.columns
                if "customer" in c.lower() or "user" in c.lower()
            ][0]
            df = df.rename(columns={id_col: "customer_id"})

        # No event timestamps exist in the aggregated customer model — a
        # synthetic placeholder satisfies the unified schema.
        df["event_time"] = pd.Timestamp("2020-01-01")
        df["event_type"] = "purchase"
        df["transaction_value"] = 0.0
        df["review_score"] = 0.0
        df["payment_type"] = "unknown"
        df["delivery_delay"] = 0.0
        df["session_id"] = "unknown"
        return df

    # ── Native churn label ───────────────────────────────────────────

    def _churn_column(self, df: pd.DataFrame) -> str:
        candidates = [
            c for c in df.columns
            if "churn" in c.lower() or "target" in c.lower()
        ]
        if not candidates:
            raise ValueError(
                f"No churn column found. Columns: {list(df.columns[:20])}"
            )
        churn_col = candidates[0]
        logger.info("REES46 churn column identified: %s", churn_col)
        return churn_col

    def get_native_churn_labels(
        self, df: pd.DataFrame, cutoff_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        churn_col = self._churn_column(df)
        labels = df[["customer_id", churn_col]].drop_duplicates("customer_id")
        labels["churn"] = (labels[churn_col] > 0).astype(int)
        return labels[["customer_id", "churn"]].copy()

    # ── Native modeling data (no temporal split) ─────────────────────

    def build_native_modeling_data(
        self, df: pd.DataFrame, train_ratio: float = 0.70,
    ) -> tuple:
        """70/30 stratified split on the native label; numeric features only."""
        from sklearn.model_selection import train_test_split

        labels = self.get_native_churn_labels(df).set_index("customer_id")
        model_df = df.drop_duplicates("customer_id").set_index("customer_id")
        model_df = model_df.loc[labels.index]

        churn_col = self._churn_column(df)
        forbid = {"customer_id", churn_col, "event_time", "event_type",
                  "transaction_value", "review_score", "payment_type",
                  "delivery_delay", "session_id"}
        features = [c for c in model_df.columns if c not in forbid]
        X = model_df[features].select_dtypes(include=[np.number, bool])
        X = X.astype(float).fillna(0.0)
        y = labels["churn"]

        if X.shape[1] == 0:
            raise RuntimeError("REES46 has no numeric feature columns after filtering")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=1 - train_ratio,
            random_state=RANDOM_SEED,
            stratify=y,
        )
        return X_train, X_test, y_train.to_frame("churn"), y_test.to_frame("churn")

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
                "REES46 Churn Dataset. "
                "https://www.kaggle.com/datasets/fridrichmrtn/e-commerce-churn-dataset-rees46"
            ),
            "source_url": "https://www.kaggle.com/datasets/fridrichmrtn/e-commerce-churn-dataset-rees46",
            "n_customers_approx": 112_610,
            "n_events_approx": None,
            "churn_window_days": None,
            "churn_justification": (
                "Native label (target_event): no transaction in the future period. "
                "The Kaggle distribution ships a pre-aggregated per-customer model "
                "with no event timestamps, so a 70/30 stratified split is used "
                "instead of a temporal split."
            ),
            "uses_native_churn_label": True,
            "available_feature_groups": self.available_feature_groups,
        }