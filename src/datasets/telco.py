"""
IBM Telco Customer Churn dataset adapter.

Ecosystem type: subscription
Churn definition: native label (Churn = Yes/No)

This dataset represents a contractual subscription ecosystem.
Churn is explicitly provided — do NOT use inactivity labeling.

Users have tenure, contract type (month-to-month, one year, two year),
monthly charges, and demographic attributes.

Data source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""
import os
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

from src.datasets.base import BaseDatasetAdapter
from src.config import ON_KAGGLE, TELCO_FILE as KAGGLE_TELCO_FILE, RANDOM_SEED
from src.utils import get_logger

logger = get_logger(__name__)

TELCO_FILE = "telco_customer_churn.csv"
TELCO_BINARY_COLS = [
    "PhoneService", "PaperlessBilling", "gender", "Partner", "Dependents",
]
TELCO_MULTI_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]


class TelcoAdapter(BaseDatasetAdapter):
    @property
    def dataset_name(self) -> str:
        return "telco"

    @property
    def ecosystem_type(self) -> str:
        return "subscription"

    @property
    def churn_window_days(self) -> Optional[int]:
        return None

    @property
    def uses_native_churn_label(self) -> bool:
        return True

    # ── Data loading ─────────────────────────────────────────────────

    def load_raw_data(self) -> pd.DataFrame:
        filepath = (
            KAGGLE_TELCO_FILE if ON_KAGGLE
            else os.path.join(self.data_dir, TELCO_FILE)
        )
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"Required file {TELCO_FILE} not found in {self.data_dir}"
            )

        df = pd.read_csv(filepath)
        logger.info("Loaded Telco: %d rows x %d cols", df.shape[0], df.shape[1])
        return df

    # ── Preprocessing ────────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "customerID" in df.columns:
            df = df.dropna(subset=["customerID"])
            df["customerID"] = df["customerID"].astype(str).str.strip()
            df = df[df["customerID"] != ""].copy()

        # TotalCharges may be object type with whitespace
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"], errors="coerce"
            ).fillna(0)

        # Convert Churn to binary
        if "Churn" in df.columns:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

        # Convert SeniorCitizen (0/1 object → int)
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = pd.to_numeric(
                df["SeniorCitizen"], errors="coerce"
            ).fillna(0).astype(int)

        for col in TELCO_BINARY_COLS:
            if col in df.columns:
                df[f"{col}_encoded"] = pd.Categorical(df[col]).codes

        multi_cols = [c for c in TELCO_MULTI_COLS if c in df.columns]
        if multi_cols:
            df = pd.get_dummies(df, columns=multi_cols, prefix=multi_cols)

        logger.info("Telco preprocessing complete — %d rows", len(df))
        return df

    # ── Schema standardisation ───────────────────────────────────────

    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        mapping = {
            "customerID": "customer_id",
            "MonthlyCharges": "transaction_value",
            "tenure": "engagement_signal",
            "TotalCharges": "total_charges",
        }
        df = df.rename(columns=mapping, errors="ignore")

        # Synthetic event_time based on tenure
        if "engagement_signal" in df.columns:
            tenure_months = pd.to_numeric(
                df["engagement_signal"], errors="coerce",
            ).fillna(0)
            df["event_time"] = pd.Timestamp("2019-01-31") - pd.to_timedelta(
                tenure_months * 30, unit="D"
            )
        else:
            df["event_time"] = pd.Timestamp("2019-01-31")

        df["event_type"] = "subscription_event"

        if "review_score" not in df.columns:
            df["review_score"] = 0.0

        if "payment_type" not in df.columns:
            df["payment_type"] = "unknown"

        df["delivery_delay"] = 0.0

        logger.info(
            "Standardised schema — columns: %s", list(df.columns)
        )
        return df

    # ── Native churn labels ──────────────────────────────────────────

    def get_native_churn_labels(
        self, df: pd.DataFrame, cutoff_date: pd.Timestamp,
    ) -> pd.DataFrame:
        if "customer_id" not in df.columns or "Churn" not in df.columns:
            raise ValueError(
                "Telco adapter: customer_id and Churn columns required "
                "for native label extraction"
            )
        labels = (
            df[["customer_id", "Churn"]]
            .drop_duplicates(subset="customer_id")
            .copy()
        )
        labels = labels.rename(columns={"Churn": "churn"})
        churn_rate = labels["churn"].mean()
        logger.info(
            "Telco native churn labels — rate: %.2f%% (%d / %d)",
            churn_rate * 100,
            int(labels["churn"].sum()), len(labels),
        )
        return labels

    def build_native_modeling_data(
        self, df: pd.DataFrame, train_ratio: float = 0.70,
    ) -> tuple:
        """Use encoded Telco row-level predictors in a stratified 70/30 split."""
        from sklearn.model_selection import train_test_split

        labels = self.get_native_churn_labels(df, pd.Timestamp("2019-01-31"))
        labels = labels.set_index("customer_id")
        model_df = df.drop_duplicates("customer_id").set_index("customer_id")
        model_df = model_df.loc[labels.index]

        excluded = {
            "Churn", "churn", "event_time", "event_type", "payment_type",
            "customer_id",
        }
        X = model_df.drop(columns=[c for c in excluded if c in model_df.columns])
        X = X.select_dtypes(include=[np.number, bool]).copy()
        X = X.astype(float).fillna(0.0)
        y = labels["churn"]

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
        return ["purchase", "monetary", "inactivity", "cadence"]

    # ── Metadata ─────────────────────────────────────────────────────

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": "telco",
            "ecosystem_type": "subscription",
            "citation": (
                "IBM Telco Customer Churn Dataset. "
                "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
            ),
            "source_url": "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
            "n_customers_approx": 7_043,
            "n_orders_approx": 7_043,
            "churn_window_days": None,
            "churn_justification": (
                "Contractual churn — uses native 'Churn' label. "
                "No inactivity window needed.  Customers are either "
                "still subscribed or have explicitly churned."
            ),
            "uses_native_churn_label": True,
            "available_feature_groups": self.available_feature_groups,
        }
