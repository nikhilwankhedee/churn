"""
IBM Telco Customer Churn dataset adapter.

Ecosystem type: subscription
Churn definition: native label (Churn = Yes/No)

This dataset represents a contractual subscription ecosystem.
Churn is explicitly provided — do NOT use inactivity labeling.

Users have tenure, contract type (month-to-month, one year, two year),
monthly charges, and demographic attributes.

Because timestamps are unavailable, a 70/30 stratified split is used
instead of temporal splitting.

Data source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.datasets.base import BaseDatasetAdapter
from src.utils import get_logger

logger = get_logger(__name__)

TELCO_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TELCO_FILE_ALT = "telco_customer_churn.csv"

NUMERICAL_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
]

STATIC_FEATURE_COLUMNS = [
    'static_tenure', 'static_monthly_charges', 'static_total_charges',
    'static_senior_citizen',
]

CATEGORICAL_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
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

    @property
    def has_temporal_data(self) -> bool:
        return False

    @property
    def required_files(self) -> list:
        return [TELCO_FILE]

    @property
    def alternate_filenames(self) -> dict:
        return {TELCO_FILE: [TELCO_FILE_ALT]}

    def _resolve_file(self) -> str:
        for name in [TELCO_FILE, TELCO_FILE_ALT]:
            path = os.path.join(self.data_dir, name)
            if os.path.isfile(path):
                return path
        return os.path.join(self.data_dir, TELCO_FILE)

    def load_raw_data(self) -> pd.DataFrame:
        filepath = self._resolve_file()
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"Required file ({TELCO_FILE} or {TELCO_FILE_ALT}) "
                f"not found in {self.data_dir}"
            )

        df = pd.read_csv(filepath)
        logger.info("Loaded Telco: %d rows x %d cols", df.shape[0], df.shape[1])
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "customerID" in df.columns:
            df = df.dropna(subset=["customerID"])
            df["customerID"] = df["customerID"].astype(str).str.strip()
            df = df[df["customerID"] != ""].copy()

        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"], errors="coerce"
            ).fillna(0)

        if "Churn" in df.columns:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = pd.to_numeric(
                df["SeniorCitizen"], errors="coerce"
            ).fillna(0).astype(int)

        logger.info("Telco preprocessing complete — %d rows", len(df))
        return df

    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        if "tenure" in df.columns:
            tenure_months = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
            df["event_time"] = pd.Timestamp("2019-01-31") - pd.to_timedelta(
                tenure_months * 30, unit="D"
            )
        else:
            df["event_time"] = pd.Timestamp("2019-01-31")

        mapping = {
            "customerID": "customer_id",
            "MonthlyCharges": "transaction_value",
            "tenure": "engagement_signal",
            "TotalCharges": "total_charges",
        }
        df = df.rename(columns=mapping, errors="ignore")

        df["event_type"] = "subscription_event"

        if "review_score" not in df.columns:
            df["review_score"] = 0.0

        if "payment_type" not in df.columns:
            df["payment_type"] = "unknown"

        df["delivery_delay"] = 0.0

        # Static (time-invariant) customer attributes exposed with the
        # 'static_' prefix for the static feature group (Section 15).
        static_vals = {
            "static_tenure": pd.to_numeric(df.get("tenure", pd.Series(0)),
                                            errors="coerce").fillna(0),
            "static_monthly_charges": pd.to_numeric(
                df.get("MonthlyCharges", pd.Series(0)),
                errors="coerce").fillna(0),
            "static_total_charges": pd.to_numeric(
                df.get("TotalCharges", pd.Series(0)),
                errors="coerce").fillna(0),
            "static_senior_citizen": pd.to_numeric(
                df.get("SeniorCitizen", pd.Series(0)),
                errors="coerce").fillna(0),
        }
        for col, series in static_vals.items():
            df[col] = series

        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])

        logger.info(
            "Standardised schema — columns: %s", list(df.columns)
        )
        return df

    def get_native_churn_labels(
        self, df: pd.DataFrame, cutoff_date: pd.Timestamp,
    ) -> pd.DataFrame:
        if "customer_id" not in df.columns or "Churn" not in df.columns:
            raise ValueError(
                "Telco adapter: customer_id and Churn columns required "
                "for native label extraction"
            )

        # Telco has no genuine event timeline, so the cutoff is ignored
        # here: the framework performs the customer-level stratified 70/30
        # split (stratified_native_split, seed 42) instead of a temporal
        # split (Section 15 of the experiment spec).  This keeps a single
        # split mechanism for every non-temporal native-label dataset.
        labels = (
            df[["customer_id", "Churn"]]
            .drop_duplicates(subset="customer_id")
            .copy()
            .rename(columns={"Churn": "churn"})
        )
        churn_rate = labels["churn"].mean()
        logger.info(
            "Telco native churn labels (all customers) — rate: %.2f%% (%d / %d)",
            churn_rate * 100,
            int(labels["churn"].sum()), len(labels),
        )
        return labels

    @property
    def available_feature_groups(self) -> List[str]:
        return ["static", "monetary", "cadence"]

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
            "train_test_split": "70/30 stratified (no timestamps available)",
            "uses_native_churn_label": True,
            "available_feature_groups": self.available_feature_groups,
        }
