"""
UCI Credit Card Churn dataset adapter.

Ecosystem type: subscription
Churn definition: native label (Attrition_Flag)

Critical requirement: Delete the last two columns immediately after loading.
These are outputs from a Naive Bayes classifier and constitute direct leakage.

Because timestamps are unavailable, a 70/30 stratified split is used
instead of temporal splitting.

Data source: https://www.kaggle.com/datasets/rikdifos/credit-card-churn-prediction
"""
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.datasets.base import BaseDatasetAdapter
from src.utils import get_logger

logger = get_logger(__name__)

CREDIT_CARD_FILE = "credit_card_customers.csv"
CREDIT_CARD_FILE_KAGGLE = "BankChurners.csv"

NUMERICAL_FEATURES = [
    'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Revolving_Bal',
    'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
    'Avg_Utilization_Ratio', 'Credit_Limit', 'Months_on_book',
    'Total_Relationship_Count', 'Total_Amt_Chng_Q4_Q1',
    'Total_Ct_Chng_Q4_Q1', 'Avg_Open_To_Buy',
]

CATEGORICAL_FEATURES = [
    'Gender', 'Education_Level', 'Marital_Status',
    'Income_Category', 'Card_Category',
]


class CreditCardAdapter(BaseDatasetAdapter):
    @property
    def dataset_name(self) -> str:
        return "credit_card"

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
        # No genuine event timeline — every row is a static customer snapshot
        # (no timestamps).  The pipeline therefore uses a stratified
        # customer-level split instead of temporal cutoffs, and disables
        # snapshot-based feature filtering.
        return False

    @property
    def required_files(self) -> list:
        return [CREDIT_CARD_FILE]

    @property
    def alternate_filenames(self) -> Dict[str, List[str]]:
        """Kaggle ships the data as ``BankChurners.csv`` (which also embeds
        two Naive-Bayes output columns that ``load_raw_data`` strips)."""
        return {
            CREDIT_CARD_FILE: [CREDIT_CARD_FILE_KAGGLE],
        }

    def _resolve_file(self, *candidates: str) -> Optional[str]:
        for name in candidates:
            path = os.path.join(self.data_dir, name)
            if os.path.isfile(path):
                return path
        return None

    def load_raw_data(self) -> pd.DataFrame:
        filepath = self._resolve_file(CREDIT_CARD_FILE, CREDIT_CARD_FILE_KAGGLE)
        if filepath is None:
            raise FileNotFoundError(
                f"Neither {CREDIT_CARD_FILE} nor {CREDIT_CARD_FILE_KAGGLE} "
                f"found in {self.data_dir}"
            )

        df = pd.read_csv(filepath)

        if len(df.columns) >= 2:
            cols_to_drop = df.columns[-2:].tolist()
            logger.info(
                "Dropping leakage columns: %s (last 2 columns)",
                cols_to_drop,
            )
            df = df.drop(columns=cols_to_drop)

        logger.info("Loaded Credit Card: %d rows x %d cols", df.shape[0], df.shape[1])
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('nan', 'Unknown')

        if 'Attrition_Flag' in df.columns:
            df['Attrition_Flag'] = df['Attrition_Flag'].map({
                'Attrited Customer': 1,
                'Existing Customer': 0,
            }).fillna(0).astype(int)

        logger.info("Credit Card preprocessing complete — %d rows", len(df))
        return df

    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        df['event_time'] = pd.Timestamp('2020-01-01')

        if 'CLIENTNUM' in df.columns:
            df['customer_id'] = df['CLIENTNUM'].astype(str)
        else:
            df['customer_id'] = df.index.astype(str)

        if 'Attrition_Flag' in df.columns:
            df['churn'] = df['Attrition_Flag']

        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])

        mapping = {}
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                mapping[col] = col.lower()

        df = df.rename(columns=mapping, errors='ignore')

        if 'transaction_value' not in df.columns:
            if 'monthlycharges' in df.columns:
                df['transaction_value'] = df['monthlycharges']
            else:
                df['transaction_value'] = 0.0

        df['event_type'] = 'subscription_event'

        if 'review_score' not in df.columns:
            df['review_score'] = 0.0

        if 'payment_type' not in df.columns:
            df['payment_type'] = 'unknown'

        df['delivery_delay'] = 0.0

        if 'engagement_signal' not in df.columns:
            if 'months_on_book' in df.columns:
                df['engagement_signal'] = df['months_on_book']
            elif 'total_relationship_count' in df.columns:
                df['engagement_signal'] = df['total_relationship_count']
            else:
                df['engagement_signal'] = 0.0

        logger.info(
            "Standardised schema — columns: %s", list(df.columns)
        )
        return df

    def get_native_churn_labels(
        self, df: pd.DataFrame, cutoff_date: pd.Timestamp,
    ) -> pd.DataFrame:
        if 'customer_id' not in df.columns or 'churn' not in df.columns:
            raise ValueError(
                "Credit Card adapter: customer_id and churn columns required "
                "for native label extraction"
            )
        labels = (
            df[['customer_id', 'churn']]
            .drop_duplicates(subset='customer_id')
            .copy()
        )
        churn_rate = labels['churn'].mean()
        logger.info(
            "Credit Card native churn labels — rate: %.2f%% (%d / %d)",
            churn_rate * 100,
            int(labels['churn'].sum()), len(labels),
        )
        return labels

    @property
    def available_feature_groups(self) -> List[str]:
        return ["purchase", "monetary", "inactivity", "cadence"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": "credit_card",
            "ecosystem_type": "subscription",
            "citation": (
                "Lara, Y. et al., Credit Card Churn Prediction. "
                "UCI Machine Learning Repository. "
                "https://www.kaggle.com/datasets/rikdifos/credit-card-churn-prediction"
            ),
            "source_url": "https://www.kaggle.com/datasets/rikdifos/credit-card-churn-prediction",
            "n_customers_approx": 10_127,
            "n_orders_approx": 10_127,
            "churn_window_days": None,
            "churn_justification": (
                "Contractual churn — uses native 'Attrition_Flag' label. "
                "No inactivity window needed.  Customers are either "
                "active or have attrited."
            ),
            "train_test_split": "70/30 stratified customer-level (no timestamps available)",
            "uses_native_churn_label": True,
            "has_temporal_data": False,
            "available_feature_groups": self.available_feature_groups,
            "critical_requirements": [
                "Delete last two columns (Naive Bayes outputs) before any processing",
            ],
        }
