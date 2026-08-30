"""
UCI credit card customer churn adapter.

Ecosystem type: banking
Churn definition: native Attrition_Flag label.
"""
import os
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from src.config import DATA_DIR, ON_KAGGLE, CREDIT_CARD_FILE, RANDOM_SEED
from src.datasets.base import BaseDatasetAdapter


LOCAL_CREDIT_CARD_FILE = "BankChurners.csv"


class CreditCardAdapter(BaseDatasetAdapter):
    @property
    def dataset_name(self) -> str:
        return "credit_card"

    @property
    def ecosystem_type(self) -> str:
        return "banking"

    @property
    def churn_window_days(self) -> Optional[int]:
        return None

    @property
    def uses_native_churn_label(self) -> bool:
        return True

    def load_raw_data(self) -> pd.DataFrame:
        filepath = (
            CREDIT_CARD_FILE if ON_KAGGLE
            else os.path.join(DATA_DIR, LOCAL_CREDIT_CARD_FILE)
        )
        df = pd.read_csv(filepath)
        if df.shape[1] >= 2:
            df = df.iloc[:, :-2]
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["churn"] = (df["Attrition_Flag"] == "Attrited Customer").astype(int)

        binary_cols = (
            ["Gender", "PaperlessBilling"]
            if "PaperlessBilling" in df.columns else ["Gender"]
        )
        for col in binary_cols:
            if col in df.columns:
                df[col] = pd.Categorical(df[col]).codes

        multi_cols = [
            "Education_Level", "Marital_Status",
            "Income_Category", "Card_Category",
        ]
        df = pd.get_dummies(
            df,
            columns=[c for c in multi_cols if c in df.columns],
            dummy_na=False,
        )
        return df

    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={"CLIENTNUM": "customer_id"})
        df["event_time"] = pd.Timestamp("2019-01-01")
        df["event_type"] = "banking_event"
        df["transaction_value"] = df.get("Total_Trans_Amt", 0.0)
        df["review_score"] = 0.0
        df["payment_type"] = "unknown"
        df["delivery_delay"] = 0.0
        df["session_id"] = "unknown"
        return df

    def get_native_churn_labels(
        self, df: pd.DataFrame, cutoff_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        if "customer_id" not in df.columns or "churn" not in df.columns:
            raise ValueError("Credit card adapter requires customer_id and churn")
        return (
            df[["customer_id", "churn"]]
            .drop_duplicates("customer_id")
            .copy()
        )

    @property
    def available_feature_groups(self) -> List[str]:
        return ["purchase", "monetary", "inactivity", "cadence"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": "credit_card",
            "ecosystem_type": "banking",
            "citation": "",
            "source_url": CREDIT_CARD_FILE if ON_KAGGLE else "",
            "n_customers_approx": None,
            "n_orders_approx": None,
            "churn_window_days": None,
            "uses_native_churn_label": True,
            "available_feature_groups": self.available_feature_groups,
        }

    def build_native_modeling_data(
        self, df: pd.DataFrame, train_ratio: float = 0.70,
    ) -> tuple:
        """Use native churn label with a 70/30 stratified customer split."""
        from sklearn.model_selection import train_test_split

        labels = self.get_native_churn_labels(df).set_index("customer_id")
        model_df = df.drop_duplicates("customer_id").set_index("customer_id")
        model_df = model_df.loc[labels.index]

        requested_features = [
            "Total_Trans_Amt", "Total_Trans_Ct", "Total_Revolving_Bal",
            "Months_Inactive_12_mon", "Contacts_Count_12_mon",
            "Avg_Utilization_Ratio", "Credit_Limit", "Months_on_book",
            "Total_Relationship_Count", "Total_Amt_Chng_Q4_Q1",
            "Total_Ct_Chng_Q4_Q1", "Avg_Open_To_Buy",
        ]
        encoded_features = [
            c for c in model_df.columns
            if c.startswith(("Education_Level_", "Marital_Status_",
                             "Income_Category_", "Card_Category_"))
            or c in {"Gender", "PaperlessBilling"}
        ]
        columns = [c for c in requested_features + encoded_features
                   if c in model_df.columns]
        X = model_df[columns].select_dtypes(include=[np.number, bool])
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
