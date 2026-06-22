"""
Last.fm 1K Users dataset adapter.

Ecosystem type: media_streaming
Churn definition: 30-day inactivity

Models music listening behavior — engagement and listening diversity.
SVM uses a stratified subset of at most 5,000 users.

Listening events are mapped to "purchase" events so the generic feature
engineering module can compute cadence and recency features.  The
standardized schema provides artist as product_id, enabling unique artist
counting and diversity computation through the existing pipeline.

Data sources:
- lastfm-dataset-1k.snappy.parquet (primary listening data)
- userid-profile.tsv (user profiles)
"""
import os
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

from src.datasets.base import BaseDatasetAdapter
from src.utils import get_logger

logger = get_logger(__name__)

LISTENING_FILE = "lastfm-dataset-1k.snappy.parquet"
PROFILE_FILE = "userid-profile.tsv"

SVM_MAX_USERS = 5000


class LastFMAdapter(BaseDatasetAdapter):
    @property
    def dataset_name(self) -> str:
        return "lastfm"

    @property
    def ecosystem_type(self) -> str:
        return "media_streaming"

    @property
    def churn_window_days(self) -> Optional[int]:
        return 30

    @property
    def required_files(self) -> list:
        return [LISTENING_FILE]

    def load_raw_data(self) -> pd.DataFrame:
        listening_path = os.path.join(self.data_dir, LISTENING_FILE)
        if not os.path.isfile(listening_path):
            raise FileNotFoundError(
                f"Required file {LISTENING_FILE} not found in {self.data_dir}"
            )

        listening = pd.read_parquet(listening_path)
        logger.info(
            "Loaded listening data: %d rows x %d cols",
            listening.shape[0], listening.shape[1],
        )

        profile_path = os.path.join(self.data_dir, PROFILE_FILE)
        if os.path.isfile(profile_path):
            profile = pd.read_csv(
                profile_path,
                sep='\t',
                header=None,
                names=['user_id', 'gender', 'age', 'country', 'signup'],
            )
            listening = listening.merge(
                profile,
                on='user_id',
                how='left',
            )
            logger.info(
                "Merged with profiles: %d rows",
                listening.shape[0],
            )

        return listening

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            before = len(df)
            df = df.dropna(subset=['timestamp'])
            dropped = before - len(df)
            if dropped:
                logger.info("Dropped %d rows with null timestamp", dropped)

        if 'user_id' in df.columns:
            df = df.dropna(subset=['user_id'])

        if 'artist' in df.columns:
            df['artist'] = df['artist'].astype(str).str.strip()

        if 'track' in df.columns:
            df['track'] = df['track'].astype(str).str.strip()

        return df

    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        mapping = {
            'user_id': 'customer_id',
            'timestamp': 'event_time',
            'artist': 'product_id',
        }
        df = df.rename(columns=mapping, errors='ignore')

        df['event_type'] = 'purchase'

        df['transaction_value'] = 0.0

        if 'review_score' not in df.columns:
            df['review_score'] = 0.0

        if 'payment_type' not in df.columns:
            df['payment_type'] = 'unknown'

        df['delivery_delay'] = 0.0

        if 'session_id' not in df.columns:
            df['session_id'] = 'unknown'

        if 'track' in df.columns:
            df['engagement_signal'] = df['track'].apply(
                lambda x: 1 if pd.notna(x) else 0
            )
        else:
            df['engagement_signal'] = 1.0

        logger.info(
            "Standardised schema — columns: %s", list(df.columns)
        )
        return df

    @property
    def available_feature_groups(self) -> List[str]:
        """Listening behaviour is the core signal (Section 17); purchase
        cadence and inactivity complement it."""
        return ["listening", "purchase", "inactivity", "cadence"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": "lastfm",
            "ecosystem_type": "media_streaming",
            "citation": (
                "Last.fm 1K Users Dataset. "
                "https://www.kaggle.com/datasets/rawanalashraf/lastfm-dataset"
            ),
            "source_url": "https://www.kaggle.com/datasets/rawanalashraf/lastfm-dataset",
            "n_customers_approx": 1_000,
            "n_events_approx": 17_000_000,
            "churn_window_days": 30,
            "churn_justification": (
                "30 days — music streaming has high daily engagement; "
                "30 days of inactivity indicates genuine disengagement."
            ),
            "uses_native_churn_label": False,
            "available_feature_groups": self.available_feature_groups,
            "svm_max_users": SVM_MAX_USERS,
            "feature_mapping": (
                "Listening events mapped to 'purchase' events for generic "
                "feature engineering. Artists mapped to 'product_id' for "
                "unique artist counting and diversity computation."
            ),
        }
