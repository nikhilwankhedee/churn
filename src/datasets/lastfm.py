"""
Last.fm music streaming dataset adapter.

Ecosystem type: music_streaming
Churn definition: user-relative listening-frequency degradation.
"""
import os
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR, ON_KAGGLE, LASTFM_PARQUET, LASTFM_PROFILE, RANDOM_SEED,
)
from src.datasets.base import BaseDatasetAdapter


LOCAL_LASTFM_PARQUET = "lastfm-dataset-1k.snappy.parquet"
LOCAL_LASTFM_PROFILE = "userid-profile.tsv"


class LastFMAdapter(BaseDatasetAdapter):
    @property
    def dataset_name(self) -> str:
        return "lastfm"

    @property
    def ecosystem_type(self) -> str:
        return "music_streaming"

    @property
    def churn_window_days(self) -> Optional[int]:
        return 90

    @property
    def uses_user_relative_churn_label(self) -> bool:
        return True

    def load_raw_data(self):
        parquet_path = (
            LASTFM_PARQUET if ON_KAGGLE
            else os.path.join(DATA_DIR, LOCAL_LASTFM_PARQUET)
        )
        profile_path = (
            LASTFM_PROFILE if ON_KAGGLE
            else os.path.join(DATA_DIR, LOCAL_LASTFM_PROFILE)
        )
        events = pd.read_parquet(parquet_path)
        profile = pd.read_csv(
            profile_path,
            sep="\t",
            header=None,
            names=["user_id", "gender", "age", "country", "signup"],
        )
        return events, profile

    def preprocess(self, data) -> pd.DataFrame:
        events, profile = data
        df = events.merge(profile, on="user_id", how="left")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["user_id", "timestamp"]).copy()
        return df

    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={
            "user_id": "customer_id",
            "timestamp": "event_time",
            "artist_name": "product_id",
        })
        df["event_type"] = "listen"
        df["transaction_value"] = 0.0
        df["review_score"] = 0.0
        df["payment_type"] = "unknown"
        df["delivery_delay"] = 0.0
        df["session_id"] = "unknown"
        return df

    @property
    def available_feature_groups(self) -> List[str]:
        return ["inactivity", "engagement", "cadence"]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": "lastfm",
            "ecosystem_type": "music_streaming",
            "citation": "",
            "source_url": LASTFM_PARQUET if ON_KAGGLE else "",
            "n_customers_approx": None,
            "n_events_approx": None,
            "churn_window_days": 90,
            "uses_native_churn_label": False,
            "available_feature_groups": self.available_feature_groups,
        }

    def build_user_relative_modeling_data(
        self, df: pd.DataFrame, train_ratio: float = 0.70,
    ) -> tuple:
        """Build Last.fm features and labels from personal listening decay."""
        from sklearn.model_selection import train_test_split

        labels = []
        features = []
        for user_id, user_df in df.groupby("customer_id"):
            user_df = user_df.sort_values("event_time")
            if len(user_df) < 50:
                continue

            split_idx = max(1, int(len(user_df) * train_ratio))
            obs = user_df.iloc[:split_idx]
            fut = user_df.iloc[split_idx:]
            if obs.empty or fut.empty:
                continue

            obs_days = max(
                1, (obs["event_time"].max() - obs["event_time"].min()).days + 1,
            )
            fut_days = max(
                1, (fut["event_time"].max() - fut["event_time"].min()).days + 1,
            )
            obs_mean = len(obs) / obs_days
            fut_mean = len(fut) / fut_days
            churn = int(fut_mean < 0.5 * obs_mean)

            active_days = obs["event_time"].dt.date.nunique()
            total_listens = len(obs)
            unique_artists = obs["product_id"].nunique() if "product_id" in obs else 0
            track_col = "track_name" if "track_name" in obs.columns else None
            unique_tracks = obs[track_col].nunique() if track_col else 0
            listens_by_day = obs.groupby(obs["event_time"].dt.date).size()
            gaps = (
                obs["event_time"].sort_values().diff().dt.total_seconds()
                .dropna() / 86400.0
            )
            max_gap = float(gaps.max()) if not gaps.empty else 0.0

            features.append({
                "customer_id": user_id,
                "total_listens": total_listens,
                "unique_artists": unique_artists,
                "unique_tracks": unique_tracks,
                "active_days": active_days,
                "avg_listens_per_day": total_listens / max(active_days, 1),
                "days_since_last_listen": (
                    obs["event_time"].max() - obs["event_time"].iloc[-1]
                ).days,
                "max_gap_between_sessions": max_gap,
                "listening_frequency": float(listens_by_day.mean()),
                "artist_diversity_ratio": unique_artists / max(total_listens, 1),
            })
            labels.append({"customer_id": user_id, "churn": churn})

        if not features:
            raise RuntimeError("No eligible Last.fm users after minimum-listen filtering")

        X = pd.DataFrame(features).set_index("customer_id").fillna(0.0)
        y = pd.DataFrame(labels).set_index("customer_id").loc[X.index, "churn"]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=RANDOM_SEED,
            stratify=y,
        )
        return X_train, X_test, y_train.to_frame("churn"), y_test.to_frame("churn")
