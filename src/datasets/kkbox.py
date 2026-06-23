"""
KKBox (WSDM Cup 2018) dataset adapter.

Ecosystem type: subscription (media streaming)
Churn definition: WSDM membership-expiration churn — a member churns when they
renew more than 30 days after their effective membership expiration date (or
not at all).  Labels are re-derived from the raw transaction history by
WSDMChurnLabeller (faithful re-implementation of the official
WSDMChurnLabeller.scala) and validated against the official train.csv /
train_v2.csv ground truth when present.

Four raw tables (any subset may be present; transactions are required):
    transactions.csv / transactions_v2.csv  — subscription purchases
    user_logs.csv / user_logs_v2.csv        — daily listening summaries
    members.csv / members_v3.csv            — static member attributes
    train.csv / train_v2.csv                — official churn labels (for validation)

The adapter emits three kinds of standardised events:
    event_type='purchase'  — one row per transaction with kkbox_* pricing cols
    event_type='listen'    — one row per member-day with kkbox_num_* columns
    event_type='member'    — one row per member with static_* attributes

Data source: https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge
"""
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as _tts

from src.config import (
    KKBOX_LOGS_FILE,
    KKBOX_LOGS_V2_FILE,
    KKBOX_MEMBERS_FILE,
    KKBOX_MEMBERS_V3_FILE,
    KKBOX_RAW_STRING_COLUMNS,
    KKBOX_RENEWAL_WINDOW_DAYS,
    KKBOX_TRANSACTIONS_FILE,
    KKBOX_TRANSACTIONS_V2_FILE,
    KKBOX_VALID_BD_RANGE,
    RANDOM_SEED,
)
from src.datasets.base import BaseDatasetAdapter
from src.kkbox.labeler import WSDMChurnLabeller
from src.utils import get_logger

logger = get_logger(__name__)

MAX_TRANSACTION_ROWS = 2_000_000
MAX_LOG_ROWS = 3_000_000


class KKBoxAdapter(BaseDatasetAdapter):
    @property
    def dataset_name(self) -> str:
        return "kkbox"

    @property
    def ecosystem_type(self) -> str:
        return "subscription"

    @property
    def churn_window_days(self) -> Optional[int]:
        return None  # label comes from WSDM labeller (uses KKBOX_CHURN_WINDOW_DAYS)

    @property
    def uses_native_churn_label(self) -> bool:
        return True

    @property
    def has_temporal_data(self) -> bool:
        return True

    @property
    def required_files(self) -> list:
        return [KKBOX_TRANSACTIONS_FILE, KKBOX_TRANSACTIONS_V2_FILE]

    # ── Data loading ─────────────────────────────────────────────────

    def _first_existing(self, *names: str) -> Optional[str]:
        for name in names:
            path = os.path.join(self.data_dir, name)
            if os.path.isfile(path):
                return path
        return None

    def _safe_read_csv(self, filepath: Optional[str], table_name: str,
                       **kwargs) -> Optional[pd.DataFrame]:
        if filepath is None or not os.path.isfile(filepath):
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
        tx_path = self._first_existing(
            KKBOX_TRANSACTIONS_FILE, KKBOX_TRANSACTIONS_V2_FILE,
        )
        if tx_path is None:
            raise FileNotFoundError(
                f"Required KKBox transactions file "
                f"({KKBOX_TRANSACTIONS_FILE}/{KKBOX_TRANSACTIONS_V2_FILE}) "
                f"not found in {self.data_dir}"
            )

        transactions = self._safe_read_csv(
            tx_path, "transactions",
            dtype={"msno": str},
        )
        if transactions is None:
            raise FileNotFoundError(
                f"Failed to load KKBox transactions from {tx_path}"
            )

        if len(transactions) > MAX_TRANSACTION_ROWS:
            transactions = transactions.sample(
                MAX_TRANSACTION_ROWS, random_state=42,
            )
            logger.warning(
                "KKBox transactions sampled to %d rows for memory safety",
                len(transactions),
            )

        logs_path = self._first_existing(
            KKBOX_LOGS_FILE, KKBOX_LOGS_V2_FILE,
        )
        logs = self._safe_read_csv(
            logs_path, "user_logs", dtype={"msno": str},
        ) if logs_path else None
        if logs is not None and len(logs) > MAX_LOG_ROWS:
            logs = logs.sample(MAX_LOG_ROWS, random_state=42)
            logger.warning("KKBox user_logs sampled to %d rows", len(logs))

        members_path = self._first_existing(
            KKBOX_MEMBERS_FILE, KKBOX_MEMBERS_V3_FILE,
        )
        members = self._safe_read_csv(
            members_path, "members", dtype={"msno": str},
        ) if members_path else None

        frames = [transactions]
        if logs is not None:
            frames.append(logs)
        if members is not None:
            frames.append(members)

        combined = pd.concat(frames, ignore_index=True, sort=False)
        logger.info("Final merged KKBox dataset: %d rows x %d cols",
                    combined.shape[0], combined.shape[1])
        return combined

    # ── Preprocessing ────────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "msno" in df.columns:
            df = df.dropna(subset=["msno"])
            df["msno"] = df["msno"].astype(str).str.strip()
            df = df[df["msno"] != ""].copy()

        for date_col in ["transaction_date", "date"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(
                    df[date_col], errors="coerce", format="%Y%m%d",
                )

        # members: bd outside [0, 99] treated as missing (WSDM spec)
        if "bd" in df.columns:
            lo, hi = KKBOX_VALID_BD_RANGE
            df["bd"] = pd.to_numeric(df["bd"], errors="coerce")
            df.loc[~df["bd"].between(lo, hi), "bd"] = np.nan

        for num_col in ["payment_plan_days", "plan_list_price",
                        "actual_amount_paid"]:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0)

        for num_col in ["num_25", "num_50", "num_75", "num_985",
                        "num_100", "num_unq", "total_secs"]:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0)

        logger.info("KKBox preprocessing complete — %d rows", len(df))
        return df

    # ── Schema standardisation ───────────────────────────────────────

    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        tx_mask = df["transaction_date"].notna() if "transaction_date" in df.columns else pd.Series(False, index=df.index)
        log_mask = df["date"].notna() if "date" in df.columns else pd.Series(False, index=df.index)

        # ── Transaction events ────────────────────────────────────────
        tx = df[tx_mask].copy()
        if not tx.empty:
            tx = tx.rename(columns={"msno": "customer_id",
                                    "transaction_date": "event_time"})
            tx["event_type"] = "purchase"
            tx["transaction_value"] = tx.get("actual_amount_paid", 0.0).fillna(0)
            tx["kkbox_actual_amount_paid"] = tx.get("actual_amount_paid", 0.0).fillna(0)
            for src, dst in [
                ("is_auto_renew", "kkbox_is_auto_renew"),
                ("plan_list_price", "kkbox_plan_list_price"),
                ("plan_id", "kkbox_plan_id"),
                ("payment_method_id", "kkbox_payment_method_id"),
                ("payment_plan_days", "kkbox_payment_plan_days"),
                # Raw WSDM fields preserved for faithful label reproduction:
                ("membership_expire_date", "kkbox_membership_expire_date"),
                ("is_cancel", "kkbox_is_cancel"),
            ]:
                if src in tx.columns:
                    tx[dst] = tx[src]
            out = pd.concat([out, tx], ignore_index=True, sort=False)

        # ── Listening log events ──────────────────────────────────────
        logs = df[log_mask].copy()
        if not logs.empty:
            logs = logs.rename(columns={"msno": "customer_id", "date": "event_time"})
            logs["event_type"] = "listen"
            logs["transaction_value"] = 0.0
            for src in ["num_25", "num_50", "num_75", "num_985",
                        "num_100", "num_unq", "total_secs"]:
                if src in logs.columns:
                    logs[f"kkbox_{src}"] = logs[src]
            out = pd.concat([out, logs], ignore_index=True, sort=False)

        # ── Member static attributes ──────────────────────────────────
        members = df[~tx_mask & ~log_mask].copy() if not out.empty else df[~tx_mask & ~log_mask].copy()
        if members.empty and "bd" in df.columns:
            members = df[~tx_mask & ~log_mask].copy()
        if not members.empty and "msno" in members.columns:
            members = members.rename(columns={"msno": "customer_id"})
            members["event_type"] = "member"
            members["event_time"] = pd.to_datetime(
                members.get("registration_init_time", "2015-01-01"),
                errors="coerce", format="%Y%m%d",
            ).fillna(pd.Timestamp("2015-01-01"))
            members["transaction_value"] = 0.0
            for src, dst in [
                ("city", "static_city"),
                ("bd", "static_bd"),
                ("gender", "static_gender"),
                ("registered_via", "static_registered_via"),
                ("registration_init_time", "static_registration_init_time"),
            ]:
                if src in members.columns:
                    members[dst] = members[src]
            out = pd.concat([out, members], ignore_index=True, sort=False)

        for col in ["customer_id", "event_time", "event_type"]:
            if col not in out.columns:
                raise ValueError(f"KKBox standardize_schema: missing '{col}'")
        out["event_time"] = pd.to_datetime(out["event_time"], errors="coerce")
        out = out.dropna(subset=["event_time"]).copy()

        out["review_score"] = 0.0
        out["payment_type"] = out.get("payment_method_id", "unknown").fillna("unknown")
        out["delivery_delay"] = 0.0
        if "product_id" not in out.columns:
            out["product_id"] = "kkbox_subscription"

        logger.info("Standardised schema — %d rows, event types: %s",
                    len(out), out["event_type"].unique())
        return out

    # ── Native churn labels (WSDM-derived) ───────────────────────────

    def _raw_transactions(self) -> pd.DataFrame:
        """Raw transaction table read as strings (cached per adapter).

        The WSDM plan signature concatenates CSV strings, so the labeler
        must see the raw string representation of the transaction columns.
        """
        if getattr(self, "_raw_tx_cache", None) is None:
            path = self._first_existing(
                KKBOX_TRANSACTIONS_FILE, KKBOX_TRANSACTIONS_V2_FILE,
            )
            if path is None:
                raise FileNotFoundError(
                    f"Required KKBox transactions file not found in "
                    f"{self.data_dir}"
                )
            dtypes = {col: str for col in KKBOX_RAW_STRING_COLUMNS}
            dtypes["msno"] = str
            raw = pd.read_csv(path, dtype=dtypes)
            self._raw_tx_cache = raw
            logger.info("Loaded raw KKBox transactions for labelling: %d rows",
                        len(raw))
        return self._raw_tx_cache

    def _member_split(self, labels: pd.DataFrame):
        """Deterministic 70/30 member split (stratified by churn, seed 42)."""
        ids = labels["customer_id"].astype(str).values
        y = labels["churn"].astype(int).values
        train_ids, test_ids = _tts(
            ids, test_size=0.3, random_state=RANDOM_SEED, stratify=y,
        )
        return set(train_ids), set(test_ids)

    def get_native_churn_labels(
        self, df: pd.DataFrame, cutoff_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Compute WSDM churn labels from the raw transaction history.

        Labels are derived deterministically with the official WSDM scheme
        (history 2017-01, candidates expiring 2017-02, renewal window 30
        days).  They are never taken from the official train.csv — those
        files are used only for validation (see src/kkbox/validation.py).

        The framework calls this method once with a training cutoff and once
        with the test cutoff.  Because the WSDM label set is a single fixed
        prediction period, the labeled members are split deterministically
        70/30 (stratified on churn, seed 42) and the requested subset is
        returned — mirroring the Telco adapter's native-label contract.
        """
        raw = self._raw_transactions()
        labeller = WSDMChurnLabeller(churn_window_days=KKBOX_RENEWAL_WINDOW_DAYS)
        labels = labeller.compute_churn_labels(raw)

        if getattr(self, "_kkbox_split", None) is None:
            self._kkbox_split = self._member_split(labels)
        train_ids, test_ids = self._kkbox_split

        is_test_call = cutoff_date >= df["event_time"].max()
        keep = test_ids if is_test_call else train_ids
        out = labels[labels["customer_id"].isin(keep)].copy()
        split_name = "test" if is_test_call else "train"
        logger.info(
            "KKBox native churn labels (%s split) — %d members, churn rate "
            "%.2f%%",
            split_name, len(out),
            (out["churn"].mean() * 100) if len(out) else 0.0,
        )
        return out

    # ── Feature groups ───────────────────────────────────────────────

    @property
    def available_feature_groups(self) -> List[str]:
        return ["kkbox", "static", "inactivity", "cadence"]

    # ── Metadata ─────────────────────────────────────────────────────

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": "kkbox",
            "ecosystem_type": "subscription",
            "citation": (
                "WSDM Cup 2018 Challenge — Churn Prediction. "
                "https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge"
            ),
            "source_url": "https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge",
            "n_customers_approx": 1_000_000,
            "n_transactions_approx": 24_000_000,
            "churn_window_days": None,
            "churn_justification": (
                "WSDM membership-expiration churn: a member is churned when "
                "they do not renew within 30 days of their effective "
                "membership expiration.  Labels are re-derived from raw "
                "transactions by WSDMChurnLabeller (faithful re-implementation "
                "of WSDMChurnLabeller.scala) and validated against the "
                "official train.csv/train_v2.csv when present."
            ),
            "uses_native_churn_label": True,
            "available_feature_groups": self.available_feature_groups,
            "labeler": "WSDMChurnLabeller (membership-expiration / renewal-gap)",
            "validation": "validated against train.csv/train_v2.csv when present",
        }
