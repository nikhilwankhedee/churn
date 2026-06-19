"""
Online Retail II dataset adapter (UK gift retailer).

Ecosystem type: habitual_retail
Churn window: 90 days of inactivity

Contains transactional data from a UK-based online gift retailer.
Includes wholesale buyers and cancellations (invoice numbers starting
with 'C').  Habitual purchasing behaviour is observable through
repeat transactions over a multi-year period.

Data source: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
"""
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.datasets.base import BaseDatasetAdapter
from src.utils import get_logger

logger = get_logger(__name__)

FILE_2009_2010 = "online_retail_II_2009_2010.csv"
FILE_2010_2011 = "online_retail_II_2010_2011.csv"
FILE_XLSX = "online_retail_II.xlsx"
FILE_XLSX_KAGGLE = "online_retail_II.xlsx"

CANCELLATION_PREFIX = "C"


class OnlineRetailIIAdapter(BaseDatasetAdapter):
    @property
    def dataset_name(self) -> str:
        return "online_retail_ii"

    @property
    def ecosystem_type(self) -> str:
        return "habitual_retail"

    @property
    def churn_window_days(self) -> Optional[int]:
        return 90

    @property
    def required_files(self) -> list:
        return [FILE_2009_2010, FILE_2010_2011]

    @property
    def alternate_filenames(self) -> Dict[str, List[str]]:
        """Kaggle ships a single ``online_retail_II.xlsx`` (two sheets) instead
        of the two CSV files.  The resolver treats a required file as present
        if the xlsx alternate exists."""
        return {
            FILE_2009_2010: [FILE_XLSX_KAGGLE],
            FILE_2010_2011: [FILE_XLSX_KAGGLE],
        }

    # ── Data loading ─────────────────────────────────────────────────

    def _resolve_file(self, *candidates: str) -> Optional[str]:
        """Return path of first existing file, or None."""
        for name in candidates:
            path = os.path.join(self.data_dir, name)
            if os.path.isfile(path):
                return path
        return None

    def _safe_read_csv(self, filepath: str, table_name: str,
                       **kwargs) -> Optional[pd.DataFrame]:
        if not os.path.isfile(filepath):
            logger.warning("File not found: %s — skipping %s", filepath, table_name)
            return None
        try:
            df = pd.read_csv(filepath, encoding="ISO-8859-1", **kwargs)
            logger.info("Loaded %s: %d rows x %d cols",
                         table_name, df.shape[0], df.shape[1])
            return df
        except Exception as exc:
            logger.error("Failed to load %s: %s", filepath, exc)
            return None

    def _load_from_xlsx(self) -> Optional[pd.DataFrame]:
        """Load from a single .xlsx file with two sheets (Kaggle format)."""
        xlsx_path = self._resolve_file(FILE_XLSX, FILE_XLSX_KAGGLE)
        if xlsx_path is None:
            return None

        try:
            xls = pd.ExcelFile(xlsx_path)
            sheet_names = xls.sheet_names
            logger.info("XLSX sheets: %s", sheet_names)

            frames = []
            for sheet in sheet_names:
                df = pd.read_excel(xlsx_path, sheet_name=sheet)
                logger.info("Sheet '%s': %d rows x %d cols", sheet, df.shape[0], df.shape[1])
                frames.append(df)

            if not frames:
                return None

            df = pd.concat(frames, ignore_index=True)
            logger.info("Combined from XLSX: %d rows", len(df))
            return df
        except Exception as exc:
            logger.error("Failed to load XLSX %s: %s", xlsx_path, exc)
            return None

    def load_raw_data(self) -> pd.DataFrame:
        # Try XLSX first (Kaggle format)
        df = self._load_from_xlsx()
        if df is not None:
            return df

        # Fall back to individual CSVs
        df_0910 = self._safe_read_csv(
            os.path.join(self.data_dir, FILE_2009_2010),
            "2009-2010 transactions",
        )
        df_1011 = self._safe_read_csv(
            os.path.join(self.data_dir, FILE_2010_2011),
            "2010-2011 transactions",
        )

        frames = []
        if df_0910 is not None:
            frames.append(df_0910)
        if df_1011 is not None:
            frames.append(df_1011)

        if not frames:
            raise FileNotFoundError(
                f"Neither XLSX ({FILE_XLSX}) nor CSVs ({FILE_2009_2010}, {FILE_2010_2011}) "
                f"found in {self.data_dir}"
            )

        df = pd.concat(frames, ignore_index=True)
        logger.info("Combined dataset: %d rows", len(df))
        return df

    # ── Preprocessing ────────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Parse InvoiceDate
        if "InvoiceDate" in df.columns:
            df["InvoiceDate"] = pd.to_datetime(
                df["InvoiceDate"], errors="coerce", dayfirst=True,
            )
            before = len(df)
            df = df.dropna(subset=["InvoiceDate"])
            dropped = before - len(df)
            if dropped:
                logger.info("Dropped %d rows with null InvoiceDate", dropped)

        # Remove cancellations (invoice starts with C)
        if "Invoice" in df.columns:
            df["Invoice"] = df["Invoice"].astype(str).str.strip()
            n_cancel = df["Invoice"].str.startswith(CANCELLATION_PREFIX).sum()
            if n_cancel:
                df = df[~df["Invoice"].str.startswith(CANCELLATION_PREFIX)].copy()
                logger.info("Removed %d cancellation invoices", n_cancel)

        # Filter valid quantities and prices
        if "Quantity" in df.columns:
            df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
            df = df[df["Quantity"] > 0].copy()

        if "Price" in df.columns:
            df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)
            # Zero/negative prices are invalid for monetary-value features —
            # drop them outright (they also break log-transformed monetary
            # features).  Section 16 of the experiment spec.
            df = df[df["Price"] > 0].copy()
            cap = df["Price"].quantile(0.999)
            if cap > 0 and not np.isnan(cap):
                df["Price"] = df["Price"].clip(upper=cap)

        # Filter customers with valid IDs
        if "Customer ID" in df.columns:
            df = df.dropna(subset=["Customer ID"])
            df["Customer ID"] = df["Customer ID"].astype(str).str.strip()
            df = df[df["Customer ID"] != ""].copy()

        return df

    # ── Schema standardisation ───────────────────────────────────────

    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        mapping = {
            "Customer ID": "customer_id",
            "InvoiceDate": "event_time",
            "StockCode": "product_id",
        }
        df = df.rename(columns=mapping, errors="ignore")

        df["event_type"] = "purchase"

        # Build transaction_value = Quantity * Price
        qty = df.get("Quantity", pd.Series(0, index=df.index))
        price = df.get("Price", pd.Series(0, index=df.index))
        df["transaction_value"] = (pd.to_numeric(qty, errors="coerce").fillna(0)
                                    * pd.to_numeric(price, errors="coerce").fillna(0))

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
            "dataset_name": "online_retail_ii",
            "ecosystem_type": "habitual_retail",
            "citation": (
                "Chen, D., Online Retail II Data Set. UCI Machine Learning Repository, 2019. "
                "https://doi.org/10.24432/C5CG6D"
            ),
            "source_url": "https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci",
            "n_customers_approx": 5_000,
            "n_orders_approx": 1_000_000,
            "churn_window_days": 90,
            "churn_justification": (
                "90 days — gift retail has seasonal patterns; 90 days captures "
                "~2× median inter-purchase interval (~45 days) without conflating "
                "seasonal absence with churn."
            ),
            "uses_native_churn_label": False,
            "available_feature_groups": self.available_feature_groups,
        }
