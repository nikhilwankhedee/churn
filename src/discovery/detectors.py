"""
Built-in dataset detectors — identify datasets by file signatures.

Each detector inspects a directory's CSV files and returns a confidence
score for how likely it is to be a specific dataset. Identification is
based on column names and file structure, NOT folder names.
"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    """Result of attempting to detect a dataset."""
    matched: bool
    dataset_type: str
    confidence: float
    adapter_key: Optional[str] = None
    matched_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    details: str = ""


class DatasetDetector(ABC):
    """Base class for dataset signature detectors."""

    @property
    @abstractmethod
    def dataset_type(self) -> str:
        """Canonical dataset name (e.g. 'olist', 'telco')."""

    @property
    @abstractmethod
    def adapter_key(self) -> str:
        """Key to use in the dataset registry (e.g. 'olist', 'telco')."""

    @property
    @abstractmethod
    def required_files(self) -> list[str]:
        """List of required CSV filenames (without path prefix)."""

    @property
    def optional_files(self) -> list[str]:
        """List of optional CSV filenames."""
        return []

    @property
    def required_columns(self) -> dict[str, list[str]]:
        """Mapping of filename -> list of required column names.

        If provided, detection will also verify column names for higher
        confidence scoring.
        """
        return {}

    def _find_csvs(self, directory: Path) -> list[Path]:
        """Find all CSV files in directory (non-recursive)."""
        if not directory.is_dir():
            return []
        return sorted(directory.glob("*.csv"))

    def _get_all_filenames(self, directory: Path) -> list[str]:
        """Get all CSV filenames in the directory."""
        csvs = self._find_csvs(directory)
        return [f.name for f in csvs]

    def _get_column_names(self, filepath: Path, max_rows: int = 5) -> set[str]:
        """Read column names from a CSV file without loading full data."""
        try:
            import pandas as pd
            df = pd.read_csv(filepath, nrows=max_rows)
            return set(df.columns)
        except Exception:
            return set()

    def _match_required_files(
        self, directory: Path
    ) -> tuple[list[str], list[str]]:
        """Check which required files exist in the directory.

        Returns (matched, missing) lists of filenames.
        """
        available = set(self._get_all_filenames(directory))

        matched = []
        missing = []
        for req in self.required_files:
            # Check for exact match or common variations
            if req in available:
                matched.append(req)
            else:
                # Try without common prefixes
                found = False
                for avail in available:
                    avail_base = avail.lower().replace("_", "").replace("-", "")
                    req_base = req.lower().replace("_", "").replace("-", "")
                    if avail_base == req_base:
                        matched.append(req)
                        found = True
                        break
                if not found:
                    missing.append(req)

        return matched, missing

    def _match_columns(
        self, directory: Path, filename: str, required_cols: list[str]
    ) -> tuple[int, int]:
        """Check if a CSV has the expected columns.

        Returns (matched_count, total_required).
        """
        filepath = directory / filename
        if not filepath.exists():
            # Try variations
            csvs = self._find_csvs(directory)
            for csv in csvs:
                if csv.stem.lower().replace("_", "") == filename.lower().replace("_", ""):
                    filepath = csv
                    break
            else:
                return 0, len(required_cols)

        actual_cols = self._get_column_names(filepath)
        if not actual_cols:
            return 0, len(required_cols)

        matched = sum(1 for c in required_cols if c in actual_cols)
        return matched, len(required_cols)

    def detect(self, directory: Path) -> DetectionResult:
        """Detect if the given directory contains this dataset.

        Parameters
        ----------
        directory : Path
            Directory to inspect.

        Returns
        -------
        DetectionResult with confidence score.
        """
        directory = Path(directory)
        if not directory.is_dir():
            return DetectionResult(
                matched=False,
                dataset_type=self.dataset_type,
                confidence=0.0,
                adapter_key=self.adapter_key,
                details=f"Directory does not exist: {directory}",
            )

        # Step 1: Match required files
        matched_files, missing_files = self._match_required_files(directory)
        file_score = len(matched_files) / max(len(self.required_files), 1)

        # Step 2: Check optional files
        available = set(self._get_all_filenames(directory))
        optional_found = sum(1 for f in self.optional_files if f in available)
        optional_score = (
            optional_found / max(len(self.optional_files), 1)
            if self.optional_files else 0.0
        )

        # Step 3: Check column names if available
        column_score = 0.0
        n_column_checks = 0
        for filename, required_cols in self.required_columns.items():
            matched, total = self._match_columns(directory, filename, required_cols)
            if total > 0:
                column_score += matched / total
                n_column_checks += 1

        if n_column_checks > 0:
            column_score /= n_column_checks

        # Compute weighted confidence
        if self.required_columns:
            confidence = (file_score * 0.5) + (column_score * 0.4) + (optional_score * 0.1)
        else:
            confidence = (file_score * 0.7) + (optional_score * 0.3)

        # Must match all required files for a positive detection
        matched = len(missing_files) == 0 and file_score > 0

        details_parts = []
        if matched_files:
            details_parts.append(f"found: {matched_files}")
        if missing_files:
            details_parts.append(f"missing: {missing_files}")
        if n_column_checks > 0:
            details_parts.append(f"column_score: {column_score:.2f}")

        return DetectionResult(
            matched=matched,
            dataset_type=self.dataset_type,
            confidence=round(confidence, 3),
            adapter_key=self.adapter_key,
            matched_files=matched_files,
            missing_files=missing_files,
            details="; ".join(details_parts),
        )


# ══════════════════════════════════════════════════════════════════
#  BUILT-IN DATASET DETECTORS
# ══════════════════════════════════════════════════════════════════

class OlistDetector(DatasetDetector):
    @property
    def dataset_type(self) -> str:
        return "olist"

    @property
    def adapter_key(self) -> str:
        return "olist"

    @property
    def required_files(self) -> list[str]:
        return [
            "olist_orders_dataset.csv",
            "olist_customers_dataset.csv",
            "olist_order_payments_dataset.csv",
        ]

    @property
    def optional_files(self) -> list[str]:
        return [
            "olist_order_reviews_dataset.csv",
            "olist_order_items_dataset.csv",
            "olist_products_dataset.csv",
            "olist_sellers_dataset.csv",
            "olist_geolocation_dataset.csv",
            "product_category_name_translation.csv",
        ]

    @property
    def required_columns(self) -> dict[str, list[str]]:
        return {
            "olist_orders_dataset.csv": [
                "order_id", "customer_id", "order_purchase_timestamp",
            ],
            "olist_customers_dataset.csv": [
                "customer_id", "customer_unique_id",
            ],
            "olist_order_payments_dataset.csv": [
                "order_id", "payment_type", "payment_value",
            ],
        }


class Rees46Detector(DatasetDetector):
    @property
    def dataset_type(self) -> str:
        return "rees46"

    @property
    def adapter_key(self) -> str:
        return "rees46"

    @property
    def required_files(self) -> list[str]:
        return [
            "rees46_events.csv",
        ]

    @property
    def optional_files(self) -> list[str]:
        return [
            "rees46_users.csv",
            "rees46_items.csv",
        ]

    @property
    def required_columns(self) -> dict[str, list[str]]:
        return {
            "rees46_events.csv": [
                "timestamp", "user_id", "event_type",
            ],
        }


class RetailRocketDetector(DatasetDetector):
    @property
    def dataset_type(self) -> str:
        return "retailrocket"

    @property
    def adapter_key(self) -> str:
        return "retailrocket"

    @property
    def required_files(self) -> list[str]:
        return [
            "retailrocket_events.csv",
        ]

    @property
    def optional_files(self) -> list[str]:
        return [
            "retailrocket_items.csv",
            "retailrocket_category_tree.csv",
            "retailrocket_visits.csv",
        ]

    @property
    def required_columns(self) -> dict[str, list[str]]:
        return {
            "retailrocket_events.csv": [
                "timestamp", "visitorid", "event",
            ],
        }


class OnlineRetailIIDetector(DatasetDetector):
    @property
    def dataset_type(self) -> str:
        return "online_retail_ii"

    @property
    def adapter_key(self) -> str:
        return "online_retail_ii"

    @property
    def required_files(self) -> list[str]:
        return [
            "online_retail_II_2009_2010.csv",
            "online_retail_II_2010_2011.csv",
        ]

    @property
    def required_columns(self) -> dict[str, list[str]]:
        return {
            "online_retail_II_2009_2010.csv": [
                "Invoice", "InvoiceDate", "Customer ID",
            ],
        }


class InstacartDetector(DatasetDetector):
    @property
    def dataset_type(self) -> str:
        return "instacart"

    @property
    def adapter_key(self) -> str:
        return "instacart"

    @property
    def required_files(self) -> list[str]:
        return [
            "instacart_orders.csv",
            "instacart_products.csv",
        ]

    @property
    def optional_files(self) -> list[str]:
        return [
            "instacart_aisles.csv",
            "instacart_departments.csv",
            "instacart_order_products__prior.csv",
        ]

    @property
    def required_columns(self) -> dict[str, list[str]]:
        return {
            "instacart_orders.csv": [
                "order_id", "user_id",
            ],
        }


class TelcoDetector(DatasetDetector):
    @property
    def dataset_type(self) -> str:
        return "telco"

    @property
    def adapter_key(self) -> str:
        return "telco"

    @property
    def required_files(self) -> list[str]:
        return [
            "telco_customer_churn.csv",
        ]

    @property
    def required_columns(self) -> dict[str, list[str]]:
        return {
            "telco_customer_churn.csv": [
                "customerID", "Churn",
            ],
        }


# ══════════════════════════════════════════════════════════════════
#  DETECTOR REGISTRY
# ══════════════════════════════════════════════════════════════════

_ALL_DETECTORS: list[DatasetDetector] = [
    OlistDetector(),
    Rees46Detector(),
    RetailRocketDetector(),
    OnlineRetailIIDetector(),
    InstacartDetector(),
    TelcoDetector(),
]


def get_all_detectors() -> list[DatasetDetector]:
    """Return all built-in dataset detectors."""
    return list(_ALL_DETECTORS)


def detect_dataset(
    directory: Path,
    detectors: Optional[list[DatasetDetector]] = None,
) -> list[DetectionResult]:
    """Run all detectors against a directory.

    Parameters
    ----------
    directory : Path
        Directory to inspect.
    detectors : list of DatasetDetector, optional
        Detectors to use. Default: all built-in detectors.

    Returns
    -------
    List of DetectionResult objects, sorted by confidence (descending).
    """
    if detectors is None:
        detectors = _ALL_DETECTORS

    results = []
    for detector in detectors:
        try:
            result = detector.detect(directory)
            results.append(result)
        except Exception as exc:
            logger.debug(
                "Detector %s failed on %s: %s",
                detector.dataset_type, directory, exc,
            )

    results.sort(key=lambda r: r.confidence, reverse=True)
    return results
