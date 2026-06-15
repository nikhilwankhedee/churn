"""
Abstract base class for all dataset adapters.

Each dataset adapter implements a common contract that the unified research
pipeline uses to load, preprocess, and standardise data from heterogeneous
ecosystems into a shared behavioural feature representation.

Ecosystem types (used for cross-dataset analysis):
    - transactional_marketplace   (Olist, REES46)
    - clickstream_commerce        (RetailRocket)
    - habitual_retail             (Online Retail II, Instacart)
    - subscription                (Telco)
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


class BaseDatasetAdapter(ABC):
    """Contract every dataset adapter must satisfy."""

    # ── Identifiers ──────────────────────────────────────────────────

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Short canonical name, e.g. 'olist', 'instacart'."""

    @property
    @abstractmethod
    def ecosystem_type(self) -> str:
        """One of: transactional_marketplace, clickstream_commerce,
        habitual_retail, subscription."""

    # ── Data loading & preprocessing ─────────────────────────────────

    @abstractmethod
    def load_raw_data(self) -> pd.DataFrame:
        """Load all raw data files and return a single merged DataFrame.

        The returned DataFrame uses the adapter's native column names.
        Missing / optional files are handled gracefully (logged, skipped).
        Raises FileNotFoundError if required files are missing.
        """

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dataset-specific cleaning: timestamp parsing, outlier clipping,
        value imputation, filtering invalid rows.

        Operates on native column names.
        """

    @abstractmethod
    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename native columns into the unified schema.

        Unified columns (subset may be present):
            customer_id          — unique customer / user identifier
            event_time           — timestamp of the event (datetime)
            transaction_value    — monetary value of the event
            event_type           — type: purchase, view, cart_add, etc.
            product_id           — product / item identifier
            review_score         — numeric review / rating
            payment_type         — payment method
            delivery_delay       — delivery delay in days (negative = early)
            engagement_signal    — numeric engagement intensity
            session_id           — session identifier
        """

    # ── Churn definition ─────────────────────────────────────────────

    @property
    def churn_window_days(self) -> Optional[int]:
        """Inactivity window in days, or None if native label is used."""
        return None

    @property
    def uses_native_churn_label(self) -> bool:
        """True if the dataset provides its own churn label (e.g. Telco)."""
        return False

    @property
    def has_temporal_data(self) -> bool:
        """True if the dataset has genuine event timestamps.

        Datasets without a real temporal dimension (e.g. Telco, whose
        event_time is derived synthetically from tenure) override this to
        False so feature engineering does not filter on event_time.
        """
        return True

    def get_native_churn_labels(
        self, df: pd.DataFrame, cutoff_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return a DataFrame with columns [customer_id, churn] computed
        from native labels.

        Only called when uses_native_churn_label is True.
        By default raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.dataset_name} does not implement get_native_churn_labels."
        )

    # ── Feature groups ───────────────────────────────────────────────

    @property
    @abstractmethod
    def available_feature_groups(self) -> List[str]:
        """List of feature group names this dataset supports.

        Standard groups: purchase, monetary, inactivity, review, delivery,
        payment, engagement, cadence.
        """

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Return dict of feature group -> list of standardized columns
        required for that group.

        Override if the default mapping is incorrect for this dataset.
        """
        return {
            "purchase":   ["customer_id", "event_time", "event_type"],
            "monetary":   ["customer_id", "event_time", "transaction_value"],
            "inactivity": ["customer_id", "event_time"],
            "review":     ["customer_id", "review_score"],
            "delivery":   ["customer_id", "delivery_delay"],
            "payment":    ["customer_id", "payment_type", "transaction_value"],
            "engagement": ["customer_id", "event_type"],
            "cadence":    ["customer_id", "event_time"],
        }

    # ── Metadata ─────────────────────────────────────────────────────

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Dataset metadata for experiment tracking.

        Should include at minimum:
            dataset_name, ecosystem_type, citation (if any),
            source_url, n_customers_approx, n_orders_approx,
            churn_window_days, uses_native_churn_label.
        """

    # ── Schema validation ────────────────────────────────────────────

    def validate_schema(self, df: pd.DataFrame) -> dict:
        """Validate the standardised schema after standardize_schema().

        Checks required columns, null ratios, timestamp validity, customer
        ID completeness, duplicate detection, and feature group availability.
        Logs every finding at the VALIDATION level.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame after standardize_schema() has been applied.

        Returns
        -------
        dict with keys: dataset, n_rows, n_columns, detected_columns,
        missing_optional_columns, enabled_feature_groups,
        disabled_feature_groups, column_null_pcts, warnings, errors.
        """
        from src.config import STANDARD_FEATURE_GROUPS
        from src.validators import validate_schema as _validate_schema
        return _validate_schema(
            df=df,
            dataset_name=self.dataset_name,
            available_groups=self.available_feature_groups,
            all_groups=STANDARD_FEATURE_GROUPS,
        )

    # ── Behavioral validation ────────────────────────────────────────

    def validate_behavioral_statistics(
        self,
        df: pd.DataFrame,
        labels: 'pd.DataFrame' = None,
    ) -> dict:
        """Compute and validate behavioral statistics for this dataset.

        Delegates to src.validators.validate_behavioral_statistics.
        """
        from src.validators import validate_behavioral_statistics
        return validate_behavioral_statistics(
            df=df,
            labels=labels,
            dataset_name=self.dataset_name,
        )

    # ── Data directory helper ────────────────────────────────────────

    @property
    def required_files(self) -> List[str]:
        """List of CSV filenames required by this adapter.

        Override in subclasses to specify required files for validation.
        """
        return []

    @property
    def alternate_filenames(self) -> Dict[str, List[str]]:
        """Map each required filename → list of acceptable alternate names.

        Some datasets ship under different filenames depending on the
        source (e.g. Kaggle vs direct download).  The dataset resolver
        treats a required file as present if ANY of its alternates exists.
        """
        return {}

    @property
    def data_dir(self) -> str:
        """Return the directory where raw data files are stored.

        Resolution order:
        1. _resolved_data_dir if set externally (via get_dataset/data_dir param)
        2. Centralized dataset resolver (discovery, environment, config)
        """
        if hasattr(self, '_resolved_data_dir') and self._resolved_data_dir:
            return str(self._resolved_data_dir)

        from src.dataset_resolver import resolve_dataset_directory
        try:
            return resolve_dataset_directory(
                dataset_name=self.dataset_name,
                required_files=self.required_files or None,
            )
        except FileNotFoundError:
            # Fall back to config-level default for backward compat
            from src.config import DATA_DIR
            return DATA_DIR

    @data_dir.setter
    def data_dir(self, value) -> None:
        """Override the data directory for this adapter instance."""
        self._resolved_data_dir = str(value) if value is not None else None
