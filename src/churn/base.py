"""
Abstract base class for churn labeling strategies.

All churn strategies implement the same interface: given a DataFrame,
cutoff date, and parameters, produce a binary churn label for each customer.

This abstraction enables:
- Swapping churn definitions without changing pipeline code
- Comparing churn definitions within the same pipeline run
- Registering custom strategies via the plugin registry
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ChurnResult:
    """Result container for churn labeling strategies.

    Attributes
    ----------
    labels : pd.DataFrame
        DataFrame with customer_id and 'churn' columns.
    strategy_name : str
        Name of the strategy that produced these labels.
    metadata : dict
        Strategy-specific metadata (window size, thresholds, etc.).
    """
    labels: pd.DataFrame
    strategy_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChurnStrategy(ABC):
    """Abstract base for churn labeling strategies.

    Subclasses must implement:
        - name: strategy identifier
        - required_columns: columns the strategy needs in the input df
        - label(df, cutoff_date, **kwargs) -> ChurnResult
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this strategy (e.g. 'inactivity', 'subscription')."""
        ...

    @property
    @abstractmethod
    def required_columns(self) -> List[str]:
        """Column names this strategy requires in the input DataFrame."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of the churn definition."""
        return self.__class__.__doc__ or ""

    @abstractmethod
    def label(
        self,
        df: pd.DataFrame,
        cutoff_date: pd.Timestamp,
        customer_id_col: str = "customer_id",
        **kwargs: Any,
    ) -> ChurnResult:
        """Produce churn labels for all active customers at the given cutoff.

        Parameters
        ----------
        df : pd.DataFrame
            Raw event-level data (must contain required_columns).
        cutoff_date : pd.Timestamp
            Temporal cutoff — only events before this date define the cohort.
        customer_id_col : str
            Name of the customer identifier column.
        **kwargs : Any
            Strategy-specific parameters.

        Returns
        -------
        ChurnResult with labels and metadata.
        """
        ...

    def validate_columns(self, df: pd.DataFrame) -> None:
        """Raise ValueError if required columns are missing."""
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Strategy '{self.name}' requires columns {missing}, "
                f"but DataFrame has columns {list(df.columns)}"
            )
