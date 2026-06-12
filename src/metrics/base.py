"""
Abstract base class for evaluation metrics.

All metrics implement a common interface: given true labels and
(optionally) predictions and probabilities, return a scalar score.

This abstraction enables:
- Registering custom metrics via the plugin registry
- Swapping metric sets without changing pipeline code
- Comparing metric implementations (e.g., different ECE binning)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class MetricResult:
    """Container for a single metric evaluation.

    Attributes
    ----------
    name : str
        Metric identifier (e.g. 'roc_auc').
    value : float
        The computed metric value.
    metadata : dict
        Metric-specific details (e.g., threshold used, number of bins).
    """
    name: str
    value: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EvaluationMetric(ABC):
    """Abstract base for evaluation metrics.

    Subclasses must implement:
        - name: metric identifier
        - higher_is_better: whether higher values are better
        - evaluate(y_true, y_pred=None, y_proba=None) -> MetricResult
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this metric (e.g. 'roc_auc')."""
        ...

    @property
    @abstractmethod
    def higher_is_better(self) -> bool:
        """Whether higher values indicate better performance."""
        ...

    @property
    def requires_proba(self) -> bool:
        """Whether this metric requires probability estimates."""
        return False

    @property
    def requires_pred(self) -> bool:
        """Whether this metric requires hard predictions."""
        return True

    @property
    def description(self) -> str:
        """Human-readable description of the metric."""
        return self.__class__.__doc__ or ""

    @abstractmethod
    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: Optional[np.ndarray] = None,
        y_proba: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute the metric.

        Parameters
        ----------
        y_true : ground truth labels
        y_pred : hard predictions (0/1)
        y_proba : probability estimates for positive class
        **kwargs : metric-specific parameters

        Returns
        -------
        MetricResult with the computed value.
        """
        ...

    def __repr__(self) -> str:
        direction = "↑" if self.higher_is_better else "↓"
        return f"<Metric: {self.name} {direction}>"
