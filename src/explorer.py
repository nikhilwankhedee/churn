"""
Experiment explorer — filesystem-based history and comparison.

Reads the experiment log (results/experiments/experiment_log.csv)
and provides functions for listing, comparing, and inspecting runs.

Lightweight implementation — no database required.

Usage:
    from src.explorer import list_experiments, compare_experiments

    experiments = list_experiments()
    comparison = compare_experiments("run1", "run2")
"""
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import RESULTS_DIR
from src.utils import get_logger

logger = get_logger(__name__)

EXPERIMENT_LOG = os.path.join(RESULTS_DIR, "experiments", "experiment_log.csv")


def _load_experiment_log() -> Optional[pd.DataFrame]:
    """Load the experiment log CSV, or return None if not found."""
    if not os.path.exists(EXPERIMENT_LOG):
        return None
    try:
        return pd.read_csv(EXPERIMENT_LOG)
    except Exception as exc:
        logger.warning("Failed to load experiment log: %s", exc)
        return None


def list_experiments(
    dataset: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """List experiment runs, most recent first.

    Parameters
    ----------
    dataset : str, optional
        Filter by dataset name.
    limit : int
        Maximum number of experiments to return.

    Returns
    -------
    List of dicts, each representing one experiment run.
    """
    df = _load_experiment_log()
    if df is None or df.empty:
        return []

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False)

    if dataset and "dataset" in df.columns:
        df = df[df["dataset"] == dataset]

    df = df.head(limit)

    experiments = []
    for _, row in df.iterrows():
        exp = {}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                exp[col] = None
            else:
                exp[col] = val
        experiments.append(exp)

    return experiments


def get_experiment(
    dataset: str,
    limit: int = 1,
) -> Optional[Dict[str, Any]]:
    """Get the most recent experiment for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name.
    limit : int
        Number of most recent experiments to return.

    Returns
    -------
    Dict of experiment metadata, or None if not found.
    """
    experiments = list_experiments(dataset=dataset, limit=limit)
    if not experiments:
        return None
    if limit == 1:
        return experiments[0]
    return experiments


def compare_experiments(
    datasets: List[str],
) -> Optional[pd.DataFrame]:
    """Compare experiments across multiple datasets.

    Parameters
    ----------
    datasets : list of str
        Dataset names to compare.

    Returns
    -------
    DataFrame with one row per dataset, columns for key metrics.
    """
    df = _load_experiment_log()
    if df is None or df.empty:
        return None

    # Get most recent experiment per dataset
    rows = []
    for ds in datasets:
        ds_df = df[df["dataset"] == ds]
        if ds_df.empty:
            continue
        if "timestamp" in ds_df.columns:
            ds_df = ds_df.sort_values("timestamp", ascending=False)
        rows.append(ds_df.iloc[0])

    if not rows:
        return None

    comparison = pd.DataFrame(rows)

    # Select key columns for comparison
    key_cols = [
        "dataset", "timestamp", "best_model", "churn_rate",
        "imbalance_ratio", "pipeline_duration_sec",
    ]
    metric_cols = [c for c in comparison.columns if "_roc_auc" in c or "_f1" in c]

    display_cols = [c for c in key_cols + metric_cols if c in comparison.columns]
    return comparison[display_cols].reset_index(drop=True)


def get_model_metrics(
    dataset: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Get per-model metrics from the experiment log.

    Parameters
    ----------
    dataset : str, optional
        Filter by dataset name.

    Returns
    -------
    DataFrame with model metrics.
    """
    df = _load_experiment_log()
    if df is None or df.empty:
        return None

    if dataset and "dataset" in df.columns:
        df = df[df["dataset"] == dataset]

    # Extract metric columns
    metric_cols = ["dataset", "timestamp"]
    for col in df.columns:
        if any(col.endswith(f"_{m}") for m in ["roc_auc", "f1", "precision", "recall", "brier_score"]):
            metric_cols.append(col)

    available = [c for c in metric_cols if c in df.columns]
    return df[available].reset_index(drop=True) if available else None


def get_feature_comparison(
    datasets: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """Compare dominant feature groups across datasets.

    Returns
    -------
    DataFrame with dataset, dominant_feature_group, and churn_rate.
    """
    df = _load_experiment_log()
    if df is None or df.empty:
        return None

    if datasets:
        df = df[df["dataset"].isin(datasets)]

    cols = ["dataset", "dominant_feature_group", "churn_rate", "imbalance_ratio"]
    available = [c for c in cols if c in df.columns]

    if not available:
        return None

    result = df[available].drop_duplicates(subset=["dataset"], keep="first")
    return result.reset_index(drop=True)
