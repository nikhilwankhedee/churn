"""
Batch execution engine.

Runs the full pipeline on multiple datasets with:
- Isolated execution per dataset (failures don't stop others)
- Combined benchmark tables across datasets
- Aggregate comparison reports
- Clear execution summary

Usage:
    from src.batch import run_batch
    results = run_batch(datasets=["olist", "rees46"], sensitivity=False)
"""
import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class BatchResult:
    """Container for batch execution results.

    Attributes
    ----------
    results : dict
        Mapping dataset name -> pipeline result (or error dict).
    successful : list of str
        Datasets that completed successfully.
    failed : list of str
        Datasets that failed.
    total_duration : float
        Total wall-clock time in seconds.
    benchmark_table : dict
        Combined metrics across all successful datasets.
    """
    results: Dict[str, Any] = field(default_factory=dict)
    successful: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    total_duration: float = 0.0
    benchmark_table: Optional[Dict[str, Any]] = None


def run_batch(
    datasets: Optional[List[str]] = None,
    sensitivity: bool = False,
    churn_window_overrides: Optional[Dict[str, int]] = None,
    data_dir: Optional[str] = None,
) -> BatchResult:
    """Run the pipeline on multiple datasets.

    Parameters
    ----------
    datasets : list of str, optional
        Datasets to run. Default: all registered datasets.
    sensitivity : bool
        Whether to run sensitivity analysis for each dataset.
    churn_window_overrides : dict, optional
        Per-dataset churn window overrides.
    data_dir : str, optional
        Explicit directory containing raw data files. Applied to all
        datasets unless overridden per-dataset.

    Returns
    -------
    BatchResult with all results, successes, failures, and benchmarks.
    """
    from src.pipeline import run_pipeline

    if datasets is None:
        from src.datasets import list_datasets
        datasets = list_datasets()

    if churn_window_overrides is None:
        churn_window_overrides = {}

    start_time = datetime.datetime.utcnow()
    results = {}
    successful = []
    failed = []

    logger.info("=" * 60)
    logger.info("BATCH EXECUTION — %d datasets", len(datasets))
    logger.info("=" * 60)

    for i, ds in enumerate(datasets, 1):
        logger.info("── %d/%d: %s ──", i, len(datasets), ds)
        try:
            window = churn_window_overrides.get(ds)
            result = run_pipeline(
                dataset=ds,
                sensitivity=sensitivity,
                churn_window_override=window,
                data_dir=data_dir,
            )
            results[ds] = result
            successful.append(ds)
            logger.info("✓ %s — best model: %s", ds, result.get("best_model", "N/A"))
        except Exception as exc:
            logger.warning("✗ %s — failed: %s", ds, exc)
            results[ds] = {"error": str(exc), "dataset": ds}
            failed.append(ds)

    elapsed = (datetime.datetime.utcnow() - start_time).total_seconds()

    # Build benchmark table
    benchmark = _build_benchmark_table(results, successful)

    batch_result = BatchResult(
        results=results,
        successful=successful,
        failed=failed,
        total_duration=elapsed,
        benchmark_table=benchmark,
    )

    logger.info("=" * 60)
    logger.info(
        "BATCH COMPLETE — %d/%d succeeded, %.1fs total",
        len(successful), len(datasets), elapsed,
    )
    if failed:
        logger.warning("Failed: %s", ", ".join(failed))
    logger.info("=" * 60)

    return batch_result


def _build_benchmark_table(
    results: Dict[str, Any],
    successful: List[str],
) -> Dict[str, Any]:
    """Build a combined benchmark table from successful results."""
    if not successful:
        return {}

    rows = []
    for ds in successful:
        r = results[ds]
        rows.append({
            "dataset": ds,
            "ecosystem": r.get("ecosystem_type", "N/A"),
            "churn_rate": r.get("churn_rate", 0),
            "imbalance_ratio": r.get("imbalance_ratio", 0),
            "best_model": r.get("best_model", "N/A"),
            "dominant_feature_group": r.get("dominant_feature_group", "N/A"),
            "duration_seconds": r.get("duration_seconds", 0),
            "schema_errors": r.get("schema_errors", 0),
        })

    # Compute summary statistics
    churn_rates = [r["churn_rate"] for r in rows]
    durations = [r["duration_seconds"] for r in rows]

    return {
        "datasets": rows,
        "summary": {
            "total_datasets": len(rows),
            "mean_churn_rate": sum(churn_rates) / len(churn_rates) if churn_rates else 0,
            "total_duration_seconds": sum(durations),
            "ecosystems": list(set(r["ecosystem"] for r in rows)),
        },
    }


def format_benchmark_table(benchmark: Dict[str, Any]) -> str:
    """Format a benchmark table as a readable string."""
    if not benchmark or "datasets" not in benchmark:
        return "No benchmark data available."

    lines = []
    lines.append("=" * 80)
    lines.append("CROSS-DATASET BENCHMARK TABLE")
    lines.append("=" * 80)
    lines.append("")

    # Header
    header = f"{'Dataset':<20} {'Ecosystem':<25} {'Churn':>8} {'Best Model':<20} {'Duration':>10}"
    lines.append(header)
    lines.append("-" * 80)

    for row in benchmark["datasets"]:
        line = (
            f"{row['dataset']:<20} "
            f"{row['ecosystem']:<25} "
            f"{row['churn_rate']:>7.1%} "
            f"{row['best_model']:<20} "
            f"{row['duration_seconds']:>9.1f}s"
        )
        lines.append(line)

    lines.append("-" * 80)

    summary = benchmark.get("summary", {})
    lines.append(
        f"{'TOTAL':<20} "
        f"{summary.get('total_datasets', 0)} datasets "
        f"{'':>15} "
        f"{'Mean churn:':>10} {summary.get('mean_churn_rate', 0):.1%}"
    )
    lines.append(
        f"{'':>20} "
        f"{'':>25} "
        f"{'':>8} "
        f"{'':>20} "
        f"{summary.get('total_duration_seconds', 0):>9.1f}s"
    )
    lines.append("=" * 80)

    return "\n".join(lines)
