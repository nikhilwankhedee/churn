"""
Unified benchmark pipeline — automatic dataset discovery and execution.

A single entry point that scans directories, identifies datasets,
runs readiness checks, and executes experiments with zero manual
path editing.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark execution."""
    dataset_root: str
    output_dir: str
    discovered_datasets: list[str] = field(default_factory=list)
    executed_datasets: list[str] = field(default_factory=list)
    skipped_datasets: list[str] = field(default_factory=list)
    registered_datasets: list[str] = field(default_factory=list)
    unknown_datasets: list[str] = field(default_factory=list)
    results: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    total_duration: float = 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Benchmark complete — {len(self.executed_datasets)} datasets executed",
            f"Discovered: {', '.join(self.discovered_datasets) or 'none'}",
            f"Executed: {', '.join(self.executed_datasets) or 'none'}",
            f"Skipped: {', '.join(self.skipped_datasets) or 'none'}",
            f"Registered: {', '.join(self.registered_datasets) or 'none'}",
            f"Unknown: {', '.join(self.unknown_datasets) or 'none'}",
            f"Duration: {self.total_duration:.1f}s",
        ]
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for e in self.errors:
                lines.append(f"  - {e}")
        return "\n".join(lines)


def benchmark(
    dataset_root: str,
    output_dir: Optional[str] = None,
    sensitivity: bool = False,
    max_depth: int = 5,
    register_unknowns: bool = False,
    dry_run: bool = False,
    config_path: Optional[str] = None,
) -> BenchmarkResult:
    """Run the unified benchmark pipeline.

    Discovers all datasets in dataset_root, identifies them using file
    signatures and registered schemas, runs readiness checks, and
    executes experiments.

    Parameters
    ----------
    dataset_root : str
        Root directory to scan for datasets (e.g. '/kaggle/input').
    output_dir : str, optional
        Output directory. Defaults based on environment.
    sensitivity : bool
        Whether to run sensitivity analysis.
    max_depth : int
        Maximum directory depth for scanning.
    register_unknowns : bool
        If True, launch wizard for unknown datasets.
    dry_run : bool
        If True, only scan and report without executing.
    config_path : str, optional
        Path to YAML configuration file.

    Returns
    -------
    BenchmarkResult with execution details.
    """
    import time
    start = time.time()

    from src.paths import resolve, ensure_dir, output_directory
    from src.environment import detect_environment
    from src.discovery.scanner import scan_directory
    from src.registry_store import get_registry_store
    from src.datasets import list_datasets, get_dataset

    # Resolve paths
    dataset_root_path = resolve(dataset_root)

    # Load config if provided
    if config_path:
        from src.config import load_config
        load_config(config_path)

    # Determine output directory
    out_path = output_directory(dataset_root, output_dir)

    # Initialize registry store and sync with existing configs
    registry = get_registry_store()
    registry.ensure_synced()

    result = BenchmarkResult(
        dataset_root=str(dataset_root_path),
        output_dir=str(out_path),
    )

    # Detect environment for informational purposes
    env = detect_environment(dataset_root)
    logger.info(
        "Environment: %s | Dataset root: %s | Output: %s",
        env.name, dataset_root_path, out_path,
    )

    # Scan for datasets
    schema_info = registry.get_schema_info()
    scan_result = scan_directory(
        root=dataset_root_path,
        max_depth=max_depth,
        previously_registered=schema_info,
    )

    logger.info(scan_result.summary())

    # Process discovered datasets
    for discovered in scan_result.high_confidence:
        ds_name = discovered.dataset_name
        result.discovered_datasets.append(ds_name)

        # Check if it's a built-in adapter
        builtin_datasets = list_datasets()
        if ds_name in builtin_datasets:
            if dry_run:
                result.executed_datasets.append(ds_name)
                logger.info("[DRY RUN] Would execute: %s", ds_name)
                continue

            # Execute pipeline with discovered directory
            discovered_dir = str(discovered.directory)
            try:
                _execute_pipeline(
                    ds_name, sensitivity, out_path, result,
                    data_dir=discovered_dir,
                )
                result.executed_datasets.append(ds_name)
            except Exception as exc:
                result.errors.append(f"{ds_name}: {exc}")
                logger.error("Pipeline failed for %s: %s", ds_name, exc)

        elif discovered.source == "registered":
            # Previously registered dataset — load from config
            registered = registry.get(ds_name)
            if registered:
                result.registered_datasets.append(ds_name)
                logger.info(
                    "Recognized previously registered dataset: %s (from %s)",
                    ds_name, registered.config_path,
                )
                if not dry_run:
                    try:
                        _execute_pipeline(
                            ds_name, sensitivity, out_path, result,
                            config_override=str(registered.config_path),
                        )
                        result.executed_datasets.append(ds_name)
                    except Exception as exc:
                        result.errors.append(f"{ds_name}: {exc}")
                        logger.error("Pipeline failed for %s: %s", ds_name, exc)
            else:
                result.unknown_datasets.append(ds_name)

        elif register_unknowns:
            result.unknown_datasets.append(ds_name)
            logger.info("Unknown dataset detected: %s in %s", ds_name, discovered.directory)

        else:
            result.unknown_datasets.append(ds_name)
            logger.info(
                "Skipping unknown dataset: %s (use register_unknowns=True to process)",
                ds_name,
            )

    # Also run any built-in datasets not found via scanning
    # but available in the registry (backward compatibility)
    for ds_name in builtin_datasets:
        if ds_name not in result.executed_datasets:
            # Check if it has data available through its own data_dir
            try:
                adapter = get_dataset(ds_name)
                data_dir_path = Path(adapter.data_dir)
                if data_dir_path.is_dir() and any(data_dir_path.glob("*.csv")):
                    if dry_run:
                        result.executed_datasets.append(ds_name)
                        logger.info("[DRY RUN] Would execute (from data_dir): %s", ds_name)
                    else:
                        try:
                            _execute_pipeline(
                                ds_name, sensitivity, out_path, result,
                                data_dir=str(data_dir_path),
                            )
                            result.executed_datasets.append(ds_name)
                        except Exception as exc:
                            result.errors.append(f"{ds_name}: {exc}")
            except Exception:
                pass

    result.total_duration = time.time() - start
    return result


def _execute_pipeline(
    dataset_name: str,
    sensitivity: bool,
    output_dir: Path,
    result: BenchmarkResult,
    config_override: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> None:
    """Execute the pipeline for a single dataset."""
    from src.pipeline import run_pipeline

    if config_override:
        from src.config import load_config
        load_config(config_override)

    logger.info("Executing pipeline for dataset: %s", dataset_name)
    pipeline_result = run_pipeline(
        dataset=dataset_name,
        sensitivity=sensitivity,
        data_dir=data_dir,
    )

    result.results[dataset_name] = pipeline_result
    logger.info("Completed pipeline for dataset: %s", dataset_name)


def discover_only(
    dataset_root: str,
    max_depth: int = 5,
) -> list[dict]:
    """Discover datasets without executing.

    Returns a list of dicts with discovery information suitable for
    display or inspection.
    """
    from src.paths import resolve
    from src.discovery.scanner import scan_directory
    from src.registry_store import get_registry_store

    root = resolve(dataset_root)
    registry = get_registry_store()
    registry.ensure_synced()

    scan_result = scan_directory(
        root=root,
        max_depth=max_depth,
        previously_registered=registry.get_schema_info(),
    )

    output = []
    for ds in scan_result.discovered:
        output.append({
            "name": ds.dataset_name,
            "adapter_key": ds.adapter_key,
            "confidence": ds.confidence,
            "source": ds.source,
            "directory": str(ds.directory),
            "matched_files": ds.detection.matched_files,
            "missing_files": ds.detection.missing_files,
            "is_high_confidence": ds.is_high_confidence,
        })

    return output
