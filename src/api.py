"""
Python API for the Churn Research Framework.

Provides a programmatic interface wrapping the existing pipeline,
profiling, batch execution, reporting, and exploration modules.

Usage:
    from src.api import ChurnFramework

    fw = ChurnFramework()
    result = fw.run("olist")
    profile = fw.profile("olist")
    comparison = fw.compare(["olist", "rees46"])
"""
import os
from typing import Any, Dict, List, Optional

from src.utils import get_logger

logger = get_logger(__name__)


class ChurnFramework:
    """High-level Python interface to the Churn Research Framework.

    Parameters
    ----------
    config_path : str, optional
        Path to a YAML configuration file.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._config_path = config_path
        if config_path:
            from src.config import load_config
            load_config(config_path)

    # ── Pipeline ───────────────────────────────────────────────────

    def run(
        self,
        dataset: str,
        sensitivity: bool = False,
        churn_window: Optional[int] = None,
        data_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full prediction pipeline on a dataset.

        Parameters
        ----------
        dataset : str
            Name of the registered dataset (e.g. 'olist', 'rees46').
        sensitivity : bool
            Whether to run sensitivity analysis.
        churn_window : int, optional
            Override churn prediction window in days.
        data_dir : str, optional
            Explicit directory containing raw data files. Overrides all
            automatic path resolution.

        Returns
        -------
        Pipeline result dictionary with models, metrics, and outputs.
        """
        from src.pipeline import run_pipeline
        return run_pipeline(
            dataset=dataset,
            sensitivity=sensitivity,
            churn_window_override=churn_window,
            data_dir=data_dir,
        )

    def run_batch(
        self,
        datasets: Optional[List[str]] = None,
        sensitivity: bool = False,
        data_dir: Optional[str] = None,
    ):
        """Run the pipeline on multiple datasets.

        Parameters
        ----------
        datasets : list of str, optional
            Datasets to run. Default: all registered datasets.
        sensitivity : bool
            Whether to run sensitivity analysis.
        data_dir : str, optional
            Explicit directory containing raw data files.

        Returns
        -------
        BatchResult with per-dataset results and benchmark table.
        """
        from src.batch import run_batch
        return run_batch(
            datasets=datasets,
            sensitivity=sensitivity,
            data_dir=data_dir,
        )

    # ── Profiling ──────────────────────────────────────────────────

    def profile(self, dataset: str, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """Profile a registered dataset.

        Loads, preprocesses, standardizes, and profiles the dataset.

        Parameters
        ----------
        dataset : str
            Name of the registered dataset.
        data_dir : str, optional
            Explicit directory containing raw data files.

        Returns
        -------
        Dataset profile as a dictionary.
        """
        from src.datasets import get_dataset
        from src.profiling import profile_dataset

        adapter = get_dataset(dataset, data_dir=data_dir)
        df = adapter.load_raw_data()
        df = adapter.preprocess(df)
        df = adapter.standardize_schema(df)
        profile = profile_dataset(df)
        return profile.to_dict()

    def profile_csv(self, csv_path: str, **kwargs) -> Dict[str, Any]:
        """Profile a raw CSV file (not yet registered).

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.
        **kwargs
            Additional arguments passed to inspect_csv().

        Returns
        -------
        Inspection result as a dictionary.
        """
        from src.wizard import inspect_csv
        import dataclasses

        result = inspect_csv(csv_path, **kwargs)
        return dataclasses.asdict(result)

    # ── Validation ─────────────────────────────────────────────────

    def validate(self, dataset: str, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """Validate a registered dataset's schema and behavior.

        Parameters
        ----------
        dataset : str
            Name of the registered dataset.
        data_dir : str, optional
            Explicit directory containing raw data files.

        Returns
        -------
        Validation report dictionary.
        """
        from src.datasets import get_dataset

        adapter = get_dataset(dataset, data_dir=data_dir)
        df = adapter.load_raw_data()
        df = adapter.preprocess(df)
        df = adapter.standardize_schema(df)

        schema_report = adapter.validate_schema(df)
        behavioral_report = adapter.validate_behavioral_statistics(df)

        return {
            "schema": schema_report,
            "behavioral": behavioral_report,
        }

    def validate_config(self, config_path: str) -> Dict[str, Any]:
        """Validate a YAML configuration file.

        Parameters
        ----------
        config_path : str
            Path to the YAML config file.

        Returns
        -------
        Validation result dictionary with 'is_valid', 'errors', 'warnings'.
        """
        from src.config_validation import validate_config_file
        import dataclasses

        result = validate_config_file(config_path)
        return dataclasses.asdict(result)

    # ── Experiment History ─────────────────────────────────────────

    def list_experiments(
        self,
        dataset: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List experiment runs from history.

        Parameters
        ----------
        dataset : str, optional
            Filter by dataset name.
        limit : int
            Maximum number of experiments to return.

        Returns
        -------
        List of experiment dictionaries, most recent first.
        """
        from src.explorer import list_experiments
        return list_experiments(dataset=dataset, limit=limit)

    def compare(
        self,
        datasets: List[str],
    ):
        """Compare experiments across multiple datasets.

        Parameters
        ----------
        datasets : list of str
            Dataset names to compare (minimum 2).

        Returns
        -------
        pandas DataFrame with comparison metrics, or None.
        """
        from src.explorer import compare_experiments
        return compare_experiments(datasets)

    def get_feature_comparison(self):
        """Compare dominant feature groups across datasets.

        Returns
        -------
        pandas DataFrame or None.
        """
        from src.explorer import get_feature_comparison
        return get_feature_comparison()

    # ── Reports ────────────────────────────────────────────────────

    def generate_reports(self, pipeline_result: Dict[str, Any]) -> Dict[str, str]:
        """Generate reports from a pipeline result.

        Parameters
        ----------
        pipeline_result : dict
            Result from run() or pipeline execution.

        Returns
        -------
        Dictionary mapping report names to Markdown content.
        """
        from src.reports import generate_all_reports

        reports = generate_all_reports(pipeline_result)
        return {name: output.to_markdown() for name, output in reports.items()}

    # ── Registry Introspection ─────────────────────────────────────

    def list_datasets(self) -> List[str]:
        """List all registered dataset names."""
        from src.datasets import list_datasets
        return list_datasets()

    def list_models(self) -> List[str]:
        """List all registered model names."""
        from src.models import list_models
        return list_models()

    def list_metrics(self) -> List[str]:
        """List all registered metric names."""
        from src.metrics import list_metrics
        return list_metrics()

    def list_strategies(self) -> List[str]:
        """List all registered churn strategy names."""
        from src.churn import list_strategies
        return list_strategies()

    def list_resamplers(self) -> List[str]:
        """List all registered resampler names."""
        from src.resamplers import list_resamplers
        return list_resamplers()

    def list_reports(self) -> List[str]:
        """List all registered report names."""
        from src.reports import list_reports
        return list_reports()

    # ── Wizard ─────────────────────────────────────────────────────

    def register_dataset(
        self,
        csv_path: str,
        name: Optional[str] = None,
        ecosystem: Optional[str] = None,
        customer_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        output: Optional[str] = None,
        source_url: str = "",
        citation: str = "",
    ) -> str:
        """Register a new dataset from a CSV file.

        Inspects the CSV, infers column roles, and generates a YAML config.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.
        name : str, optional
            Dataset name override.
        ecosystem : str, optional
            Ecosystem type override.
        customer_id : str, optional
            Customer ID column name.
        timestamp : str, optional
            Timestamp column name.
        output : str, optional
            Output path for the YAML config.
        source_url : str
            URL where the dataset can be obtained.
        citation : str
            How to cite this dataset.

        Returns
        -------
        Path to the generated YAML config file.
        """
        from src.wizard import inspect_csv, generate_config
        from src.config import PROJECT_ROOT

        inspection = inspect_csv(
            csv_path,
            customer_id_hint=customer_id,
            timestamp_hint=timestamp,
        )

        config = generate_config(
            inspection,
            dataset_name=name,
            ecosystem_type=ecosystem,
            source_url=source_url,
            citation=citation,
        )

        if output is None:
            from src.config import get_configs_dir
            output = str(
                get_configs_dir() / "datasets" / f"{config.dataset_name}.yaml"
            )

        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w") as f:
            f.write(config.to_yaml())

        logger.info("Generated config: %s", output)
        return output

    # ── Health Check ───────────────────────────────────────────────

    def doctor(self) -> Dict[str, Any]:
        """Run a health check on the framework.

        Returns
        -------
        Dictionary with component statuses.
        """
        checks = {}

        # Core registry
        try:
            from src.core.registry import registry
            cats = registry.list_categories()
            checks["core_registry"] = {"ok": True, "detail": f"{len(cats)} categories"}
        except Exception as e:
            checks["core_registry"] = {"ok": False, "detail": str(e)}

        # Datasets
        try:
            from src.datasets import list_datasets
            ds = list_datasets()
            checks["datasets"] = {"ok": True, "detail": ds}
        except Exception as e:
            checks["datasets"] = {"ok": False, "detail": str(e)}

        # Models
        try:
            from src.models import list_models
            models = list_models()
            checks["models"] = {"ok": True, "detail": models}
        except Exception as e:
            checks["models"] = {"ok": False, "detail": str(e)}

        # Metrics
        try:
            from src.metrics import list_metrics
            metrics = list_metrics()
            checks["metrics"] = {"ok": True, "detail": metrics}
        except Exception as e:
            checks["metrics"] = {"ok": False, "detail": str(e)}

        # Strategies
        try:
            from src.churn import list_strategies
            strats = list_strategies()
            checks["strategies"] = {"ok": True, "detail": strats}
        except Exception as e:
            checks["strategies"] = {"ok": False, "detail": str(e)}

        # Resamplers
        try:
            from src.resamplers import list_resamplers
            resamplers = list_resamplers()
            checks["resamplers"] = {"ok": True, "detail": resamplers}
        except Exception as e:
            checks["resamplers"] = {"ok": False, "detail": str(e)}

        # Reports
        try:
            from src.reports import list_reports
            reports = list_reports()
            checks["reports"] = {"ok": True, "detail": reports}
        except Exception as e:
            checks["reports"] = {"ok": False, "detail": str(e)}

        # Dependencies
        deps = {
            "pandas": "pandas", "numpy": "numpy", "sklearn": "scikit-learn",
            "xgboost": "xgboost", "yaml": "PyYAML",
        }
        for mod_name, pkg_name in deps.items():
            try:
                __import__(mod_name)
                checks[pkg_name] = {"ok": True, "detail": "installed"}
            except ImportError:
                checks[pkg_name] = {"ok": False, "detail": "not installed"}

        return checks

    # ── Benchmark ─────────────────────────────────────────────────

    def benchmark(
        self,
        dataset_root: str,
        output_dir: Optional[str] = None,
        sensitivity: bool = False,
        max_depth: int = 5,
        register_unknowns: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Discover and benchmark all datasets in a directory.

        Recursively scans the dataset root, identifies datasets by their
        file signatures, and executes the full pipeline.

        Parameters
        ----------
        dataset_root : str
            Root directory to scan (e.g. '/kaggle/input' or 'datasets').
        output_dir : str, optional
            Output directory for results.
        sensitivity : bool
            Whether to run sensitivity analysis.
        max_depth : int
            Maximum directory depth to scan.
        register_unknowns : bool
            If True, launch wizard for unknown datasets.
        dry_run : bool
            If True, only scan and report without executing.

        Returns
        -------
        BenchmarkResult with execution details.
        """
        from src.benchmark import benchmark as run_benchmark
        result = run_benchmark(
            dataset_root=dataset_root,
            output_dir=output_dir,
            sensitivity=sensitivity,
            max_depth=max_depth,
            register_unknowns=register_unknowns,
            dry_run=dry_run,
        )
        return {
            "dataset_root": result.dataset_root,
            "output_dir": result.output_dir,
            "discovered_datasets": result.discovered_datasets,
            "executed_datasets": result.executed_datasets,
            "registered_datasets": result.registered_datasets,
            "unknown_datasets": result.unknown_datasets,
            "results": result.results,
            "errors": result.errors,
            "total_duration": result.total_duration,
        }

    def discover(
        self,
        dataset_root: str,
        max_depth: int = 5,
    ) -> List[Dict[str, Any]]:
        """Discover datasets in a directory without executing.

        Parameters
        ----------
        dataset_root : str
            Root directory to scan.
        max_depth : int
            Maximum directory depth to scan.

        Returns
        -------
        List of discovery dicts with name, confidence, source, etc.
        """
        from src.benchmark import discover_only
        return discover_only(dataset_root, max_depth)

    def detect_environment(
        self,
        dataset_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect the current execution environment.

        Parameters
        ----------
        dataset_root : str, optional
            Dataset root to check.

        Returns
        -------
        Environment info dict.
        """
        from src.environment import detect_environment
        env = detect_environment(dataset_root)
        return {
            "name": env.name,
            "is_kaggle": env.is_kaggle,
            "is_colab": env.is_colab,
            "is_local": env.is_local,
            "dataset_root": str(env.dataset_root) if env.dataset_root else None,
            "working_dir": str(env.working_dir) if env.working_dir else None,
            "default_output_dir": str(env.default_output_dir) if env.default_output_dir else None,
        }
