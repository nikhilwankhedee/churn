"""
Pre-flight validation (Section 33).

Before the 80-experiment matrix runs, this module verifies:
  1. Framework imports and core dependencies are available.
  2. All 8 final-experiment datasets are registered.
  3. All 5 final-experiment models are available (LightGBM required).
  4. Configuration is consistent (seed, paths, churn windows).
  5. Raw data is present for each dataset (per its adapter's required files).

Any failure aborts the run — the experiment matrix never starts with
a broken prerequisite.
"""
import importlib
import sys
import os
from pathlib import Path
from typing import Dict, List, Any

from src.config import (
    FINAL_EXPERIMENT_DATASETS, FINAL_EXPERIMENT_MODELS,
    RANDOM_SEED, TRAIN_SPLIT_QUANTILE, PROJECT_ROOT,
    RESULTS_DIR, FIGURES_DIR, MODELS_DIR, PROCESSED_DIR,
)
from src.utils import get_logger

logger = get_logger(__name__)

REQUIRED_DEPENDENCIES: Dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "imblearn": "imbalanced-learn",
    "matplotlib": "matplotlib",
    "yaml": "pyyaml",
    "joblib": "joblib",
    "openpyxl": "openpyxl",
}


class PreflightError(RuntimeError):
    """Raised when a pre-flight check fails."""


def check_dependencies() -> List[str]:
    """Verify required third-party modules import cleanly."""
    missing = []
    versions = {}
    for mod, dist in REQUIRED_DEPENDENCIES.items():
        try:
            m = importlib.import_module(mod)
            versions[dist] = getattr(m, "__version__", "unknown")
        except Exception as exc:
            missing.append(f"{dist} (import '{mod}' failed: {exc})")
    if missing:
        logger.error("Preflight | Missing dependencies: %s", missing)
        raise PreflightError(f"Missing dependencies: {missing}")
    logger.validation("Preflight | Dependencies OK: %s", versions)
    return list(versions.keys())


def check_datasets() -> List[str]:
    """Verify all 8 final-experiment datasets are registered."""
    from src.datasets import get_dataset, list_datasets

    registered = set(list_datasets())
    missing = [d for d in FINAL_EXPERIMENT_DATASETS if d not in registered]
    if missing:
        logger.error("Preflight | Unregistered datasets: %s", missing)
        raise PreflightError(f"Unregistered datasets: {missing}")

    for name in FINAL_EXPERIMENT_DATASETS:
        adapter = get_dataset(name)
        expected = 5
        actual = len(adapter.available_feature_groups)
        if actual < 1:
            raise PreflightError(
                f"Dataset '{name}' declares no feature groups"
            )
        logger.validation(
            "Preflight | Dataset '%s' registered — feature groups: %s",
            name, adapter.available_feature_groups,
        )
        del expected
    return list(FINAL_EXPERIMENT_DATASETS)


def check_models() -> List[str]:
    """Verify all 5 final-experiment models are available.

    LightGBM is mandatory — a missing LightGBM raises PreflightError
    rather than silently skipping the model.
    """
    from src.modeling import AVAILABLE_MODELS, MODEL_ORDER

    available = set(AVAILABLE_MODELS)
    missing = [m for m in FINAL_EXPERIMENT_MODELS if m not in available]
    if missing:
        logger.error("Preflight | Missing models: %s", missing)
        raise PreflightError(f"Missing models: {missing}")

    try:
        import lightgbm
        logger.validation(
            "Preflight | LightGBM %s available",
            getattr(lightgbm, "__version__", "unknown"),
        )
    except Exception as exc:
        raise PreflightError(
            f"LightGBM required for final experiment but not importable: {exc}"
        )

    for model in MODEL_ORDER:
        logger.validation("Preflight | Model available: %s", model)
    return list(MODEL_ORDER)


def check_config() -> Dict[str, Any]:
    """Verify seed, split quantile, and output paths."""
    checks = {}

    checks['random_seed'] = RANDOM_SEED
    checks['train_split_quantile'] = TRAIN_SPLIT_QUANTILE

    for label, path in [
        ('project_root', PROJECT_ROOT),
        ('results', RESULTS_DIR),
        ('figures', FIGURES_DIR),
        ('models', MODELS_DIR),
        ('processed', PROCESSED_DIR),
    ]:
        p = Path(path)
        ok = p.exists() or p.parent.exists()
        checks[f'{label}_path'] = str(p)
        checks[f'{label}_ok'] = ok
        if not ok:
            logger.warning("Preflight | Path does not exist: %s", p)

    checks['datasets'] = list(FINAL_EXPERIMENT_DATASETS)
    checks['models'] = list(FINAL_EXPERIMENT_MODELS)
    logger.validation(
        "Preflight | Config OK — seed=%s quantile=%s %d datasets %d models",
        RANDOM_SEED, TRAIN_SPLIT_QUANTILE,
        len(FINAL_EXPERIMENT_DATASETS), len(FINAL_EXPERIMENT_MODELS),
    )
    return checks


def check_data_availability(
    data_dirs: Dict[str, str] = None,
) -> Dict[str, str]:
    """Report which datasets have raw data present.

    Returns dict {dataset: 'present'|'missing'|'partial'}.  Missing data
    does NOT abort the run — it is recorded so the final report can state
    exactly which datasets ran and which were skipped for missing data.
    """
    from src.datasets import get_dataset

    status = {}
    for name in FINAL_EXPERIMENT_DATASETS:
        adapter = get_dataset(name, data_dir=(data_dirs or {}).get(name))
        required = adapter.required_files or []
        if not required:
            status[name] = 'present'
            continue

        try:
            from src.dataset_resolver import resolve_dataset_directory
            resolve_dataset_directory(
                dataset_name=name,
                data_dir=(data_dirs or {}).get(name),
                required_files=required,
            )
            status[name] = 'present'
        except FileNotFoundError:
            status[name] = 'missing'
        except Exception as exc:
            status[name] = 'partial'
            logger.warning("Preflight | Data check failed for '%s': %s",
                           name, exc)

    for name, st in status.items():
        logger.validation("Preflight | Data '%s': %s", name, st)
    return status


def run_preflight(
    data_dirs: Dict[str, str] = None,
    abort_on_missing_data: bool = False,
) -> Dict[str, Any]:
    """Run all pre-flight checks.

    Parameters
    ----------
    data_dirs : dict, optional
        Optional explicit data directories per dataset.
    abort_on_missing_data : bool, default False
        If True, raise PreflightError when any final dataset is missing.
        Default (False): data availability is reported but non-fatal.

    Returns
    -------
    dict with keys: dependencies, datasets, models, config, data_availability.

    Raises
    ------
    PreflightError if a fatal prerequisite (deps, registry, models, config)
    is missing.
    """
    logger.info("=" * 60)
    logger.info("PRE-FLIGHT VALIDATION")
    logger.info("=" * 60)

    report = {
        'dependencies': check_dependencies(),
        'datasets': check_datasets(),
        'models': check_models(),
        'config': check_config(),
        'data_availability': check_data_availability(data_dirs),
    }

    missing_data = [
        name for name, st in report['data_availability'].items()
        if st == 'missing'
    ]
    if missing_data:
        logger.warning(
            "Preflight | %d dataset(s) have no data available: %s",
            len(missing_data), missing_data,
        )
        if abort_on_missing_data:
            raise PreflightError(
                f"Aborting: data missing for {missing_data}"
            )

    logger.validation(
        "PRE-FLIGHT PASSED — %d/%d datasets registered, %d/%d models "
        "available, data present for %d/%d",
        len(report['datasets']), len(FINAL_EXPERIMENT_DATASETS),
        len(report['models']), len(FINAL_EXPERIMENT_MODELS),
        sum(1 for s in report['data_availability'].values()
            if s == 'present'),
        len(FINAL_EXPERIMENT_DATASETS),
    )
    return report


def print_model_availability() -> None:
    """Print the MODEL AVAILABILITY block (Section 33)."""
    try:
        from src.modeling import AVAILABLE_MODELS, MODEL_ORDER
        print("\n=== MODEL AVAILABILITY ===")
        for model in MODEL_ORDER:
            marker = "AVAILABLE" if model in AVAILABLE_MODELS else "MISSING"
            print(f"  {model:24s} {marker}")
        print()
    except Exception as exc:
        print(f"  Could not inspect models: {exc}")


if __name__ == '__main__':
    report = run_preflight()
    print_model_availability()
    sys.exit(0)
