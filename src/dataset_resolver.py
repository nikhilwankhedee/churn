"""
Centralized dataset directory resolution.

No dataset adapter should ever hardcode filesystem paths.
This module provides the single authoritative mechanism for resolving
where dataset files live, used by discovery, validation, pipeline,
benchmark, and all public APIs.

Resolution order:
  1. Explicit data_dir parameter (highest priority)
  2. Discovery engine result
  3. Registered dataset config
  4. Environment detection (Kaggle/Colab/local)
  5. Configured DATA_DIR fallback
"""
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.utils import get_logger

logger = get_logger(__name__)


def resolve_dataset_directory(
    dataset_name: str,
    data_dir: Optional[str] = None,
    required_files: Optional[List[str]] = None,
) -> str:
    """Resolve the directory containing raw data files for a dataset.

    Parameters
    ----------
    dataset_name : str
        Canonical dataset name (e.g. 'olist', 'retailrocket').
    data_dir : str, optional
        Explicit directory override. Takes highest priority.
    required_files : list of str, optional
        Files that must exist in the resolved directory. If provided,
        validates the directory contains them and raises a detailed error.

    Returns
    -------
    str
        Absolute path to the dataset directory.

    Raises
    ------
    FileNotFoundError
        If the resolved directory does not exist or is missing required files.
    """
    resolved = None

    # 1. Explicit override — highest priority
    if data_dir is not None:
        p = Path(data_dir)
        if p.is_dir():
            resolved = str(p)
            logger.info(
                "Dataset '%s' resolved via explicit override: %s",
                dataset_name, resolved,
            )
        else:
            logger.warning(
                "Explicit data_dir for '%s' does not exist: %s — "
                "falling back to auto-detection",
                dataset_name, data_dir,
            )

    # 2. Try to find via discovery engine
    if resolved is None:
        resolved = _discover_dataset_directory(dataset_name)

    # 3. Try environment detection
    if resolved is None:
        resolved = _detect_from_environment(dataset_name)

    # 4. Check manifest root_directory
    if resolved is None:
        resolved = _get_manifest_root_directory(dataset_name)

    # 4b. Builtin data fallback (data/builtin/) — for smoke tests and runs
    # without external data downloads.
    if resolved is None:
        try:
            from src.config import BUILTIN_DATA_DIR
            builtin_dir = Path(BUILTIN_DATA_DIR)
            if builtin_dir.is_dir():
                if required_files:
                    present = {
                        f.name
                        for f in builtin_dir.glob("*")
                        if f.is_file()
                    }
                    alternates = {}
                    try:
                        from src.datasets import get_dataset
                        alternates = get_dataset(dataset_name).alternate_filenames
                    except Exception:
                        alternates = {}
                    have_all = all(
                        any(
                            p == f or p.endswith(f)
                            or p == alt or p.endswith(alt)
                            for p in present
                            for alt in (alternates.get(f) or [f])
                        )
                        for f in required_files
                    )
                else:
                    have_all = True
                if have_all:
                    resolved = str(builtin_dir)
                    logger.info(
                        "Dataset '%s' resolved via builtin data: %s",
                        dataset_name, resolved,
                    )
        except Exception:
            pass

    # 5. Fall back to configured DATA_DIR
    if resolved is None:
        from src.config import DATA_DIR
        if DATA_DIR and Path(DATA_DIR).is_dir():
            resolved = DATA_DIR

    # 5. Final validation
    if resolved is None or not Path(resolved).is_dir():
        raise FileNotFoundError(
            _build_not_found_message(dataset_name, data_dir, required_files)
        )

    # Validate required files if provided
    if required_files:
        _validate_required_files(resolved, dataset_name, required_files)

    return resolved


def _discover_dataset_directory(dataset_name: str) -> Optional[str]:
    """Use the discovery engine to find a dataset directory."""
    try:
        from src.discovery.scanner import scan_directory
        from src.discovery.detectors import get_all_detectors
        from src.registry_store import get_registry_store

        # Try common roots
        roots = _get_scan_roots()
        registry = get_registry_store()
        registry.ensure_synced()
        schema_info = registry.get_schema_info()

        for root in roots:
            if not Path(root).is_dir():
                continue
            try:
                scan_result = scan_directory(
                    root=Path(root),
                    max_depth=3,
                    previously_registered=schema_info,
                )
                for discovered in scan_result.high_confidence:
                    if discovered.dataset_name == dataset_name:
                        dir_path = str(discovered.directory)
                        logger.info(
                            "Dataset '%s' found via discovery: %s",
                            dataset_name, dir_path,
                        )
                        return dir_path
            except Exception:
                continue
    except Exception as exc:
        logger.debug("Discovery search failed for '%s': %s", dataset_name, exc)
    return None


def _detect_from_environment(dataset_name: str) -> Optional[str]:
    """Try to locate dataset using environment-specific paths."""
    try:
        from src.environment import detect_environment
        env = detect_environment()

        if env.dataset_root is not None:
            root = Path(env.dataset_root)
            if root.is_dir():
                # Check if dataset exists as subdirectory
                for candidate in [
                    root / dataset_name,
                    root / f"{dataset_name}_dataset",
                    root,
                ]:
                    if candidate.is_dir() and any(candidate.glob("*.csv")):
                        return str(candidate)
    except Exception:
        pass
    return None


def _get_manifest_root_directory(dataset_name: str) -> Optional[str]:
    """Check if the manifest defines a root_directory for this dataset."""
    try:
        import yaml

        # Check configs/datasets/
        from src.config import get_configs_dir
        configs_dir = get_configs_dir()
        manifest_path = configs_dir / "datasets" / f"{dataset_name}.yaml"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f) or {}
            root = manifest.get("root_directory")
            if root and Path(root).is_dir():
                logger.info(
                    "Dataset '%s' resolved via manifest root_directory: %s",
                    dataset_name, root,
                )
                return str(root)

        # Check .dataset_registry/manifests/
        from src.config import PROJECT_ROOT
        registry_path = Path(PROJECT_ROOT) / ".dataset_registry" / "manifests" / f"{dataset_name}.yaml"
        if registry_path.exists():
            with open(registry_path) as f:
                manifest = yaml.safe_load(f) or {}
            root = manifest.get("root_directory")
            if root and Path(root).is_dir():
                return str(root)
    except Exception:
        pass
    return None


def _get_scan_roots() -> List[str]:
    """Get common directories to scan for datasets."""
    roots = []

    # Environment-specific roots
    try:
        from src.environment import detect_environment
        env = detect_environment()
        if env.dataset_root is not None:
            roots.append(str(env.dataset_root))
        if env.working_dir is not None:
            roots.append(str(env.working_dir))
    except Exception:
        pass

    # Standard Kaggle paths
    for path in ["/kaggle/input", "/kaggle/working"]:
        if os.path.exists(path):
            roots.append(path)

    # Colab paths
    for path in ["/content/datasets", "/content/data", "/content"]:
        if os.path.exists(path):
            roots.append(path)

    # Project-local data directory
    try:
        from src.config import PROJECT_ROOT
        roots.append(os.path.join(PROJECT_ROOT, "data"))
    except Exception:
        pass

    return roots


def _validate_required_files(
    directory: str,
    dataset_name: str,
    required_files: List[str],
) -> None:
    """Check that required files exist and raise detailed error if not."""
    dir_path = Path(directory)
    found_files = {f.name for f in dir_path.iterdir() if f.is_file()}

    alternates = {}
    try:
        from src.datasets import get_dataset
        alternates = get_dataset(dataset_name).alternate_filenames
    except Exception:
        alternates = {}

    def _is_present(f: str) -> bool:
        names = [f] + list(alternates.get(f) or [])
        return any(
            found == n or found.endswith(n)
            for found in found_files
            for n in names
        )

    missing = [f for f in required_files if not _is_present(f)]

    if missing:
        raise FileNotFoundError(
            _build_missing_files_message(
                dataset_name, directory, required_files, found_files, missing,
                alternates=alternates,
            )
        )


def _build_not_found_message(
    dataset_name: str,
    explicit_dir: Optional[str],
    required_files: Optional[List[str]],
) -> str:
    """Build a detailed error message when dataset directory cannot be found."""
    lines = [
        f"Dataset '{dataset_name}': no data directory found.",
        "",
    ]

    if explicit_dir:
        lines.append(f"  Explicit data_dir provided: {explicit_dir}")
        lines.append(f"    Exists: {Path(explicit_dir).exists()}")
        lines.append("")

    lines.append("Searched locations:")
    lines.append("  1. Explicit data_dir parameter")
    lines.append("  2. Discovery engine (scanned common roots)")
    lines.append("  3. Environment detection (Kaggle/Colab/local)")
    lines.append(f"  4. Configured DATA_DIR fallback")
    lines.append("")

    lines.append("To fix this, either:")
    lines.append("  - Pass data_dir explicitly: framework.run('olist', data_dir='/path/to/data')")
    lines.append("  - Use the discovery engine: framework.benchmark('/path/to/kaggle/input')")
    lines.append("  - Set the environment variable: CHURN_DATA_DIR=/path/to/data")

    if required_files:
        lines.append("")
        lines.append(f"Required files: {', '.join(required_files)}")

    return "\n".join(lines)


def _build_missing_files_message(
    dataset_name: str,
    directory: str,
    required_files: List[str],
    found_files: set,
    missing: List[str],
    alternates: Optional[dict] = None,
) -> str:
    """Build a detailed error message when required files are missing."""
    alternates = alternates or {}
    lines = [
        f"Dataset: {dataset_name}",
        "",
        f"Resolved directory: {directory}",
        "",
        "Required files:",
    ]

    for f in required_files:
        names = [f] + list(alternates.get(f) or [])
        found = any(
            found == n or found.endswith(n)
            for found in found_files
            for n in names
        )
        marker = "\u2713" if found else "\u2717"
        lines.append(f"  {marker} {f}")

    lines.append("")
    if found_files:
        lines.append(f"Files found in directory: {sorted(found_files)}")
    else:
        lines.append("No files found in directory.")
    lines.append("")
    lines.append(f"Missing: {', '.join(missing)}")

    return "\n".join(lines)
