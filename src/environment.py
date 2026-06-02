"""
Environment detection for the Churn Research Framework.

Automatically detects execution environment (Kaggle, Colab, local)
and provides sensible default paths. Users can always override defaults.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EnvironmentInfo:
    """Detected execution environment."""
    name: str
    is_kaggle: bool = False
    is_colab: bool = False
    is_local: bool = True
    dataset_root: Optional[Path] = None
    working_dir: Optional[Path] = None
    default_output_dir: Optional[Path] = None
    details: dict = field(default_factory=dict)


def _is_kaggle() -> bool:
    """Detect if running on Kaggle."""
    return (
        os.path.exists("/kaggle/input")
        or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
        or os.environ.get("KAGGLE_DATA_PROXY_URL") is not None
    )


def _is_colab() -> bool:
    """Detect if running on Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        pass
    return (
        os.environ.get("COLAB_RELEASE_TAG") is not None
        or os.environ.get("COLAB_GPU") is not None
        or "google.colab" in os.environ.get("_", "")
    )


def _kaggle_dataset_root() -> Optional[Path]:
    """Find the Kaggle input directory."""
    root = Path("/kaggle/input")
    if root.exists():
        return root
    return None


def _kaggle_working_dir() -> Optional[Path]:
    """Find the Kaggle working directory."""
    wd = Path("/kaggle/working")
    if wd.exists():
        return wd
    return Path("/kaggle/working")  # Return even if not existing yet


def _colab_dataset_root() -> Optional[Path]:
    """Find dataset root in Google Colab.

    Colab typically mounts Google Drive at /content/drive
    and datasets are in /content or subdirectories.
    """
    for candidate in [
        Path("/content/datasets"),
        Path("/content/data"),
        Path("/content"),
    ]:
        if candidate.exists():
            csv_files = list(candidate.glob("**/*.csv"))
            if csv_files:
                return candidate
    return Path.cwd()


def detect_environment(
    dataset_root: Optional[str] = None,
) -> EnvironmentInfo:
    """Detect the current execution environment.

    Parameters
    ----------
    dataset_root : str, optional
        Explicit dataset root override. If provided, takes precedence
        over auto-detected paths.

    Returns
    -------
    EnvironmentInfo with detected environment details.
    """
    info = EnvironmentInfo(name="unknown")

    if _is_kaggle():
        info.name = "kaggle"
        info.is_kaggle = True
        info.is_local = False
        info.working_dir = _kaggle_working_dir()
        info.default_output_dir = info.working_dir / "outputs" if info.working_dir else None
        info.dataset_root = _kaggle_dataset_root()
        info.details["kaggle_input"] = str(info.dataset_root)
        info.details["kaggle_working"] = str(info.working_dir)

    elif _is_colab():
        info.name = "colab"
        info.is_colab = True
        info.is_local = False
        info.working_dir = Path.cwd()
        info.default_output_dir = info.working_dir / "outputs"
        info.dataset_root = _colab_dataset_root()
        info.details["colab_cwd"] = str(info.working_dir)

    else:
        info.name = "local"
        info.is_local = True
        info.working_dir = Path.cwd()
        info.default_output_dir = info.working_dir / "outputs"
        info.dataset_root = None  # Must be provided by user

    # Override dataset_root if explicitly provided
    if dataset_root is not None:
        info.dataset_root = Path(dataset_root)
        info.details["explicit_dataset_root"] = str(info.dataset_root)

    return info


def get_default_dataset_root() -> Optional[str]:
    """Get the default dataset root for the current environment.

    Returns None if no default can be determined (local environment).
    """
    env = detect_environment()
    if env.dataset_root is not None:
        return str(env.dataset_root)
    return None


def get_default_output_dir() -> str:
    """Get the default output directory for the current environment."""
    env = detect_environment()
    if env.default_output_dir is not None:
        return str(env.default_output_dir)
    return str(Path.cwd() / "outputs")
