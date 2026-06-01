"""
Platform-independent path resolution for the Churn Research Framework.

All filesystem operations use pathlib.Path. No hardcoded OS-specific paths.
Supports Kaggle, Google Colab, Windows, macOS, and Linux environments.
"""
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional, Union


PathLike = Union[str, Path]


def resolve(path: PathLike) -> Path:
    """Resolve a path to an absolute Path object.

    Handles relative paths, ~ expansion, and environment variables.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


def safe_resolve(path: PathLike) -> Optional[Path]:
    """Resolve a path if it exists, otherwise return None."""
    p = resolve(path)
    return p if p.exists() else None


def ensure_dir(path: PathLike) -> Path:
    """Ensure a directory exists, creating it if necessary. Returns the Path."""
    p = resolve(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def relative_to(path: PathLike, base: PathLike) -> Optional[Path]:
    """Return path relative to base, or None if not relative."""
    try:
        return resolve(path).relative_to(resolve(base))
    except ValueError:
        return None


def is_subpath(path: PathLike, base: PathLike) -> bool:
    """Check if path is within (or equal to) base directory."""
    try:
        resolve(path).relative_to(resolve(base))
        return True
    except ValueError:
        return False


def find_csv_files(directory: PathLike, recursive: bool = True) -> list[Path]:
    """Find all CSV files in a directory.

    Parameters
    ----------
    directory : path-like
        Directory to search.
    recursive : bool
        If True, search subdirectories too.

    Returns
    -------
    list of Path objects for found CSV files, sorted by name.
    """
    d = resolve(directory)
    if not d.is_dir():
        return []
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(d.glob(pattern))


def list_immediate_subdirs(directory: PathLike) -> list[Path]:
    """List immediate subdirectories of the given path.

    Parameters
    ----------
    directory : path-like
        Directory to inspect.

    Returns
    -------
    list of Path objects for immediate subdirectories, sorted by name.
    """
    d = resolve(directory)
    if not d.is_dir():
        return []
    return sorted(
        entry for entry in d.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )


def get_filename(path: PathLike) -> str:
    """Get the filename (without extension) from a path."""
    return Path(path).stem


def get_extension(path: PathLike) -> str:
    """Get the file extension (including dot) from a path."""
    return Path(path).suffix


def normalize_path_for_display(path: PathLike) -> str:
    """Normalize a path for consistent display across platforms."""
    p = resolve(path)
    return str(p).replace("\\", "/")


def output_directory(
    dataset_root: Optional[PathLike] = None,
    output_dir: Optional[PathLike] = None,
) -> Path:
    """Determine the output directory.

    Priority:
    1. Explicit output_dir if provided
    2. Kaggle working dir if on Kaggle
    3. ./outputs relative to project root

    Parameters
    ----------
    dataset_root : path-like, optional
        The dataset root (used to detect Kaggle environment).
    output_dir : path-like, optional
        Explicit output directory override.

    Returns
    -------
    Path to the output directory.
    """
    if output_dir is not None:
        return ensure_dir(output_dir)

    from src.environment import detect_environment
    env = detect_environment(dataset_root)

    if env.is_kaggle:
        return ensure_dir(env.working_dir / "outputs")

    from src.config import PROJECT_ROOT
    return ensure_dir(Path(PROJECT_ROOT) / "outputs")
