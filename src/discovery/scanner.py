"""
Recursive dataset scanner — discovers and identifies datasets in directories.

Scans a dataset root recursively, runs all built-in detectors plus
previously registered dataset recognizers, and returns confidence-scored
matches.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.discovery.detectors import (
    DetectionResult,
    detect_dataset,
    get_all_detectors,
)
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class DiscoveredDataset:
    """A dataset discovered in a directory."""
    directory: Path
    detection: DetectionResult
    source: str = "builtin"
    registered_config: Optional[Path] = None

    @property
    def dataset_name(self) -> str:
        return self.detection.dataset_type

    @property
    def adapter_key(self) -> str:
        return self.detection.adapter_key or self.detection.dataset_type

    @property
    def confidence(self) -> float:
        return self.detection.confidence

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.8

    def summary(self) -> str:
        """Human-readable summary of this discovery."""
        status = "HIGH" if self.is_high_confidence else "LOW"
        files_str = ", ".join(self.detection.matched_files[:3])
        if len(self.detection.matched_files) > 3:
            files_str += f" +{len(self.detection.matched_files) - 3} more"
        missing = len(self.detection.missing_files)
        return (
            f"{self.dataset_name} ({self.confidence:.0%}) [{status}] — "
            f"files: [{files_str}]"
            + (f", missing: {missing}" if missing else "")
        )


@dataclass
class ScanResult:
    """Result of scanning a directory for datasets."""
    root: Path
    total_directories_scanned: int
    discovered: list[DiscoveredDataset] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def high_confidence(self) -> list[DiscoveredDataset]:
        """Datasets matched with high confidence (>= 80%)."""
        return [d for d in self.discovered if d.is_high_confidence]

    @property
    def low_confidence(self) -> list[DiscoveredDataset]:
        """Datasets matched with low confidence (< 80%)."""
        return [d for d in self.discovered if not d.is_high_confidence]

    @property
    def by_name(self) -> dict[str, DiscoveredDataset]:
        """Deduplicated by dataset name, keeping highest confidence match."""
        best: dict[str, DiscoveredDataset] = {}
        for d in self.discovered:
            name = d.dataset_name
            if name not in best or d.confidence > best[name].confidence:
                best[name] = d
        return best

    def summary(self) -> str:
        """Human-readable summary of scan results."""
        lines = [
            f"Scanned {self.total_directories_scanned} directories under {self.root}",
            f"Discovered {len(self.discovered)} dataset(s), "
            f"{len(self.high_confidence)} high-confidence",
            "",
        ]
        for d in self.high_confidence:
            lines.append(f"  ✓ {d.summary()}")
        if self.low_confidence:
            lines.append("")
            lines.append("Low-confidence matches:")
            for d in self.low_confidence:
                lines.append(f"  ? {d.summary()}")
        if self.errors:
            lines.append("")
            lines.append(f"Errors: {len(self.errors)}")
            for e in self.errors[:5]:
                lines.append(f"  - {e}")
        return "\n".join(lines)


def _collect_directories(root: Path, max_depth: int = 5) -> list[Path]:
    """Collect all directories under root up to max_depth.

    Skips hidden directories and common non-dataset directories.
    """
    skip_names = {
        ".git", "__pycache__", ".ipynb_checkpoints", "node_modules",
        ".venv", "venv", ".env", ".mypy_cache", ".pytest_cache",
        "outputs", "figures", "results", "models", "processed_data",
        "notebooks", "docs", ".opencode",
    }

    directories = [root]
    _collect_recursive(root, directories, 0, max_depth, skip_names)
    return directories


def _collect_recursive(
    current: Path,
    collected: list[Path],
    depth: int,
    max_depth: int,
    skip_names: set[str],
) -> None:
    """Recursively collect directories."""
    if depth >= max_depth:
        return
    try:
        for entry in current.iterdir():
            if (
                entry.is_dir()
                and not entry.name.startswith(".")
                and entry.name not in skip_names
            ):
                collected.append(entry)
                _collect_recursive(entry, collected, depth + 1, max_depth, skip_names)
    except PermissionError:
        pass


def scan_directory(
    root: str | Path,
    max_depth: int = 5,
    detectors: Optional[list] = None,
    previously_registered: Optional[dict[str, dict]] = None,
) -> ScanResult:
    """Recursively scan a directory for datasets.

    Parameters
    ----------
    root : str or Path
        Root directory to scan (e.g. '/kaggle/input' or 'datasets').
    max_depth : int
        Maximum directory depth to scan.
    detectors : list of DatasetDetector, optional
        Custom detectors. Default: all built-in detectors.
    previously_registered : dict, optional
        Previously registered dataset schemas for recognition.
        Maps dataset_name -> {schema_info}.

    Returns
    -------
    ScanResult with all discovered datasets.
    """
    root = Path(root)
    result = ScanResult(root=root, total_directories_scanned=0)

    if not root.is_dir():
        result.errors.append(f"Root directory does not exist: {root}")
        return result

    directories = _collect_directories(root, max_depth)
    result.total_directories_scanned = len(directories)

    logger.info(
        "Scanning %d directories under %s (max_depth=%d)",
        len(directories), root, max_depth,
    )

    seen_directories: set[Path] = set()

    for directory in directories:
        # Skip if already covered by a parent's detection
        try:
            if any(
                directory != d and directory.is_relative_to(d)
                for d in seen_directories
            ):
                continue
        except (ValueError, OSError):
            continue

        # Run built-in detectors
        try:
            detection_results = detect_dataset(directory, detectors)
        except Exception as exc:
            result.errors.append(f"Detection failed for {directory}: {exc}")
            continue

        for detection in detection_results:
            if detection.matched and detection.confidence >= 0.5:
                discovered = DiscoveredDataset(
                    directory=directory,
                    detection=detection,
                    source="builtin",
                )
                result.discovered.append(discovered)
                seen_directories.add(directory)
                logger.info(
                    "Detected %s in %s (confidence: %.0f%%)",
                    detection.dataset_type, directory, detection.confidence * 100,
                )
                break  # Only one match per directory

        # Check against previously registered datasets
        if previously_registered and directory not in seen_directories:
            _check_registered(
                directory, previously_registered, result
            )

    # Deduplicate by name, keeping highest confidence
    result.discovered = _deduplicate(result.discovered)

    return result


def _check_registered(
    directory: Path,
    previously_registered: dict[str, dict],
    result: ScanResult,
) -> None:
    """Check if a directory matches any previously registered dataset schema."""
    csv_files = list(directory.glob("*.csv"))
    if not csv_files:
        return

    try:
        import pandas as pd
    except ImportError:
        return

    # Build a column signature from CSVs in this directory
    dir_columns: dict[str, set[str]] = {}
    for csv in csv_files[:10]:  # Limit to first 10 CSVs
        try:
            df = pd.read_csv(csv, nrows=5)
            dir_columns[csv.name] = set(df.columns)
        except Exception:
            continue

    if not dir_columns:
        return

    for dataset_name, schema_info in previously_registered.items():
        required_columns = schema_info.get("required_columns", {})
        if not required_columns:
            continue

        matched_tables = 0
        total_tables = len(required_columns)

        for filename, cols in required_columns.items():
            # Check if any CSV in directory has these columns
            cols_set = set(cols)
            for dir_file, dir_cols in dir_columns.items():
                overlap = len(cols_set & dir_cols)
                if overlap >= len(cols_set) * 0.7:  # 70% column overlap
                    matched_tables += 1
                    break

        if total_tables > 0:
            confidence = matched_tables / total_tables
            if confidence >= 0.8:
                detection = DetectionResult(
                    matched=True,
                    dataset_type=dataset_name,
                    confidence=confidence,
                    adapter_key=dataset_name,
                    matched_files=list(dir_columns.keys()),
                    details=f"Schema match against registered '{dataset_name}'",
                )
                result.discovered.append(DiscoveredDataset(
                    directory=directory,
                    detection=detection,
                    source="registered",
                ))


def _deduplicate(discovered: list[DiscoveredDataset]) -> list[DiscoveredDataset]:
    """Deduplicate by dataset name, keeping highest confidence match."""
    best: dict[str, DiscoveredDataset] = {}
    for d in discovered:
        name = d.dataset_name
        if name not in best or d.confidence > best[name].confidence:
            best[name] = d
    return sorted(best.values(), key=lambda d: d.confidence, reverse=True)
