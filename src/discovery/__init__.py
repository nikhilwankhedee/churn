"""
Dataset discovery and automatic detection.

Scans directories recursively, identifies built-in and previously registered
datasets by their file signatures (not folder names), and returns confidence
scores for each match.
"""
from src.discovery.detectors import (
    DatasetDetector,
    DetectionResult,
    get_all_detectors,
    detect_dataset,
)
from src.discovery.scanner import (
    scan_directory,
    ScanResult,
    DiscoveredDataset,
)

__all__ = [
    "DatasetDetector",
    "DetectionResult",
    "get_all_detectors",
    "detect_dataset",
    "scan_directory",
    "ScanResult",
    "DiscoveredDataset",
]
