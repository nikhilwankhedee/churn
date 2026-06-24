"""Manifest YAML validation for the dataset registration system.

Provides schema validation for dataset manifests to catch configuration
errors before they cause runtime failures during dataset loading.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

log = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """A single validation issue found in a manifest."""
    path: str
    message: str
    severity: str = "error"  # "error" or "warning"


@dataclass
class ValidationResult:
    """Result of manifest validation."""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def summary(self) -> str:
        parts = []
        if self.valid:
            parts.append("Manifest is valid")
        else:
            parts.append(f"Manifest has {self.error_count} error(s)")
        if self.warning_count:
            parts.append(f"{self.warning_count} warning(s)")
        return ", ".join(parts)

    def report(self) -> str:
        lines = [self.summary(), ""]
        for err in self.errors:
            lines.append(f"  ERROR   [{err.path}] {err.message}")
        for warn in self.warnings:
            lines.append(f"  WARNING [{warn.path}] {warn.message}")
        if not self.errors and not self.warnings:
            lines.append("  All checks passed.")
        return "\n".join(lines)


REQUIRED_TOP_LEVEL = set()
REQUIRED_DATASET_FIELDS = {"name"}
VALID_ADAPTER_TYPES = {"generic", "olist", "telco", "rees46", "retailrocket", "online_retail_ii", "instacart"}
VALID_SCHEMA_COLUMNS = {"customer_id", "event_time", "transaction_value"}
VALID_COMPUTE_OPS = {"multiply", "add", "subtract", "divide", "concat"}


def _check_type(data: Any, expected_type: type, path: str, results: ValidationResult) -> bool:
    """Check that a value has the expected type."""
    if not isinstance(data, expected_type):
        results.errors.append(ValidationError(
            path=path,
            message=f"Expected {expected_type.__name__}, got {type(data).__name__}",
        ))
        return False
    return True


def _check_required_fields(data: Dict[str, Any], fields: set, prefix: str, results: ValidationResult) -> None:
    """Check that required fields exist in a dictionary."""
    for f in fields:
        if f not in data:
            results.valid = False
            results.errors.append(ValidationError(
                path=f"{prefix}.{f}" if prefix else f,
                message=f"Required field '{f}' is missing",
            ))


def validate_manifest_path(manifest_path: str) -> ValidationResult:
    """Validate a manifest YAML file on disk.

    Parameters
    ----------
    manifest_path : str
        Path to the manifest YAML file.

    Returns
    -------
    ValidationResult
        Validation results with errors and warnings.
    """
    results = ValidationResult(valid=True)
    path = Path(manifest_path)

    if not path.exists():
        results.valid = False
        results.errors.append(ValidationError(
            path=str(path),
            message=f"File does not exist: {path}",
        ))
        return results

    if not path.suffix in (".yaml", ".yml"):
        results.warnings.append(ValidationError(
            path=str(path),
            message=f"File extension is not .yaml or .yml: {path.suffix}",
            severity="warning",
        ))

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        results.valid = False
        results.errors.append(ValidationError(
            path=str(path),
            message=f"Invalid YAML syntax: {e}",
        ))
        return results

    if data is None:
        results.valid = False
        results.errors.append(ValidationError(
            path=str(path),
            message="File is empty",
        ))
        return results

    validate_manifest_dict(data, results=results)
    return results


def validate_manifest_dict(
    data: Dict[str, Any],
    prefix: str = "",
    results: Optional[ValidationResult] = None,
) -> ValidationResult:
    """Validate a manifest dictionary structure.

    Parameters
    ----------
    data : dict
        The manifest dictionary to validate.
    prefix : str
        Prefix for error paths (for nested validation).
    results : ValidationResult, optional
        Results object to append to. Created if None.

    Returns
    -------
    ValidationResult
        Validation results with errors and warnings.
    """
    if results is None:
        results = ValidationResult(valid=True)

    if not _check_type(data, dict, prefix or "manifest", results):
        return results

    # dataset.name (nested under dataset section)
    if "dataset" in data:
        dataset = data["dataset"]
        if _check_type(dataset, dict, f"{prefix}.dataset", results):
            _check_required_fields(dataset, REQUIRED_DATASET_FIELDS, f"{prefix}.dataset", results)
            if "name" in dataset:
                name = dataset["name"]
                if _check_type(name, str, f"{prefix}.dataset.name", results):
                    if not name.strip():
                        results.errors.append(ValidationError(
                            path=f"{prefix}.dataset.name",
                            message="Name must not be empty",
                        ))
                    elif " " in name:
                        results.warnings.append(ValidationError(
                            path=f"{prefix}.dataset.name",
                            message=f"Name contains spaces: '{name}' — may cause issues",
                            severity="warning",
                        ))
    else:
        results.valid = False
        results.errors.append(ValidationError(
            path=f"{prefix}.dataset",
            message="Required section 'dataset' is missing",
        ))

    # schema.version
    if "schema" in data:
        schema = data["schema"]
        if _check_type(schema, dict, f"{prefix}.schema", results):
            ver = schema.get("version")
            if ver is not None and ver != 2:
                results.warnings.append(ValidationError(
                    path=f"{prefix}.schema.version",
                    message=f"Schema version {ver} is not the current version (2)",
                    severity="warning",
                ))

    # adapter section
    if "adapter" in data:
        adapter = data["adapter"]
        if _check_type(adapter, dict, f"{prefix}.adapter", results):
            adapter_type = adapter.get("type")
            if adapter_type is not None:
                if adapter_type not in VALID_ADAPTER_TYPES:
                    results.warnings.append(ValidationError(
                        path=f"{prefix}.adapter.type",
                        message=f"Unknown adapter type '{adapter_type}' — will use GenericDatasetAdapter",
                        severity="warning",
                    ))

    # files section
    if "files" in data:
        files = data["files"]
        if _check_type(files, dict, f"{prefix}.files", results):
            for section in ("required", "optional"):
                if section in files:
                    file_entries = files[section]
                    if not _check_type(file_entries, (dict, list), f"{prefix}.files.{section}", results):
                        continue
                    if isinstance(file_entries, dict):
                        for key, val in file_entries.items():
                            if not _check_type(val, str, f"{prefix}.files.{section}.{key}", results):
                                continue
                    elif isinstance(file_entries, list):
                        for i, val in enumerate(file_entries):
                            if not _check_type(val, str, f"{prefix}.files.{section}[{i}]", results):
                                continue

    # root_directory
    if "root_directory" in data:
        rd = data["root_directory"]
        if _check_type(rd, str, f"{prefix}.root_directory", results):
            if not rd.strip():
                results.warnings.append(ValidationError(
                    path=f"{prefix}.root_directory",
                    message="root_directory is empty",
                    severity="warning",
                ))

    # schema.columns
    if "schema" in data and isinstance(data["schema"], dict):
        schema = data["schema"]
        if "columns" in schema:
            columns = schema["columns"]
            if _check_type(columns, dict, f"{prefix}.schema.columns", results):
                missing_required = VALID_SCHEMA_COLUMNS - set(columns.keys())
                if missing_required:
                    results.warnings.append(ValidationError(
                        path=f"{prefix}.schema.columns",
                        message=f"Missing recommended columns: {', '.join(sorted(missing_required))}",
                        severity="warning",
                    ))

    # computed_columns
    if "computed_columns" in data:
        cc = data["computed_columns"]
        if _check_type(cc, dict, f"{prefix}.computed_columns", results):
            for col_name, col_def in cc.items():
                col_path = f"{prefix}.computed_columns.{col_name}"
                if isinstance(col_def, dict):
                    op = col_def.get("operation")
                    if op and op not in VALID_COMPUTE_OPS:
                        results.warnings.append(ValidationError(
                            path=f"{col_path}.operation",
                            message=f"Unknown operation '{op}' — supported: {', '.join(sorted(VALID_COMPUTE_OPS))}",
                            severity="warning",
                        ))
                    if "fields" not in col_def and "value" not in col_def:
                        results.errors.append(ValidationError(
                            path=col_path,
                            message="computed_column must have either 'fields' or 'value'",
                        ))

    # preprocessing
    if "preprocessing" in data:
        pp = data["preprocessing"]
        if _check_type(pp, dict, f"{prefix}.preprocessing", results):
            if "timestamp_format" in pp:
                fmt = pp["timestamp_format"]
                if not _check_type(fmt, str, f"{prefix}.preprocessing.timestamp_format", results):
                    pass
                elif fmt not in ("unix", "iso8601", "excel", "mixed", "infer"):
                    results.warnings.append(ValidationError(
                        path=f"{prefix}.preprocessing.timestamp_format",
                        message=f"Unknown timestamp_format '{fmt}'",
                        severity="warning",
                    ))

    return results
