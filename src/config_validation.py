"""
Configuration validator: validates YAML configs before pipeline execution.

Catches user mistakes early with clear error messages instead of
producing Python tracebacks during pipeline execution.
"""
import dataclasses
import os
from typing import Any, Dict, List, Optional, Set

import yaml

from src.utils import get_logger

logger = get_logger(__name__)

# ── Known valid values ────────────────────────────────────────────
VALID_ECOSYSTEM_TYPES = {
    "transactional_marketplace",
    "clickstream_commerce",
    "habitual_retail",
    "subscription",
}

VALID_CHURN_STRATEGIES = {"inactivity", "subscription", "cadence"}

VALID_FEATURE_GROUPS = {
    "purchase", "monetary", "inactivity", "review",
    "delivery", "payment", "engagement", "cadence",
}

VALID_MODEL_NAMES = {"logistic_regression", "random_forest", "xgboost"}

VALID_METRIC_NAMES = {
    "accuracy", "precision", "recall", "f1",
    "roc_auc", "pr_auc", "brier_score", "expected_calibration_error",
}

VALID_RESAMPLER_NAMES = {"smote", "adasyn", "none"}

# ── Standardized schema columns ───────────────────────────────────
REQUIRED_SCHEMA_COLUMNS = {"customer_id", "event_time"}
OPTIONAL_SCHEMA_COLUMNS = {
    "transaction_value", "event_type", "product_id",
    "review_score", "payment_type", "delivery_delay",
    "engagement_signal", "session_id",
}


@dataclasses.dataclass
class ValidationError:
    """A single validation error."""
    path: str  # dot-notation path to the error
    message: str
    severity: str = "error"  # "error" or "warning"


@dataclasses.dataclass
class ValidationResult:
    """Complete validation result for a config."""
    is_valid: bool
    errors: List[ValidationError] = dataclasses.field(default_factory=list)
    warnings: List[ValidationError] = dataclasses.field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def summary(self) -> str:
        """Human-readable summary."""
        if self.is_valid:
            return f"Config is valid ({self.warning_count} warnings)"
        lines = [f"Config has {self.error_count} error(s) and {self.warning_count} warning(s):"]
        for err in self.errors:
            lines.append(f"  [ERROR] {err.path}: {err.message}")
        for warn in self.warnings:
            lines.append(f"  [WARN]  {warn.path}: {warn.message}")
        return "\n".join(lines)


def _add_error(result: ValidationResult, path: str, message: str) -> None:
    result.errors.append(ValidationError(path=path, message=message, severity="error"))
    result.is_valid = False


def _add_warning(result: ValidationResult, path: str, message: str) -> None:
    result.warnings.append(ValidationError(path=path, message=message, severity="warning"))


def validate_config(
    config: Dict[str, Any],
    available_datasets: Optional[Set[str]] = None,
    config_path: Optional[str] = None,
) -> ValidationResult:
    """Validate a dataset configuration.

    Parameters
    ----------
    config : dict
        The parsed YAML configuration.
    available_datasets : set, optional
        Names of registered datasets (for cross-reference validation).
    config_path : str, optional
        Path to the config file (for error messages).

    Returns
    -------
    ValidationResult with errors and warnings.
    """
    result = ValidationResult(is_valid=True)
    loc = config_path or "config"

    # ── Top-level structure ────────────────────────────────────────
    if not isinstance(config, dict):
        _add_error(result, loc, "Config must be a YAML mapping (dict)")
        return result

    # ── dataset section ────────────────────────────────────────────
    ds = config.get("dataset")
    if ds is None:
        _add_error(result, f"{loc}.dataset", "Missing required 'dataset' section")
    elif isinstance(ds, dict):
        if not ds.get("name"):
            _add_error(result, f"{loc}.dataset.name", "Missing required 'dataset.name'")
        if not ds.get("ecosystem_type"):
            _add_warning(result, f"{loc}.dataset.ecosystem_type", "Missing 'dataset.ecosystem_type'")
        elif ds["ecosystem_type"] not in VALID_ECOSYSTEM_TYPES:
            _add_warning(
                result,
                f"{loc}.dataset.ecosystem_type",
                f"Unknown ecosystem type '{ds['ecosystem_type']}'. "
                f"Valid types: {sorted(VALID_ECOSYSTEM_TYPES)}",
            )

    # ── churn section ──────────────────────────────────────────────
    churn = config.get("churn")
    if churn is None:
        _add_warning(result, f"{loc}.churn", "Missing 'churn' section — using defaults")
    elif isinstance(churn, dict):
        strategy = churn.get("strategy")
        if strategy and strategy not in VALID_CHURN_STRATEGIES:
            _add_error(
                result,
                f"{loc}.churn.strategy",
                f"Unknown churn strategy '{strategy}'. "
                f"Valid strategies: {sorted(VALID_CHURN_STRATEGIES)}",
            )

        window = churn.get("prediction_window_days")
        if window is not None:
            if not isinstance(window, (int, float)) or window <= 0:
                _add_error(
                    result,
                    f"{loc}.churn.prediction_window_days",
                    f"prediction_window_days must be a positive number, got {window!r}",
                )
            elif window < 30:
                _add_warning(
                    result,
                    f"{loc}.churn.prediction_window_days",
                    f"Very short prediction window ({window} days) — churn labels may be unreliable",
                )
            elif window > 365:
                _add_warning(
                    result,
                    f"{loc}.churn.prediction_window_days",
                    f"Very long prediction window ({window} days) — may reduce label quality",
                )

    # ── features section ───────────────────────────────────────────
    features = config.get("features")
    if features and isinstance(features, dict):
        groups = features.get("available_groups", [])
        if groups:
            invalid_groups = set(groups) - VALID_FEATURE_GROUPS
            if invalid_groups:
                _add_error(
                    result,
                    f"{loc}.features.available_groups",
                    f"Unknown feature groups: {sorted(invalid_groups)}. "
                    f"Valid groups: {sorted(VALID_FEATURE_GROUPS)}",
                )

    # ── schema section ─────────────────────────────────────────────
    schema = config.get("schema")
    if schema and isinstance(schema, dict):
        mapping = schema.get("column_mapping", {})
        if mapping:
            mapped_standard_cols = set(mapping.values())
            # Check that required columns are mapped
            missing_required = REQUIRED_SCHEMA_COLUMNS - mapped_standard_cols
            if missing_required:
                _add_warning(
                    result,
                    f"{loc}.schema.column_mapping",
                    f"Required standardized columns not mapped: {sorted(missing_required)}. "
                    "Pipeline may fail if these columns are not present in the data.",
                )
            # Check for invalid standard column names
            all_valid = REQUIRED_SCHEMA_COLUMNS | OPTIONAL_SCHEMA_COLUMNS
            invalid_cols = mapped_standard_cols - all_valid
            if invalid_cols:
                _add_warning(
                    result,
                    f"{loc}.schema.column_mapping",
                    f"Non-standard column names in mapping: {sorted(invalid_cols)}. "
                    f"Valid columns: {sorted(all_valid)}",
                )

    # ── preprocessing section ──────────────────────────────────────
    preproc = config.get("preprocessing")
    if preproc and isinstance(preproc, dict):
        ts_cols = preproc.get("timestamp_columns", [])
        if ts_cols and not isinstance(ts_cols, list):
            _add_error(
                result,
                f"{loc}.preprocessing.timestamp_columns",
                "timestamp_columns must be a list",
            )

        drop_null = preproc.get("drop_null_timestamp")
        if drop_null and ts_cols and drop_null not in ts_cols:
            _add_warning(
                result,
                f"{loc}.preprocessing.drop_null_timestamp",
                f"drop_null_timestamp '{drop_null}' is not in timestamp_columns",
            )

    # ── files section ──────────────────────────────────────────────
    files = config.get("files")
    if files and isinstance(files, dict):
        if config_path:
            config_dir = os.path.dirname(os.path.abspath(config_path))
            data_dir = os.path.join(config_dir, "..", "data")
            for key, filename in files.items():
                filepath = os.path.join(data_dir, filename)
                if not os.path.isfile(filepath):
                    _add_warning(
                        result,
                        f"{loc}.files.{key}",
                        f"File not found: {filename} (looked in {data_dir})",
                    )

    logger.info(
        "Config validation: valid=%s, errors=%d, warnings=%d",
        result.is_valid, result.error_count, result.warning_count,
    )

    return result


def validate_config_file(
    file_path: str,
    available_datasets: Optional[Set[str]] = None,
) -> ValidationResult:
    """Validate a YAML config file.

    Parameters
    ----------
    file_path : str
        Path to the YAML config file.
    available_datasets : set, optional
        Names of registered datasets.

    Returns
    -------
    ValidationResult.
    """
    if not os.path.isfile(file_path):
        result = ValidationResult(is_valid=False)
        _add_error(result, file_path, f"File not found: {file_path}")
        return result

    try:
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        result = ValidationResult(is_valid=False)
        _add_error(result, file_path, f"Invalid YAML: {e}")
        return result
    except Exception as e:
        result = ValidationResult(is_valid=False)
        _add_error(result, file_path, f"Could not read file: {e}")
        return result

    return validate_config(config, available_datasets=available_datasets, config_path=file_path)
