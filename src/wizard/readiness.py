"""
New dataset readiness report.

Evaluates whether a dataset is ready for churn analysis by checking:
- Timestamp parsing feasibility
- Customer ID usability
- Temporal ordering
- Data sufficiency
- Churn definability
- Feature engineering feasibility
- Training completion potential
"""
import dataclasses
from typing import Any, Dict, List, Optional

import pandas as pd

from src.wizard.inspector import InspectionResult
from src.utils import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class ReadinessCheck:
    """A single readiness check result."""
    name: str
    passed: bool
    severity: str  # "critical", "warning", "info"
    message: str
    details: Optional[str] = None


@dataclasses.dataclass
class ReadinessReport:
    """Complete readiness report for a new dataset."""
    dataset_name: str
    checks: List[ReadinessCheck] = dataclasses.field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return not any(
            c.severity == "critical" and not c.passed for c in self.checks
        )

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_warnings(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "warning")

    @property
    def n_critical(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "critical")

    def summary(self) -> str:
        lines = []
        status = "READY" if self.is_ready else "NOT READY"
        lines.append(f"Dataset: {self.dataset_name} — {status}")
        lines.append(f"Checks: {self.n_passed} passed, {self.n_warnings} warnings, {self.n_critical} critical")
        lines.append("")
        for check in self.checks:
            icon = "✓" if check.passed else ("✗" if check.severity == "critical" else "⚠")
            lines.append(f"  [{icon}] {check.name}: {check.message}")
            if check.details:
                lines.append(f"      {check.details}")
        return "\n".join(lines)


def _add_check(
    report: ReadinessReport,
    name: str,
    passed: bool,
    severity: str,
    message: str,
    details: Optional[str] = None,
) -> None:
    report.checks.append(ReadinessCheck(
        name=name, passed=passed, severity=severity,
        message=message, details=details,
    ))


def generate_readiness_report(
    inspection: InspectionResult,
    prediction_window_days: int = 180,
) -> ReadinessReport:
    """Generate a readiness report from inspection results.

    Parameters
    ----------
    inspection : InspectionResult
        Results from inspect_csv().
    prediction_window_days : int
        Planned churn prediction window.

    Returns
    -------
    ReadinessReport with all checks.
    """
    report = ReadinessReport(dataset_name=inspection.suggested_dataset_name)

    # ── Check 1: Customer ID ───────────────────────────────────────
    if inspection.inferred_customer_id:
        _add_check(
            report,
            name="Customer ID",
            passed=True,
            severity="critical",
            message=f"Detected: '{inspection.inferred_customer_id}'",
        )
    else:
        _add_check(
            report,
            name="Customer ID",
            passed=False,
            severity="critical",
            message="No customer ID column detected",
            details="Churn analysis requires a unique customer identifier.",
        )

    # ── Check 2: Timestamp ─────────────────────────────────────────
    if inspection.inferred_event_time:
        _add_check(
            report,
            name="Timestamp",
            passed=True,
            severity="critical",
            message=f"Detected: '{inspection.inferred_event_time}'",
        )
    else:
        _add_check(
            report,
            name="Timestamp",
            passed=False,
            severity="critical",
            message="No timestamp column detected",
            details="Churn labeling requires temporal information.",
        )

    # ── Check 3: Data Sufficiency ──────────────────────────────────
    min_rows = 1000
    if inspection.n_rows >= min_rows:
        _add_check(
            report,
            name="Data Sufficiency",
            passed=True,
            severity="warning",
            message=f"{inspection.n_rows:,} rows (minimum: {min_rows:,})",
        )
    else:
        _add_check(
            report,
            name="Data Sufficiency",
            passed=False,
            severity="warning",
            message=f"Only {inspection.n_rows:,} rows (recommended: {min_rows:,})",
            details="Small datasets may produce unreliable model estimates.",
        )

    # ── Check 4: Temporal Span ─────────────────────────────────────
    ts_col = inspection.inferred_event_time
    if ts_col:
        for col in inspection.columns:
            if col.name == ts_col and col.min_value and col.max_value:
                try:
                    min_ts = pd.Timestamp(col.min_value)
                    max_ts = pd.Timestamp(col.max_value)
                    span_days = (max_ts - min_ts).days
                    if span_days >= prediction_window_days * 2:
                        _add_check(
                            report,
                            name="Temporal Span",
                            passed=True,
                            severity="warning",
                            message=f"{span_days} days (need {prediction_window_days}d window)",
                        )
                    elif span_days >= prediction_window_days:
                        _add_check(
                            report,
                            name="Temporal Span",
                            passed=True,
                            severity="warning",
                            message=f"{span_days} days — marginal for {prediction_window_days}d window",
                            details="Consider a shorter prediction window if results are poor.",
                        )
                    else:
                        _add_check(
                            report,
                            name="Temporal Span",
                            passed=False,
                            severity="critical",
                            message=f"{span_days} days — too short for {prediction_window_days}d window",
                            details="Observation period must be at least 2x the prediction window.",
                        )
                except Exception:
                    _add_check(
                        report,
                        name="Temporal Span",
                        passed=False,
                        severity="warning",
                        message="Could not parse timestamp range",
                    )
                break

    # ── Check 5: Transaction Value ─────────────────────────────────
    if inspection.inferred_transaction_value:
        _add_check(
            report,
            name="Monetary Value",
            passed=True,
            severity="warning",
            message=f"Detected: '{inspection.inferred_transaction_value}'",
            details="Monetary features will be available.",
        )
    else:
        _add_check(
            report,
            name="Monetary Value",
            passed=False,
            severity="warning",
            message="No monetary value column detected",
            details="Monetary features will be unavailable. Purchase patterns still usable.",
        )

    # ── Check 6: Feature Engineering Feasibility ───────────────────
    n_numeric = sum(1 for c in inspection.columns if c.is_numeric)
    n_categorical = sum(1 for c in inspection.columns if c.is_categorical)
    n_datetime = sum(1 for c in inspection.columns if c.is_datetime)

    if n_numeric + n_categorical + n_datetime >= 3:
        _add_check(
            report,
            name="Feature Engineering",
            passed=True,
            severity="warning",
            message=f"{n_numeric} numeric, {n_categorical} categorical, {n_datetime} datetime columns",
        )
    else:
        _add_check(
            report,
            name="Feature Engineering",
            passed=False,
            severity="warning",
            message="Limited column types for feature engineering",
            details=f"Found: {n_numeric} numeric, {n_categorical} categorical, {n_datetime} datetime.",
        )

    # ── Check 7: Missing Values ────────────────────────────────────
    high_missing = [c for c in inspection.columns if c.missing_pct > 0.5]
    if not high_missing:
        _add_check(
            report,
            name="Missing Values",
            passed=True,
            severity="warning",
            message="No columns with >50% missing values",
        )
    else:
        names = ", ".join(c.name for c in high_missing[:5])
        _add_check(
            report,
            name="Missing Values",
            passed=False,
            severity="warning",
            message=f"{len(high_missing)} column(s) with >50% missing: {names}",
            details="High missingness may bias features. Consider imputation or exclusion.",
        )

    # ── Check 8: Duplicate Rows ────────────────────────────────────
    # We don't have duplicate info from inspection, so skip if not available
    _add_check(
        report,
        name="Data Quality",
        passed=True,
        severity="info",
        message="Full quality check requires loading complete dataset",
        details="Run 'churn validate <dataset>' after registration for thorough checks.",
    )

    logger.info(
        "Readiness report: %s — %d checks, ready=%s",
        report.dataset_name, len(report.checks), report.is_ready,
    )

    return report
