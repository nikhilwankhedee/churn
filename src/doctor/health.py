"""
Dataset Doctor: comprehensive data health analyzer.

Inspects a dataset for quality issues and produces actionable recommendations.
Checks:
  - Duplicate customers
  - Duplicate transactions
  - Missing timestamps
  - Future timestamps
  - Invalid dates
  - Negative monetary values
  - Missing IDs
  - High missingness
  - Class imbalance
  - Data leakage
  - Duplicate columns
  - Constant features
  - Extreme outliers
  - Unsupported datatypes
  - Schema compliance
"""
import dataclasses
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class HealthCheck:
    """A single health check result."""
    name: str
    passed: bool
    severity: str  # "critical", "warning", "info"
    message: str
    recommendation: str = ""
    details: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class DoctorReport:
    """Complete health report from the Dataset Doctor."""
    dataset_name: str
    n_rows: int
    n_columns: int
    checks: List[HealthCheck] = dataclasses.field(default_factory=list)

    @property
    def overall_score(self) -> float:
        if not self.checks:
            return 0.0
        total_weight = 0.0
        passed_weight = 0.0
        for c in self.checks:
            w = {"critical": 3.0, "warning": 1.0, "info": 0.3}.get(c.severity, 1.0)
            total_weight += w
            if c.passed:
                passed_weight += w
        return (passed_weight / total_weight * 100) if total_weight > 0 else 100.0

    @property
    def n_critical(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "critical")

    @property
    def n_warnings(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "warning")

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    def summary(self) -> str:
        lines = [
            f"Dataset Doctor Report: {self.dataset_name}",
            f"Shape: {self.n_rows:,} rows x {self.n_columns} columns",
            f"Health Score: {self.overall_score:.0f}%",
            f"Checks: {self.n_passed} passed, {self.n_warnings} warnings, {self.n_critical} critical",
            "",
        ]
        for c in self.checks:
            icon = "✓" if c.passed else ("✗" if c.severity == "critical" else "⚠")
            lines.append(f"  [{icon}] {c.name}: {c.message}")
            if c.recommendation and not c.passed:
                lines.append(f"      → {c.recommendation}")
        return "\n".join(lines)


def _add_check(
    report: DoctorReport,
    name: str,
    passed: bool,
    severity: str,
    message: str,
    recommendation: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    report.checks.append(HealthCheck(
        name=name, passed=passed, severity=severity,
        message=message, recommendation=recommendation, details=details,
    ))


def run_doctor(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
    customer_id_col: str = "customer_id",
    timestamp_col: str = "event_time",
    monetary_col: Optional[str] = None,
    target_col: Optional[str] = None,
) -> DoctorReport:
    """Run the Dataset Doctor on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze.
    dataset_name : str
        Name of the dataset for reporting.
    customer_id_col : str
        Customer ID column name.
    timestamp_col : str
        Timestamp column name.
    monetary_col : str, optional
        Monetary value column name.
    target_col : str, optional
        Target/label column name.

    Returns
    -------
    DoctorReport with all findings.
    """
    report = DoctorReport(
        dataset_name=dataset_name,
        n_rows=len(df),
        n_columns=len(df.columns),
    )

    # ── Check 1: Duplicate rows ────────────────────────────────────
    n_dup = int(df.duplicated().sum())
    if n_dup == 0:
        _add_check(report, "Duplicate Rows", True, "warning",
                    "No duplicate rows found")
    else:
        pct = n_dup / len(df) * 100
        _add_check(report, "Duplicate Rows", False, "critical",
                    f"{n_dup:,} duplicate rows ({pct:.1f}%)",
                    f"Remove duplicates with df.drop_duplicates()")

    # ── Check 2: Duplicate customers ───────────────────────────────
    if customer_id_col in df.columns:
        n_cust = df[customer_id_col].nunique()
        n_rows = len(df)
        ratio = n_rows / max(n_cust, 1)
        if ratio > 10:
            _add_check(report, "Duplicate Customers", True, "info",
                        f"{n_cust:,} unique customers, {ratio:.1f} rows/customer (expected for transactional data)")
        else:
            _add_check(report, "Duplicate Customers", True, "info",
                        f"{n_cust:,} unique customers, {ratio:.1f} rows/customer")
    else:
        _add_check(report, "Duplicate Customers", False, "warning",
                    f"Column '{customer_id_col}' not found",
                    "Identify the customer identifier column")

    # ── Check 3: Duplicate transactions ────────────────────────────
    n_row_dup = int(df.duplicated().sum())
    if n_row_dup > len(df) * 0.01:
        _add_check(report, "Duplicate Transactions", False, "warning",
                    f"{n_row_dup:,} fully duplicated rows",
                    "Investigate if these are true duplicates or expected multi-row records")
    else:
        _add_check(report, "Duplicate Transactions", True, "warning",
                    "Duplicate transaction rate is acceptable")

    # ── Check 4: Missing timestamps ────────────────────────────────
    if timestamp_col in df.columns:
        n_missing_ts = int(df[timestamp_col].isnull().sum())
        if n_missing_ts == 0:
            _add_check(report, "Missing Timestamps", True, "critical",
                        "All timestamps present")
        else:
            pct = n_missing_ts / len(df) * 100
            _add_check(report, "Missing Timestamps", False, "critical",
                        f"{n_missing_ts:,} rows ({pct:.1f}%) missing timestamps",
                        "Drop rows with missing timestamps or impute from context")
    else:
        _add_check(report, "Missing Timestamps", False, "critical",
                    f"Timestamp column '{timestamp_col}' not found",
                    "Identify the timestamp column for temporal analysis")

    # ── Check 5: Future timestamps ─────────────────────────────────
    if timestamp_col in df.columns and df[timestamp_col].dtype == 'datetime64[ns]':
        now = pd.Timestamp.now()
        n_future = int((df[timestamp_col] > now).sum())
        if n_future == 0:
            _add_check(report, "Future Timestamps", True, "critical",
                        "No future timestamps detected")
        else:
            _add_check(report, "Future Timestamps", False, "critical",
                        f"{n_future:,} rows have timestamps in the future",
                        "Investigate data collection errors or timezone issues")

    # ── Check 6: Invalid dates ─────────────────────────────────────
    if timestamp_col in df.columns:
        ts = df[timestamp_col].dropna()
        if ts.dtype == object:
            try:
                pd.to_datetime(ts.head(100))
                _add_check(report, "Invalid Dates", True, "warning",
                            "Timestamps parse successfully")
            except Exception:
                _add_check(report, "Invalid Dates", False, "critical",
                            "Timestamps cannot be parsed as datetime",
                            "Fix date format or use pd.to_datetime with explicit format")
        elif pd.api.types.is_datetime64_any_dtype(ts):
            min_date = ts.min()
            if min_date.year < 1900:
                _add_check(report, "Invalid Dates", False, "warning",
                            f"Earliest date is {min_date} — unusually old",
                            "Verify date encoding is correct")
            else:
                _add_check(report, "Invalid Dates", True, "warning",
                            "Date range appears valid")

    # ── Check 7: Negative monetary values ──────────────────────────
    if monetary_col and monetary_col in df.columns:
        if pd.api.types.is_numeric_dtype(df[monetary_col]):
            n_neg = int((df[monetary_col] < 0).sum())
            if n_neg == 0:
                _add_check(report, "Negative Monetary Values", True, "warning",
                            "All monetary values are non-negative")
            else:
                _add_check(report, "Negative Monetary Values", False, "warning",
                            f"{n_neg:,} rows have negative values in '{monetary_col}'",
                            "Check if negatives represent refunds or data errors")
        else:
            _add_check(report, "Negative Monetary Values", True, "info",
                        f"'{monetary_col}' is not numeric — skipping check")
    else:
        _add_check(report, "Negative Monetary Values", True, "info",
                    "No monetary column specified for check")

    # ── Check 8: Missing IDs ───────────────────────────────────────
    id_cols = [c for c in df.columns if "id" in c.lower() or c == customer_id_col]
    for col in id_cols[:5]:
        n_miss = int(df[col].isnull().sum())
        if n_miss > 0:
            pct = n_miss / len(df) * 100
            _add_check(report, f"Missing IDs ({col})", False, "warning",
                        f"{n_miss:,} missing values ({pct:.1f}%)",
                        f"Impute or drop rows with missing '{col}'")
        else:
            _add_check(report, f"Missing IDs ({col})", True, "warning",
                        f"All values present in '{col}'")

    # ── Check 9: High missingness ──────────────────────────────────
    miss_pct = df.isnull().mean()
    high_miss = miss_pct[miss_pct > 0.5].sort_values(ascending=False)
    if len(high_miss) == 0:
        _add_check(report, "High Missingness", True, "warning",
                    "No columns with >50% missing values")
    else:
        cols_str = ", ".join(f"{c} ({v:.0%})" for c, v in high_miss.head(5).items())
        _add_check(report, "High Missingness", False, "warning",
                    f"{len(high_miss)} column(s) with >50% missing: {cols_str}",
                    "Consider dropping these columns or using advanced imputation")

    # ── Check 10: Class imbalance ──────────────────────────────────
    if target_col and target_col in df.columns:
        target = df[target_col].dropna()
        if target.nunique() <= 20:
            counts = target.value_counts()
            if len(counts) == 2:
                ratio = counts.min() / counts.max()
                if ratio < 0.1:
                    _add_check(report, "Class Imbalance", False, "warning",
                                f"Severe imbalance: {counts.to_dict()} (ratio: {ratio:.3f})",
                                "Use SMOTE, class weights, or stratified sampling")
                elif ratio < 0.3:
                    _add_check(report, "Class Imbalance", False, "info",
                                f"Moderate imbalance: {counts.to_dict()} (ratio: {ratio:.3f})",
                                "Consider balanced class weights")
                else:
                    _add_check(report, "Class Imbalance", True, "warning",
                                f"Acceptable balance: {counts.to_dict()} (ratio: {ratio:.3f})")
    else:
        _add_check(report, "Class Imbalance", True, "info",
                    "No target column specified for imbalance check")

    # ── Check 11: Data leakage heuristic ───────────────────────────
    if target_col and target_col in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns
        leakage_candidates = []
        for col in num_cols:
            if col == target_col:
                continue
            try:
                corr = abs(df[col].corr(df[target_col]))
                if corr > 0.95:
                    leakage_candidates.append((col, corr))
            except Exception:
                pass
        if leakage_candidates:
            cols_str = ", ".join(f"{c} (r={v:.3f})" for c, v in leakage_candidates)
            _add_check(report, "Data Leakage", False, "critical",
                        f"Potential leakage: {cols_str}",
                        "Investigate if these features are derived from the target")
        else:
            _add_check(report, "Data Leakage", True, "critical",
                        "No obvious data leakage detected")

    # ── Check 12: Duplicate columns ────────────────────────────────
    dup_cols = []
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            try:
                if df[cols[i]].equals(df[cols[j]]):
                    dup_cols.append((cols[i], cols[j]))
            except Exception:
                pass
    if dup_cols:
        pairs_str = ", ".join(f"{a}={b}" for a, b in dup_cols[:3])
        _add_check(report, "Duplicate Columns", False, "warning",
                    f"Identical columns found: {pairs_str}",
                    "Drop one of each duplicate pair")
    else:
        _add_check(report, "Duplicate Columns", True, "warning",
                    "No duplicate columns found")

    # ── Check 13: Constant features ────────────────────────────────
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if const_cols:
        _add_check(report, "Constant Features", False, "warning",
                    f"{len(const_cols)} constant column(s): {', '.join(const_cols[:5])}",
                    "Drop constant columns — they carry no information")
    else:
        _add_check(report, "Constant Features", True, "warning",
                    "No constant features detected")

    # ── Check 14: Extreme outliers ─────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns
    outlier_cols = []
    for col in num_cols:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        n_outliers = int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum())
        if n_outliers > len(s) * 0.05:
            outlier_cols.append((col, n_outliers, n_outliers / len(s) * 100))
    if outlier_cols:
        cols_str = ", ".join(f"{c} ({n:.1f}%)" for c, n, p in outlier_cols[:5])
        _add_check(report, "Extreme Outliers", False, "warning",
                    f"{len(outlier_cols)} column(s) with >5% outliers (3xIQR): {cols_str}",
                    "Investigate outliers — may need winsorization or transformation")
    else:
        _add_check(report, "Extreme Outliers", True, "warning",
                    "No extreme outliers detected (3xIQR threshold)")

    # ── Check 15: Unsupported datatypes ────────────────────────────
    unsupported = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        if dtype_str.startswith("complex") or dtype_str.startswith("object"):
            if df[col].dtype == object:
                n_unique = df[col].nunique()
                if n_unique > 1000:
                    unsupported.append((col, dtype_str, n_unique))
    if unsupported:
        cols_str = ", ".join(f"{c} ({t}, {u} unique)" for c, t, u in unsupported[:3])
        _add_check(report, "Unsupported Datatypes", False, "info",
                    f"High-cardinality text columns: {cols_str}",
                    "Consider encoding or dropping high-cardinality text columns")
    else:
        _add_check(report, "Unsupported Datatypes", True, "info",
                    "All datatypes appear compatible")

    # ── Check 16: Schema compliance ────────────────────────────────
    has_cust = customer_id_col in df.columns
    has_ts = timestamp_col in df.columns
    schema_score = sum([has_cust, has_ts])
    if schema_score == 2:
        _add_check(report, "Schema Compliance", True, "critical",
                    "Core schema present (customer_id + timestamp)")
    elif schema_score == 1:
        missing = "customer_id" if not has_cust else "timestamp"
        _add_check(report, "Schema Compliance", False, "critical",
                    f"Missing core column: {missing}",
                    f"Add the '{missing}' column for churn analysis")
    else:
        _add_check(report, "Schema Compliance", False, "critical",
                    "Missing both customer_id and timestamp",
                    "Both columns are required for churn analysis")

    logger.info(
        "Doctor report for '%s': score=%.0f%%, %d checks, %d critical, %d warnings",
        dataset_name, report.overall_score, len(report.checks),
        report.n_critical, report.n_warnings,
    )

    return report
