"""
Dataset profiler: produces a structured report from a standardized DataFrame.

Works on both raw and standardized schemas. The profiler inspects the data
without modifying it and returns a DatasetProfile dataclass.
"""
import dataclasses
import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class ColumnProfile:
    """Profiling results for a single column."""
    name: str
    dtype: str
    n_unique: int
    n_missing: int
    missing_pct: float
    is_constant: bool
    is_near_constant: bool
    unique_ratio: float
    sample_values: List[Any] = dataclasses.field(default_factory=list)
    stats: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class DatasetProfile:
    """Complete profiling results for a dataset."""

    # ── Overview ──────────────────────────────────────────────────
    n_rows: int = 0
    n_columns: int = 0
    memory_mb: float = 0.0
    duplicate_rows: int = 0
    duplicate_pct: float = 0.0
    timestamp: str = ""

    # ── Column Classification ─────────────────────────────────────
    numeric_columns: List[str] = dataclasses.field(default_factory=list)
    datetime_columns: List[str] = dataclasses.field(default_factory=list)
    categorical_columns: List[str] = dataclasses.field(default_factory=list)
    id_columns: List[str] = dataclasses.field(default_factory=list)
    constant_columns: List[str] = dataclasses.field(default_factory=list)
    near_constant_columns: List[str] = dataclasses.field(default_factory=list)
    column_profiles: Dict[str, ColumnProfile] = dataclasses.field(default_factory=dict)

    # ── Customer Analysis ─────────────────────────────────────────
    n_customers: int = 0
    avg_orders_per_customer: float = 0.0
    median_orders_per_customer: float = 0.0
    max_orders_per_customer: int = 0
    single_purchase_pct: float = 0.0

    # ── Temporal Analysis ─────────────────────────────────────────
    time_range: Optional[Tuple[str, str]] = None
    time_span_days: int = 0
    has_timestamps: bool = False
    timestamp_issues: List[str] = dataclasses.field(default_factory=list)

    # ── Missing Values ────────────────────────────────────────────
    columns_with_missing: int = 0
    total_missing_cells: int = 0
    missing_pct_overall: float = 0.0
    high_missing_columns: List[str] = dataclasses.field(default_factory=list)

    # ── Quality Issues ────────────────────────────────────────────
    warnings: List[str] = dataclasses.field(default_factory=list)
    critical_warnings: List[str] = dataclasses.field(default_factory=list)

    # ── Potential Targets ─────────────────────────────────────────
    likely_customer_id: Optional[str] = None
    likely_timestamp: Optional[str] = None
    likely_target: Optional[str] = None
    likely_monetary: Optional[str] = None
    likely_event_type: Optional[str] = None

    # ── Class Imbalance (if target detected) ──────────────────────
    class_distribution: Optional[Dict[str, int]] = None
    imbalance_ratio: Optional[float] = None

    # ── High Correlations ─────────────────────────────────────────
    high_correlations: List[Tuple[str, str, float]] = dataclasses.field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary."""
        return dataclasses.asdict(self)

    def summary_text(self) -> str:
        """Generate a human-readable summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("DATASET PROFILE")
        lines.append("=" * 60)
        lines.append(f"  Rows:              {self.n_rows:,}")
        lines.append(f"  Columns:           {self.n_columns}")
        lines.append(f"  Memory:            {self.memory_mb:.1f} MB")
        lines.append(f"  Duplicate rows:    {self.duplicate_rows:,} ({self.duplicate_pct:.1%})")
        lines.append("")

        if self.n_customers > 0:
            lines.append("  CUSTOMERS")
            lines.append(f"    Unique customers:        {self.n_customers:,}")
            lines.append(f"    Avg orders/customer:     {self.avg_orders_per_customer:.1f}")
            lines.append(f"    Median orders/customer:  {self.median_orders_per_customer:.0f}")
            lines.append(f"    Max orders/customer:     {self.max_orders_per_customer}")
            lines.append(f"    Single-purchase users:   {self.single_purchase_pct:.1%}")
            lines.append("")

        if self.has_timestamps and self.time_range:
            lines.append("  TEMPORAL")
            lines.append(f"    Range:           {self.time_range[0]} to {self.time_range[1]}")
            lines.append(f"    Span:            {self.time_span_days} days")
            lines.append("")

        lines.append("  COLUMNS")
        lines.append(f"    Numeric:         {len(self.numeric_columns)}")
        lines.append(f"    Datetime:        {len(self.datetime_columns)}")
        lines.append(f"    Categorical:     {len(self.categorical_columns)}")
        lines.append(f"    ID-like:         {len(self.id_columns)}")
        lines.append(f"    Constant:        {len(self.constant_columns)}")
        lines.append(f"    Near-constant:   {len(self.near_constant_columns)}")
        lines.append("")

        if self.columns_with_missing > 0:
            lines.append("  MISSING VALUES")
            lines.append(f"    Columns affected: {self.columns_with_missing}")
            lines.append(f"    Total missing:    {self.total_missing_cells:,} ({self.missing_pct_overall:.1%})")
            if self.high_missing_columns:
                lines.append(f"    High missing:     {', '.join(self.high_missing_columns[:5])}")
            lines.append("")

        if self.likely_customer_id:
            lines.append("  AUTO-DETECTED COLUMNS")
            lines.append(f"    Customer ID:      {self.likely_customer_id}")
            if self.likely_timestamp:
                lines.append(f"    Timestamp:        {self.likely_timestamp}")
            if self.likely_monetary:
                lines.append(f"    Monetary value:   {self.likely_monetary}")
            if self.likely_event_type:
                lines.append(f"    Event type:       {self.likely_event_type}")
            lines.append("")

        if self.class_distribution:
            lines.append("  CLASS DISTRIBUTION")
            for cls_name, count in self.class_distribution.items():
                pct = count / self.n_rows * 100 if self.n_rows > 0 else 0
                lines.append(f"    {cls_name}: {count:,} ({pct:.1f}%)")
            if self.imbalance_ratio is not None:
                lines.append(f"    Imbalance ratio: {self.imbalance_ratio:.1f}")
            lines.append("")

        if self.high_correlations:
            lines.append("  HIGH CORRELATIONS")
            for col_a, col_b, corr in self.high_correlations[:5]:
                lines.append(f"    {col_a} <-> {col_b}: {corr:.3f}")
            lines.append("")

        if self.warnings:
            lines.append("  WARNINGS")
            for w in self.warnings:
                lines.append(f"    - {w}")
            lines.append("")

        if self.critical_warnings:
            lines.append("  CRITICAL WARNINGS")
            for w in self.critical_warnings:
                lines.append(f"    ! {w}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


def profile_dataset(
    df: pd.DataFrame,
    customer_id_col: str = "customer_id",
    timestamp_col: str = "event_time",
    target_col: Optional[str] = None,
    constant_threshold: float = 0.99,
    near_constant_threshold: float = 0.95,
    high_missing_threshold: float = 0.5,
    correlation_threshold: float = 0.95,
) -> DatasetProfile:
    """Profile a dataset and return a structured report.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to profile (raw or standardized).
    customer_id_col : str
        Name of the customer identifier column.
    timestamp_col : str
        Name of the timestamp column.
    target_col : str, optional
        Name of the target/label column if known.
    constant_threshold : float
        Fraction of identical values above which a column is "constant".
    near_constant_threshold : float
        Fraction above which a column is "near-constant".
    high_missing_threshold : float
        Fraction above which a column has "high missing".
    correlation_threshold : float
        Absolute correlation above which pairs are flagged.

    Returns
    -------
    DatasetProfile with all findings.
    """
    profile = DatasetProfile()
    profile.n_rows = len(df)
    profile.n_columns = len(df.columns)
    profile.memory_mb = df.memory_usage(deep=True).sum() / 1e6
    profile.timestamp = datetime.datetime.now().isoformat()

    # ── Duplicates ────────────────────────────────────────────────
    profile.duplicate_rows = int(df.duplicated().sum())
    profile.duplicate_pct = profile.duplicate_rows / max(profile.n_rows, 1)

    # ── Column classification ─────────────────────────────────────
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        n_missing = int(df[col].isnull().sum())
        missing_pct = n_missing / max(profile.n_rows, 1)
        unique_ratio = n_unique / max(profile.n_rows, 1)
        value_counts = df[col].value_counts(dropna=True)
        most_common_pct = value_counts.iloc[0] / max(profile.n_rows, 1) if len(value_counts) > 0 else 0.0
        is_const = n_unique <= 1 or most_common_pct >= constant_threshold
        is_near_const = (not is_const and
                         (n_unique <= 3 or most_common_pct >= near_constant_threshold))

        cp = ColumnProfile(
            name=col,
            dtype=str(df[col].dtype),
            n_unique=n_unique,
            n_missing=n_missing,
            missing_pct=missing_pct,
            is_constant=is_const,
            is_near_constant=is_near_const,
            unique_ratio=unique_ratio,
        )

        # Stats for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]) and not is_const:
            cp.stats = {
                "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                "std": float(df[col].std()) if not df[col].isnull().all() else None,
                "min": float(df[col].min()) if not df[col].isnull().all() else None,
                "max": float(df[col].max()) if not df[col].isnull().all() else None,
                "median": float(df[col].median()) if not df[col].isnull().all() else None,
                "skew": float(df[col].skew()) if not df[col].isnull().all() else None,
            }
            profile.numeric_columns.append(col)

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            profile.datetime_columns.append(col)
            cp.stats = {
                "min": str(df[col].min()),
                "max": str(df[col].max()),
            }

        elif df[col].dtype == object or df[col].dtype.name == "category":
            if n_unique < 100 and n_unique < profile.n_rows * 0.5:
                profile.categorical_columns.append(col)
            # Sample values
            top_vals = df[col].value_counts().head(5).index.tolist()
            cp.sample_values = [str(v) for v in top_vals]
            cp.stats = {"top_values": cp.sample_values}

        profile.column_profiles[col] = cp

        if is_const:
            profile.constant_columns.append(col)
        if is_near_const:
            profile.near_constant_columns.append(col)

    # ── ID column detection ───────────────────────────────────────
    for col in df.columns:
        if col in profile.constant_columns:
            continue
        n_unique = df[col].nunique()
        if n_unique == profile.n_rows:
            profile.id_columns.append(col)
        elif (profile.n_rows > 100 and
              n_unique > profile.n_rows * 0.95 and
              pd.api.types.is_numeric_dtype(df[col])):
            profile.id_columns.append(col)

    # ── Auto-detect key columns ───────────────────────────────────
    profile.likely_customer_id = _detect_customer_id(df, customer_id_col)
    profile.likely_timestamp = _detect_timestamp(df, timestamp_col)
    profile.likely_monetary = _detect_monetary(df)
    profile.likely_event_type = _detect_event_type(df)
    if target_col and target_col in df.columns:
        profile.likely_target = target_col

    # ── Customer analysis ─────────────────────────────────────────
    cust_col = profile.likely_customer_id
    if cust_col and cust_col in df.columns:
        profile.n_customers = int(df[cust_col].nunique())
        orders_per_customer = df.groupby(cust_col).size()
        profile.avg_orders_per_customer = float(orders_per_customer.mean())
        profile.median_orders_per_customer = float(orders_per_customer.median())
        profile.max_orders_per_customer = int(orders_per_customer.max())
        profile.single_purchase_pct = float(
            (orders_per_customer == 1).sum() / max(len(orders_per_customer), 1)
        )

    # ── Temporal analysis ─────────────────────────────────────────
    ts_col = profile.likely_timestamp
    if ts_col and ts_col in df.columns:
        ts = df[ts_col].dropna()
        if not ts.empty and pd.api.types.is_datetime64_any_dtype(ts):
            profile.has_timestamps = True
            profile.time_range = (str(ts.min()), str(ts.max()))
            profile.time_span_days = int((ts.max() - ts.min()).days)

            # Check for future timestamps
            now = pd.Timestamp.now()
            future_count = int((ts > now).sum())
            if future_count > 0:
                profile.timestamp_issues.append(
                    f"{future_count} timestamps are in the future"
                )

            # Check for non-monotonic ordering
            if len(ts) > 1000:
                sample = ts.sample(min(10000, len(ts)), random_state=42)
                n_decreasing = int((sample.diff() < pd.Timedelta(0)).sum())
                if n_decreasing > len(sample) * 0.5:
                    profile.timestamp_issues.append(
                        "Timestamps are not monotonically ordered"
                    )

    # ── Missing values ────────────────────────────────────────────
    total_cells = profile.n_rows * profile.n_columns
    profile.total_missing_cells = int(df.isnull().sum().sum())
    profile.missing_pct_overall = profile.total_missing_cells / max(total_cells, 1)
    profile.columns_with_missing = int((df.isnull().sum() > 0).sum())
    miss_pct = df.isnull().mean()
    profile.high_missing_columns = sorted(
        miss_pct[miss_pct > high_missing_threshold].index.tolist()
    )

    # ── Class imbalance ───────────────────────────────────────────
    if profile.likely_target and profile.likely_target in df.columns:
        target = df[profile.likely_target].dropna()
        if target.nunique() <= 20:
            profile.class_distribution = target.value_counts().to_dict()
            counts = target.value_counts()
            if len(counts) == 2:
                profile.imbalance_ratio = float(counts.min() / counts.max())

    # ── High correlations ─────────────────────────────────────────
    if len(profile.numeric_columns) >= 2:
        num_df = df[profile.numeric_columns].select_dtypes(include=[np.number])
        if num_df.shape[1] >= 2:
            try:
                corr_matrix = num_df.corr().abs()
                upper = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                for col_a in upper.columns:
                    for col_b in upper.index:
                        val = upper.loc[col_b, col_a]
                        if pd.notna(val) and val >= correlation_threshold:
                            profile.high_correlations.append(
                                (col_a, col_b, float(val))
                            )
                profile.high_correlations.sort(key=lambda x: x[2], reverse=True)
            except Exception:
                pass

    # ── Warnings ──────────────────────────────────────────────────
    if profile.duplicate_rows > 0:
        profile.warnings.append(
            f"{profile.duplicate_rows:,} duplicate rows detected"
        )

    if profile.high_missing_columns:
        profile.warnings.append(
            f"High missing values in: {', '.join(profile.high_missing_columns[:5])}"
        )

    if profile.constant_columns:
        profile.warnings.append(
            f"Constant columns detected: {', '.join(profile.constant_columns[:5])}"
        )

    if profile.near_constant_columns:
        profile.warnings.append(
            f"Near-constant columns: {', '.join(profile.near_constant_columns[:5])}"
        )

    if profile.single_purchase_pct > 0.5:
        profile.warnings.append(
            f"{profile.single_purchase_pct:.0%} of customers have only 1 order — "
            "repeat-purchase features may be uninformative"
        )

    if profile.imbalance_ratio is not None and profile.imbalance_ratio > 10:
        profile.warnings.append(
            f"High class imbalance (ratio: {profile.imbalance_ratio:.1f}) — "
            "consider stratified splits and balanced class weights"
        )

    if profile.time_span_days and profile.time_span_days < 180:
        profile.warnings.append(
            f"Short observation window ({profile.time_span_days} days) — "
            "churn labeling may be unreliable"
        )

    if profile.high_correlations:
        top_pairs = profile.high_correlations[:3]
        pairs_str = ", ".join(f"{a}↔{b} ({c:.2f})" for a, b, c in top_pairs)
        profile.warnings.append(
            f"High feature correlations detected: {pairs_str}"
        )

    # Leakage heuristic: if any feature has correlation > 0.99 with target
    if profile.likely_target and profile.likely_target in df.columns:
        for col in profile.numeric_columns:
            if col == profile.likely_target:
                continue
            try:
                corr = abs(df[col].corr(df[profile.likely_target]))
                if corr > 0.99:
                    profile.critical_warnings.append(
                        f"POTENTIAL LEAKAGE: '{col}' has {corr:.3f} correlation "
                        f"with target '{profile.likely_target}'"
                    )
            except Exception:
                pass

    logger.info(
        "Dataset profiled: %d rows, %d columns, %d warnings, %d critical",
        profile.n_rows, profile.n_columns,
        len(profile.warnings), len(profile.critical_warnings),
    )
    return profile


# ── Column Detection Helpers ─────────────────────────────────────

def _detect_customer_id(
    df: pd.DataFrame, hint: str = "customer_id",
) -> Optional[str]:
    """Detect the most likely customer identifier column."""
    # Check hint first
    if hint in df.columns:
        n_unique = df[hint].nunique()
        if n_unique > 0 and n_unique <= len(df) * 0.99:
            return hint

    # Heuristic: column with unique-ish values and 'id' in the name
    candidates = []
    for col in df.columns:
        if col in ("event_type", "product_id", "session_id"):
            continue
        n_unique = df[col].nunique()
        name_lower = col.lower()
        if any(kw in name_lower for kw in ("customer", "user", "visitor", "client")):
            candidates.append((col, n_unique, "name_match"))
        elif (n_unique > 100 and n_unique < len(df) * 0.99 and
              "id" in name_lower):
            candidates.append((col, n_unique, "id_suffix"))

    if candidates:
        # Prefer name_match, then highest cardinality
        candidates.sort(key=lambda x: (0 if x[2] == "name_match" else 1, -x[1]))
        return candidates[0][0]

    return None


def _detect_timestamp(
    df: pd.DataFrame, hint: str = "event_time",
) -> Optional[str]:
    """Detect the most likely timestamp column."""
    if hint in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[hint]):
            return hint

    # Check for datetime columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    # Check for string columns that look like timestamps
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(5)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample)
                    return col
                except Exception:
                    continue

    return None


def _detect_monetary(df: pd.DataFrame) -> Optional[str]:
    """Detect the most likely monetary value column."""
    candidates = []
    for col in df.columns:
        name_lower = col.lower()
        if col.lower() in ("transaction_value", "payment_value", "order_value"):
            candidates.append((col, df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else 0, "exact_match"))
        elif any(kw in name_lower for kw in
               ("value", "amount", "price", "revenue", "spend", "total", "payment")):
            if pd.api.types.is_numeric_dtype(df[col]):
                candidates.append((col, df[col].mean(), "name_match"))

    if candidates:
        candidates.sort(key=lambda x: (0 if x[2] == "exact_match" else 1, -x[1]))
        return candidates[0][0]

    return None


def _detect_event_type(df: pd.DataFrame) -> Optional[str]:
    """Detect the most likely event type column."""
    candidates = ["event_type", "event", "action", "type", "activity"]
    for col in candidates:
        if col in df.columns:
            return col

    # Check for low-cardinality object columns
    for col in df.columns:
        if df[col].dtype == object:
            n_unique = df[col].nunique()
            if 2 <= n_unique <= 20:
                vals = set(str(v).lower() for v in df[col].unique())
                event_words = {"view", "purchase", "cart", "add", "remove",
                               "click", "buy", "order", "transaction"}
                if vals & event_words:
                    return col

    return None
