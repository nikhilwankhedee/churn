"""
CSV column inspector: infers column roles for churn analysis.

Reuses detection heuristics from src.profiling.profiler but operates
on raw CSV data before any framework preprocessing.
"""
import dataclasses
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

# ── Standardized column roles the wizard can infer ────────────────
COLUMN_ROLES = [
    "customer_id",
    "event_time",
    "transaction_value",
    "event_type",
    "product_id",
    "review_score",
    "payment_type",
    "delivery_delay",
    "engagement_signal",
    "session_id",
]

# Name hints for each role
_ROLE_HINTS: Dict[str, List[str]] = {
    "customer_id": [
        "customer", "user", "visitor", "client", "subscriber", "account",
        "buyer", "shopper", "member",
    ],
    "event_time": [
        "timestamp", "date", "time", "created_at", "updated_at",
        "event_time", "order_date", "purchase_date", "transaction_date",
    ],
    "transaction_value": [
        "value", "amount", "price", "revenue", "spend", "total",
        "payment", "cost", "order_value", "transaction_value",
    ],
    "event_type": [
        "event_type", "event", "action", "type", "activity",
    ],
    "product_id": [
        "product_id", "item_id", "sku", "product", "article_id",
    ],
    "review_score": [
        "review", "rating", "score", "feedback", "star",
    ],
    "payment_type": [
        "payment_type", "payment_method", "payment", "method", "tender",
    ],
    "session_id": [
        "session_id", "visit_id", "browse_session", "session",
    ],
}


@dataclasses.dataclass
class ColumnInspection:
    """Inspection results for a single column."""
    name: str
    dtype: str
    n_unique: int
    n_missing: int
    missing_pct: float
    inferred_role: Optional[str]
    confidence: float  # 0.0 to 1.0
    reason: str
    sample_values: List[Any] = dataclasses.field(default_factory=list)
    is_numeric: bool = False
    is_datetime: bool = False
    is_categorical: bool = False
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None


@dataclasses.dataclass
class InspectionResult:
    """Complete inspection results for a CSV file."""
    file_path: str
    n_rows: int
    n_columns: int
    columns: List[ColumnInspection] = dataclasses.field(default_factory=list)
    inferred_customer_id: Optional[str] = None
    inferred_event_time: Optional[str] = None
    inferred_transaction_value: Optional[str] = None
    inferred_event_type: Optional[str] = None
    suggested_dataset_name: str = "custom"
    warnings: List[str] = dataclasses.field(default_factory=list)
    suggestions: List[str] = dataclasses.field(default_factory=list)


def _infer_role_from_name(col_name: str) -> tuple[Optional[str], float, str]:
    """Infer column role from name heuristics. Returns (role, confidence, reason)."""
    name_lower = col_name.lower().strip()

    for role, hints in _ROLE_HINTS.items():
        for hint in hints:
            if hint == name_lower:
                return role, 0.95, f"exact match: '{col_name}' == '{hint}'"
            if hint in name_lower:
                return role, 0.8, f"name contains '{hint}'"

    return None, 0.0, ""


def _infer_role_from_values(
    col_name: str,
    series: pd.Series,
    n_rows: int,
) -> tuple[Optional[str], float, str]:
    """Infer column role from value patterns. Returns (role, confidence, reason)."""
    n_unique = series.nunique(dropna=True)
    missing_pct = series.isnull().sum() / max(n_rows, 1)
    unique_ratio = n_unique / max(n_rows, 1)

    # ── Timestamp detection ────────────────────────────────────────
    if pd.api.types.is_datetime64_any_dtype(series):
        return "event_time", 0.9, "datetime64 dtype"

    if series.dtype == object:
        sample = series.dropna().head(10)
        if len(sample) > 0:
            try:
                pd.to_datetime(sample)
                return "event_time", 0.85, "string parses as datetime"
            except Exception:
                pass

    # ── Numeric roles ──────────────────────────────────────────────
    if pd.api.types.is_numeric_dtype(series):
        mean_val = series.mean() if not series.isnull().all() else 0

        # ID-like: high cardinality, all unique
        if n_unique == n_rows and n_rows > 10:
            return None, 0.0, ""  # Could be an ID, but not customer_id

        # Transaction value: positive, right-skewed, monetary-like range
        if mean_val > 0 and series.skew() > 0.5:
            name_hints = [h for h in _ROLE_HINTS["transaction_value"]
                         if h in col_name.lower()]
            if name_hints:
                return "transaction_value", 0.85, "positive numeric, right-skewed, name match"

        # Review score: low cardinality integer-like
        if n_unique <= 10 and all(series.dropna() == series.dropna().astype(int)):
            name_hints = [h for h in _ROLE_HINTS["review_score"]
                         if h in col_name.lower()]
            if name_hints:
                return "review_score", 0.8, "low-cardinality integer, name match"

    # ── Categorical / event type ───────────────────────────────────
    if series.dtype == object and 2 <= n_unique <= 30:
        vals = set(str(v).lower() for v in series.dropna().unique())
        event_words = {
            "view", "purchase", "cart", "add", "remove", "click",
            "buy", "order", "transaction", "install", "uninstall",
            "login", "logout", "signup", "pageview",
        }
        if vals & event_words:
            return "event_type", 0.85, f"contains event-like values: {vals & event_words}"

        payment_words = {
            "credit", "debit", "cash", "paypal", "visa", "mastercard",
            "bank", "transfer", "wallet", "apple_pay", "google_pay",
        }
        if vals & payment_words:
            return "payment_type", 0.8, f"contains payment-like values: {vals & payment_words}"

    return None, 0.0, ""


def _guess_dataset_name(file_path: str) -> str:
    """Guess a dataset name from the file path."""
    import os
    base = os.path.splitext(os.path.basename(file_path))[0]
    # Clean up common prefixes/suffixes
    for suffix in ["_data", "_dataset", "_raw", "_clean", "_processed"]:
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
    return base.lower().replace(" ", "_").replace("-", "_")


def inspect_csv(
    file_path: str,
    customer_id_hint: Optional[str] = None,
    timestamp_hint: Optional[str] = None,
    max_sample_rows: int = 10_000,
) -> InspectionResult:
    """Inspect a CSV file and infer column roles.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    customer_id_hint : str, optional
        User-provided customer ID column name.
    timestamp_hint : str, optional
        User-provided timestamp column name.
    max_sample_rows : int
        Maximum rows to load for inspection (for performance).

    Returns
    -------
    InspectionResult with inferred roles and suggestions.
    """
    logger.info("Inspecting CSV: %s", file_path)

    # Load sample
    df = pd.read_csv(file_path, nrows=max_sample_rows, low_memory=False)
    n_rows = len(df)

    result = InspectionResult(
        file_path=file_path,
        n_rows=n_rows,
        n_columns=len(df.columns),
        suggested_dataset_name=_guess_dataset_name(file_path),
    )

    # Track which roles have been assigned to avoid duplicates
    assigned_roles: Dict[str, str] = {}  # role -> column name

    # ── Phase 1: Name-based inference (highest confidence) ─────────
    for col in df.columns:
        role, confidence, reason = _infer_role_from_name(col)
        if role and confidence >= 0.8 and role not in assigned_roles:
            assigned_roles[role] = col

    # ── Phase 2: Value-based inference ─────────────────────────────
    for col in df.columns:
        if col in assigned_roles.values():
            continue
        role, confidence, reason = _infer_role_from_values(col, df[col], n_rows)
        if role and confidence >= 0.75 and role not in assigned_roles:
            assigned_roles[role] = col

    # ── Phase 3: Apply user hints (override everything) ────────────
    if customer_id_hint and customer_id_hint in df.columns:
        assigned_roles["customer_id"] = customer_id_hint
    if timestamp_hint and timestamp_hint in df.columns:
        assigned_roles["event_time"] = timestamp_hint

    # ── Build per-column inspection ────────────────────────────────
    for col in df.columns:
        series = df[col]
        n_unique = series.nunique(dropna=True)
        n_missing = int(series.isnull().sum())
        missing_pct = n_missing / max(n_rows, 1)

        # Find the role assigned to this column
        inferred_role = None
        confidence = 0.0
        reason = ""
        for role, role_col in assigned_roles.items():
            if role_col == col:
                inferred_role = role
                confidence = 0.9  # High confidence for assigned roles
                _, _, reason = _infer_role_from_name(col)
                if not reason:
                    _, _, reason = _infer_role_from_values(col, series, n_rows)
                reason = reason or "user hint" if (customer_id_hint == col or timestamp_hint == col) else reason
                break

        # If no role, still try name inference for display
        if inferred_role is None:
            inferred_role_name, confidence, reason = _infer_role_from_name(col)
            # Don't assign, just display
            display_role = inferred_role_name
            display_confidence = confidence
            display_reason = reason
        else:
            display_role = inferred_role
            display_confidence = confidence
            display_reason = reason

        # Sample values
        sample_values = []
        if series.dtype == object:
            sample_values = [str(v) for v in series.dropna().unique()[:5]]
        elif pd.api.types.is_numeric_dtype(series):
            sample_values = [float(series.min()), float(series.max())]

        cp = ColumnInspection(
            name=col,
            dtype=str(series.dtype),
            n_unique=n_unique,
            n_missing=n_missing,
            missing_pct=missing_pct,
            inferred_role=display_role if display_role else None,
            confidence=display_confidence,
            reason=display_reason,
            sample_values=sample_values,
            is_numeric=pd.api.types.is_numeric_dtype(series),
            is_datetime=pd.api.types.is_datetime64_any_dtype(series),
            is_categorical=series.dtype == object and n_unique < 100,
            min_value=str(series.min()) if not series.isnull().all() else None,
            max_value=str(series.max()) if not series.isnull().all() else None,
        )
        result.columns.append(cp)

    # ── Set primary inferences ─────────────────────────────────────
    result.inferred_customer_id = assigned_roles.get("customer_id")
    result.inferred_event_time = assigned_roles.get("event_time")
    result.inferred_transaction_value = assigned_roles.get("transaction_value")
    result.inferred_event_type = assigned_roles.get("event_type")

    # ── Generate warnings ──────────────────────────────────────────
    if not result.inferred_customer_id:
        result.warnings.append(
            "Could not auto-detect customer ID column — "
            "you must specify it manually"
        )

    if not result.inferred_event_time:
        result.warnings.append(
            "Could not auto-detect timestamp column — "
            "churn labeling requires timestamps"
        )

    if not result.inferred_transaction_value:
        result.suggestions.append(
            "No monetary value column detected — "
            "monetary features will be unavailable"
        )

    # Check for high-missing columns
    for cp in result.columns:
        if cp.missing_pct > 0.5:
            result.warnings.append(
                f"Column '{cp.name}' has {cp.missing_pct:.0%} missing values"
            )

    # Check temporal span
    ts_col = result.inferred_event_time
    if ts_col and ts_col in df.columns:
        ts = df[ts_col]
        if pd.api.types.is_datetime64_any_dtype(ts):
            span = (ts.max() - ts.min()).days
            if span < 180:
                result.warnings.append(
                    f"Temporal span is only {span} days — "
                    "180-day churn window may not be reliable"
                )
            result.suggestions.append(
                f"Time range: {ts.min()} to {ts.max()} ({span} days)"
            )

    # Customer count
    cust_col = result.inferred_customer_id
    if cust_col and cust_col in df.columns:
        n_customers = df[cust_col].nunique()
        result.suggestions.append(f"Detected {n_customers:,} unique customers")

    logger.info(
        "Inspection complete: %d columns, customer_id=%s, event_time=%s",
        result.n_columns, result.inferred_customer_id, result.inferred_event_time,
    )

    return result
