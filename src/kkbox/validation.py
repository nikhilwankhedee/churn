"""
KKBox validation harness.

Automatically validates the framework's WSDM-derived churn labels against the
official ground-truth labels whenever the official files (``train.csv`` /
``train_v2.csv``) are present in the data directory.

The harness is deliberately strict — a material discrepancy (agreement below
``KKBOX_AGREEMENT_THRESHOLD`` or coverage of the official set below
``KKBOX_MIN_OFFICIAL_COVERAGE``) is reported as VALIDATED_MISMATCH.  When the
raw transactions are absent the dataset is reported as DATA_MISSING and no
labels are ever fabricated; when transactions exist but official files do not,
labels are reported as UNVALIDATED.
"""
import hashlib
import os
from typing import Any, Dict, Optional

import pandas as pd

from src.config import (
    KKBOX_AGREEMENT_THRESHOLD,
    KKBOX_LOGS_FILE,
    KKBOX_LOGS_V2_FILE,
    KKBOX_MEMBERS_FILE,
    KKBOX_MEMBERS_V3_FILE,
    KKBOX_MIN_OFFICIAL_COVERAGE,
    KKBOX_RENEWAL_WINDOW_DAYS,
    KKBOX_TRAIN_FILE,
    KKBOX_TRAIN_V2_FILE,
    KKBOX_TRANSACTIONS_FILE,
    KKBOX_TRANSACTIONS_V2_FILE,
)
from src.utils import get_logger

logger = get_logger(__name__)

STATUS_MISSING = "DATA_MISSING"
STATUS_UNVALIDATED = "UNVALIDATED"
STATUS_MATCH = "VALIDATED_MATCH"
STATUS_MISMATCH = "VALIDATED_MISMATCH"

LABEL_COLUMNS = ["customer_id", "churn"]


def _first_existing(data_dir: str, *names: str) -> Optional[str]:
    for name in names:
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            return path
    return None


def check_data_availability(data_dir: str) -> Dict[str, Any]:
    """Report which KKBox files are present (or missing) in data_dir."""
    files = [
        ("train", KKBOX_TRAIN_FILE, KKBOX_TRAIN_V2_FILE),
        ("transactions", KKBOX_TRANSACTIONS_FILE, KKBOX_TRANSACTIONS_V2_FILE),
        ("logs", KKBOX_LOGS_FILE, KKBOX_LOGS_V2_FILE),
        ("members", KKBOX_MEMBERS_FILE, KKBOX_MEMBERS_V3_FILE),
    ]
    report = {}
    for key, *names in files:
        path = _first_existing(data_dir, *names)
        report[key] = os.path.basename(path) if path else None
    report["status"] = (
        STATUS_MISSING if report["transactions"] is None
        else (STATUS_UNVALIDATED if report["train"] is None else STATUS_MATCH)
    )
    return report


def _load_official(data_dir: str) -> Optional[Dict[str, Any]]:
    """Load the official label map, or None when no official file exists."""
    official_path = _first_existing(data_dir, KKBOX_TRAIN_FILE, KKBOX_TRAIN_V2_FILE)
    if official_path is None:
        return None

    official = pd.read_csv(official_path, usecols=lambda c: c in {"msno", "is_churn"})
    official["msno"] = official["msno"].astype(str)
    official = official.rename(columns={"msno": "customer_id", "is_churn": "churn"})
    official = official.drop_duplicates(subset="customer_id")
    official_map = official.set_index("customer_id")["churn"].astype(int)
    return {
        "path": official_path,
        "frame": official,
        "map": official_map,
        "n": len(official_map),
        "n_positive": int(official_map.sum()),
        "n_negative": int((official_map == 0).sum()),
    }


def validate_against_official(
    data_dir: str,
    computed_labels: pd.DataFrame,
) -> Dict[str, Any]:
    """Compare derived labels against official train.csv ground truth.

    Parameters
    ----------
    data_dir : str
        Directory to search for official label files.
    computed_labels : pd.DataFrame
        Labels from WSDMChurnLabeller (columns: customer_id, churn).

    Returns
    -------
    dict with keys: status, official_file, n_official, n_official_positive,
    n_official_negative, n_derived, n_derived_positive, n_derived_negative,
    n_overlap, n_matches, n_mismatches, agreement_rate, coverage, note.
    """
    official = _load_official(data_dir)
    if official is None:
        return {
            "status": STATUS_UNVALIDATED,
            "official_file": None,
            "n_official": 0,
            "n_official_positive": 0,
            "n_official_negative": 0,
            "n_derived": int(len(computed_labels)) if not computed_labels.empty else 0,
            "n_derived_positive": 0,
            "n_derived_negative": 0,
            "n_overlap": 0,
            "n_matches": 0,
            "n_mismatches": 0,
            "agreement_rate": None,
            "coverage": None,
            "note": "No official train.csv/train_v2.csv present — labels "
                    "remain UNVALIDATED.",
        }

    official_map = official["map"]
    if computed_labels is None or computed_labels.empty:
        return {
            "status": STATUS_MISMATCH,
            "official_file": os.path.basename(official["path"]),
            "n_official": official["n"],
            "n_official_positive": official["n_positive"],
            "n_official_negative": official["n_negative"],
            "n_derived": 0,
            "n_derived_positive": 0,
            "n_derived_negative": 0,
            "n_overlap": 0,
            "n_matches": 0,
            "n_mismatches": 0,
            "agreement_rate": 0.0,
            "coverage": 0.0,
            "note": "No derived labels to validate against official ground truth.",
        }

    computed = computed_labels.copy()
    computed["customer_id"] = computed["customer_id"].astype(str)
    computed_map = computed.set_index("customer_id")["churn"].astype(int)
    computed_map = computed_map[~computed_map.index.duplicated(keep="first")]

    overlap = computed_map.index.intersection(official_map.index)
    n_overlap = len(overlap)
    coverage = n_overlap / official["n"] if official["n"] else 0.0
    if n_overlap == 0:
        return {
            "status": STATUS_MISMATCH,
            "official_file": os.path.basename(official["path"]),
            "n_official": official["n"],
            "n_official_positive": official["n_positive"],
            "n_official_negative": official["n_negative"],
            "n_derived": len(computed_map),
            "n_derived_positive": int(computed_map.sum()),
            "n_derived_negative": int((computed_map == 0).sum()),
            "n_overlap": 0,
            "n_matches": 0,
            "n_mismatches": 0,
            "agreement_rate": 0.0,
            "coverage": round(coverage, 6),
            "note": "Zero overlap between derived and official customers.",
        }

    comp_overlap = computed_map.loc[overlap]
    offic_overlap = official_map.loc[overlap]
    n_matches = int((comp_overlap == offic_overlap).sum())
    n_mismatches = n_overlap - n_matches
    agreement = n_matches / n_overlap
    n_conf_tp = int(((comp_overlap == 1) & (offic_overlap == 1)).sum())
    n_conf_fp = int(((comp_overlap == 1) & (offic_overlap == 0)).sum())
    n_conf_fn = int(((comp_overlap == 0) & (offic_overlap == 1)).sum())
    n_conf_tn = int(((comp_overlap == 0) & (offic_overlap == 0)).sum())

    material = (agreement < KKBOX_AGREEMENT_THRESHOLD) or (
        coverage < KKBOX_MIN_OFFICIAL_COVERAGE
    )
    status = STATUS_MISMATCH if material else STATUS_MATCH
    note = (
        "Derived labels validated against official ground truth (PASS)."
        if not material
        else (
            "Material discrepancy vs official labels: "
            f"agreement {agreement:.4f} (< {KKBOX_AGREEMENT_THRESHOLD}) "
            f"or coverage {coverage:.4f} (< {KKBOX_MIN_OFFICIAL_COVERAGE})."
        )
    )

    logger.validation(
        "KKBox validation | agreement %.4f, coverage %.4f over %d customers "
        "(official %d, derived %d)",
        agreement, coverage, n_overlap, official["n"], len(computed_map),
    )
    return {
        "status": status,
        "official_file": os.path.basename(official["path"]),
        "n_official": official["n"],
        "n_official_positive": official["n_positive"],
        "n_official_negative": official["n_negative"],
        "n_derived": len(computed_map),
        "n_derived_positive": int(computed_map.sum()),
        "n_derived_negative": int((computed_map == 0).sum()),
        "n_overlap": n_overlap,
        "n_matches": n_matches,
        "n_mismatches": n_mismatches,
        "agreement_rate": round(agreement, 6),
        "coverage": round(coverage, 6),
        "confusion": {
            "official_pos_reproduced_pos": n_conf_tp,
            "official_pos_reproduced_neg": n_conf_fn,
            "official_neg_reproduced_pos": n_conf_fp,
            "official_neg_reproduced_neg": n_conf_tn,
        },
        "note": note,
    }


def run_kkbox_validation(
    data_dir: str,
    adapter: Optional["object"] = None,
) -> Dict[str, Any]:
    """Full automated KKBox validation: load → label → compare.

    Derives WSDM labels from the raw transaction file(s) in ``data_dir`` and
    validates them against the official ground truth whenever present.

    Returns
    -------
    dict status report.  Never fabricates: returns DATA_MISSING when no
    transactions exist and UNVALIDATED when no official labels exist.
    """
    report = check_data_availability(data_dir)
    if report["status"] == STATUS_MISSING:
        report["note"] = (
            "KKBox data NOT PRESENT — labels IMPLEMENTED but VALIDATION "
            "PENDING. Place the WSDM transaction files in the data directory "
            "to auto-validate."
        )
        return report

    if adapter is None:
        from src.datasets.kkbox import KKBoxAdapter
        adapter = KKBoxAdapter()
        adapter.data_dir = data_dir

    from src.kkbox.labeler import WSDMChurnLabeller

    raw = adapter._raw_transactions()
    labeller = WSDMChurnLabeller(churn_window_days=KKBOX_RENEWAL_WINDOW_DAYS)
    labels = labeller.compute_churn_labels(raw)

    report["n_derived"] = int(len(labels))
    report["derived_churn_rate"] = (
        float(labels["churn"].mean()) if len(labels) else None
    )
    report["labeller"] = "WSDMChurnLabeller"
    report["renewal_window_days"] = KKBOX_RENEWAL_WINDOW_DAYS

    validation = validate_against_official(data_dir, labels)
    report.update(validation)
    if validation["status"] == STATUS_MATCH:
        report["note"] = (
            "KKBox labels reproduced from raw transactions and validated "
            "against official ground truth (PASS)."
        )
    return report


def fingerprint_of(frame: pd.DataFrame) -> str:
    """Stable fingerprint of a label frame for experiment tracking."""
    ordered = frame.sort_values("customer_id")
    payload = ordered.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
