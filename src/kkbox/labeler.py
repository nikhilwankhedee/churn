"""
WSDM Cup 2018 KKBox churn labeler — faithful re-implementation.

Reference
---------
Official ``WSDMChurnLabeller.scala`` (see ``KKBOX_LABELLER_SCRIPT`` in
``src/config.py``).  This module reproduces the competition's churn semantics
deterministically from the raw transaction history, so labels can be derived
without the official ``train.csv`` / ``train_v2.csv`` files and validated
against them when those files are present (see ``src/kkbox/validation.py``).

The reference algorithm (verbatim semantics of the Scala source)
---------------------------------------------------------------
1. History is restricted to transactions with ``history_start <=
   transaction_date <= history_cutoff`` (20170101..20170131 by default).
2. For every member a *last effective expiration date* (``last_expire``) is
   derived from that history sequence, sorted by the reference comparator:
   (transaction_date ASC; plan signature DESC; plan-level tie-breaks).  The
   last row of the ordering determines ``last_expire``:
     * same date, different plan signature → larger signature sorts first
     * same date+signature, multiple cancellations → the smallest
       ``membership_expire_date`` wins (expiration moves earlier)
     * same date+signature, multiple renewals → the largest
       ``membership_expire_date`` wins (expiration extends)
     * same date+signature, renewal + cancellation → the cancellation wins
3. Prediction candidates are members whose ``last_expire`` falls inside the
   prediction month (``20170201``..``20170228`` by default).
4. Each candidate's future transactions (``transaction_date > history_cutoff``)
   are inspected in the same sorted order.  A cancellation that precedes any
   renewal moves ``last_expire`` earlier; the first subscription yields the
   *renewal gap* in days.  Members with no future activity get ``gap = 9999``.
5. ``churn = 1`` iff ``gap >= churn_window_days`` (30 by default); ``0`` iff a
   renewal occurred within the window.

Data-representation notes
-------------------------
- The Scala reads CSV strings and concatenates ``plan_list_price +
  payment_plan_days + payment_method_id`` into the plan signature.  For exact
  reproduction those three columns (and the date/expire/cancel columns) must
  therefore be passed as the raw strings found in the CSV.  The KKBox adapter
  feeds this labeller the raw, string-dtype transaction table.
- Dates are normalised to ``YYYYMMDD`` integers.  A transaction whose
  ``membership_expire_date`` cannot be parsed is treated as absent for the
  purpose of ``last_expire``; a member whose winning row has no valid
  expiration date is excluded from the candidate set (matching the reference,
  where a null ``last_expire`` never survives the prediction-window filter).
"""
from datetime import date, datetime
from functools import cmp_to_key
from typing import Optional

import numpy as np
import pandas as pd

from src.config import KKBOX_CHURN_WINDOW_DAYS
from src.utils import get_logger

logger = get_logger(__name__)

GAP_NO_RENEWAL: int = 9999


def _parse_yyyymmdd(value) -> Optional[int]:
    """Normalise a YYYYMMDD value (str/int/date/datetime) to int, or None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        value = pd.Timestamp(value).to_pydatetime()
    if isinstance(value, datetime):
        return int(value.strftime("%Y%m%d"))
    if isinstance(value, date):
        return int(value.strftime("%Y%m%d"))
    s = str(value).strip().replace("-", "").replace("/", "")
    if not (s.isdigit() and len(s) == 8):
        return None
    return int(s)


def _to_date(value: int) -> date:
    return date(value // 10000, (value // 100) % 100, value % 100)


def _to_csv_string(value) -> str:
    """String representation faithful to how the Scala reads CSV strings."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _raw_str(series: pd.Series) -> pd.Series:
    return series.map(_to_csv_string)


def _compare_rows(a: dict, b: dict) -> int:
    """Reference sort comparator from WSDMChurnLabeller.scala.

    Keys: ``_date`` (int), ``_sig`` (str), ``_expire`` (float/int/None),
    ``_cancel`` (int).  Returns -1/0/1 mirroring the Scala ``sortWith``
    predicate (x before y ⇔ lt(x, y) = true).
    """
    if a["_date"] != b["_date"]:
        return -1 if a["_date"] < b["_date"] else 1
    if a["_sig"] != b["_sig"]:
        return -1 if a["_sig"] > b["_sig"] else 1  # larger signature first
    if a["_cancel"] != b["_cancel"]:
        return -1 if a["_cancel"] < b["_cancel"] else 1  # renewal before cancel
    ea, eb = a["_expire"], b["_expire"]
    ea_nan = ea is None or (isinstance(ea, float) and np.isnan(ea))
    eb_nan = eb is None or (isinstance(eb, float) and np.isnan(eb))
    if ea_nan and eb_nan:
        return 0
    if ea_nan:
        return 1
    if eb_nan:
        return -1
    if a["_cancel"] == 1:  # both cancellations → larger expire first
        return -1 if ea > eb else (1 if ea < eb else 0)
    # both renewals → smaller expire first
    return -1 if ea < eb else (1 if ea > eb else 0)


class WSDMChurnLabeller:
    """Deterministic KKBox churn labeling from raw transaction data.

    Parameters
    ----------
    churn_window_days : int, optional
        Renewal window (days) inside which a subscription marks the member as
        NOT churned.  Defaults to KKBOX_CHURN_WINDOW_DAYS (30).
    history_start, history_cutoff : str, optional
        YYYYMMDD bounds of the history window used to derive ``last_expire``.
        Defaults to the official WSDM window (2017-01-01 .. 2017-01-31).
    prediction_start, prediction_end : str, optional
        YYYYMMDD bounds of the prediction month.  Defaults to 2017-02-01 ..
        2017-02-28.
    """

    def __init__(
        self,
        churn_window_days: Optional[int] = None,
        history_start: str = "20170101",
        history_cutoff: str = "20170131",
        prediction_start: str = "20170201",
        prediction_end: str = "20170228",
    ) -> None:
        self.churn_window_days = (KKBOX_CHURN_WINDOW_DAYS
                                  if churn_window_days is None
                                  else churn_window_days)
        self.history_start = _parse_yyyymmdd(history_start)
        self.history_cutoff = _parse_yyyymmdd(history_cutoff)
        self.prediction_start = _parse_yyyymmdd(prediction_start)
        self.prediction_end = _parse_yyyymmdd(prediction_end)
        if None in (self.history_start, self.history_cutoff,
                    self.prediction_start, self.prediction_end):
            raise ValueError("WSDM labeler windows must be YYYYMMDD strings")

    # ── Public API ───────────────────────────────────────────────────

    def compute_churn_labels(
        self,
        transactions: pd.DataFrame,
        msno_col: str = "msno",
        date_col: str = "transaction_date",
        expire_col: str = "membership_expire_date",
        plan_days_col: str = "payment_plan_days",
        price_col: str = "plan_list_price",
        method_col: str = "payment_method_id",
        cancel_col: str = "is_cancel",
    ) -> pd.DataFrame:
        """Compute per-member churn labels from the raw transaction table.

        Parameters
        ----------
        transactions : pd.DataFrame
            Raw transaction records.  Must contain ``msno``,
            ``transaction_date``, ``membership_expire_date``,
            ``payment_plan_days``, ``plan_list_price``,
            ``payment_method_id`` and ``is_cancel`` (ideally as raw CSV
            strings for exact plan-signature reproduction).
        msno_col, date_col, expire_col, plan_days_col, price_col,
        method_col, cancel_col : str
            Column-name overrides (defaults match the official schema).

        Returns
        -------
        pd.DataFrame with columns [customer_id, churn], one row per member
        whose ``last_expire`` falls in the prediction month.  Members outside
        that scope (no history, no valid expiration, different expiry month)
        receive no label.
        """
        required = [msno_col, date_col, expire_col, plan_days_col,
                    price_col, method_col, cancel_col]
        missing = [c for c in required if c not in transactions.columns]
        if missing:
            raise ValueError(
                f"KKBox labeler requires columns {missing}; "
                f"got {list(transactions.columns)}"
            )

        tx = transactions[[msno_col, date_col, expire_col, plan_days_col,
                           price_col, method_col, cancel_col]].copy()
        tx = tx.dropna(subset=[msno_col])
        tx[msno_col] = tx[msno_col].astype(str).str.strip()
        tx = tx[tx[msno_col] != ""].copy()
        if tx.empty:
            logger.warning("KKBox labeler: empty transaction history — no labels")
            return pd.DataFrame(columns=["customer_id", "churn"])

        tx["_date"] = tx[date_col].map(_parse_yyyymmdd)
        tx = tx.dropna(subset=["_date"]).copy()
        if tx.empty:
            logger.warning("KKBox labeler: no valid transaction dates — no labels")
            return pd.DataFrame(columns=["customer_id", "churn"])

        tx["_expire"] = tx[expire_col].map(_parse_yyyymmdd).astype(float)
        tx["_cancel"] = (
            pd.to_numeric(tx[cancel_col], errors="coerce").fillna(0).astype(int)
        )
        tx["_sig"] = (
            _raw_str(tx[price_col])
            + _raw_str(tx[plan_days_col])
            + _raw_str(tx[method_col])
        )

        last_expire = self._compute_last_expire(tx)
        if last_expire.empty:
            logger.warning("KKBox labeler: no history rows in window — no labels")
            return pd.DataFrame(columns=["customer_id", "churn"])

        lo, hi = self.prediction_start, self.prediction_end
        candidates = last_expire[(last_expire >= lo) & (last_expire <= hi)]
        if candidates.empty:
            logger.warning(
                "KKBox labeler: no members with last_expire in prediction "
                "month %s..%s",
                lo, hi,
            )
            return pd.DataFrame(columns=["customer_id", "churn"])

        future = tx[tx["_date"] > self.history_cutoff]
        future = future[future[msno_col].isin(candidates.index)]

        gaps = pd.Series(GAP_NO_RENEWAL, index=candidates.index, dtype=int)
        if not future.empty:
            computed = future.groupby(msno_col).apply(
                lambda g: self._renewal_gap(g, candidates.loc[g.name]),
            )
            computed.index = computed.index.astype(candidates.index.dtype)
            gaps.loc[computed.index] = computed.values

        churn = (gaps >= self.churn_window_days).astype(int)
        labels = pd.DataFrame({
            "customer_id": candidates.index.astype(str),
            "churn": churn.values,
        }).reset_index(drop=True)

        n_candidates = len(labels)
        n_churn = int(churn.sum())
        logger.info(
            "WSDM labels | %d candidates (expiry in %s..%s), churn rate "
            "%.2f%%, window %d days",
            n_candidates, lo, hi,
            (n_churn / n_candidates * 100) if n_candidates else 0.0,
            self.churn_window_days,
        )
        return labels

    # ── Core algorithm ───────────────────────────────────────────────

    def _compute_last_expire(self, tx: pd.DataFrame) -> pd.Series:
        """Effective expiration date per member (reference ``calculateLastday``).

        Returns a Series of ``_expire`` (float YYYYMMDD) indexed by msno for
        members with history rows and a valid winning expiration date.
        """
        history = tx[
            (tx["_date"] >= self.history_start)
            & (tx["_date"] <= self.history_cutoff)
        ]
        if history.empty:
            return pd.Series(dtype=float, name="_expire")

        # The winning row lies at the member's max date and, within that
        # date, at the smallest plan signature (larger signatures sort first).
        max_date = history.groupby("msno")["_date"].transform("max")
        hist_max = history[history["_date"] == max_date].copy()
        min_sig = hist_max.groupby("msno")["_sig"].transform("min")
        hist_win = hist_max[hist_max["_sig"] == min_sig].copy()

        # Within (member, date, signature): any cancellation wins → min
        # expire among cancellations; otherwise renewals → max expire.
        cancel_mask = hist_win["_cancel"] == 1
        renew_max = hist_win[~cancel_mask].groupby("msno")["_expire"].max()
        cancel_min = hist_win[cancel_mask].groupby("msno")["_expire"].min()
        cancel_members = hist_win[cancel_mask]["msno"].unique()

        last_expire = renew_max.reindex(hist_win["msno"].unique())
        if len(cancel_members):
            last_expire.loc[cancel_members] = cancel_min.loc[cancel_members]
        last_expire = last_expire.dropna()
        last_expire.name = "_expire"
        return last_expire

    def _renewal_gap(self, group: pd.DataFrame, last_expire_int: float) -> int:
        """Reference ``calculateRenewalGap``: days until first renewal.

        Cancellations preceding the first subscription move ``last_expire``
        earlier; the first subscription fixes the gap.  Returns
        GAP_NO_RENEWAL (9999) when no subscription is found.
        """
        rows = sorted(group.to_dict("records"), key=cmp_to_key(_compare_rows))
        last_date = _to_date(int(last_expire_int))
        gap = GAP_NO_RENEWAL
        for row in rows:
            if row["_cancel"] == 1:
                expire = row["_expire"]
                if expire is not None and not pd.isna(expire):
                    expiry = _to_date(int(expire))
                    if expiry < last_date:
                        last_date = expiry
            else:
                gap = (_to_date(int(row["_date"])) - last_date).days
                break
        return gap
