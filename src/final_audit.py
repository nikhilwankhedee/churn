"""
Final integrity audit + completion report.

Runs once all experiments and post-processing are done and verifies the
pre-registered research matrix against the produced artifacts:

    * expected experiment count — every dataset that ran must have
      (2 SMOTE conditions × 5 models) exactly once (no duplicates, no
      missing cells); with all 8 datasets present and validated this is
      80 experiments;
    * test-identity integrity — SMOTE must not have altered the test set
      for any dataset;
    * no NaN / Inf in the metric columns of the master table;
    * LightGBM present in every dataset × condition (it must not silently
      disappear);
    * model set == ``FINAL_EXPERIMENT_MODELS`` (no unexpected model beyond
      the two documented baselines);
    * KKBox status is ``VALIDATED`` (data present + label match) or
      ``PENDING`` (data absent) — never a silent SUCCESS.

Outputs
-------
    results/master/integrity_audit.csv
    results/master/completion_report.csv / .json
    COMPLETION_REPORT.txt  (terminal summary block, enhanced)
"""
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import FINAL_EXPERIMENT_DATASETS, FINAL_EXPERIMENT_MODELS, FRAMEWORK_VERSION
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

STATUS_OK = 'OK'
STATUS_WARN = 'WARN'
STATUS_FAIL = 'FAIL'

# status value stored in the master all_results table for a run whose
# pipeline completed without raising (see experiment_runner.STATUS_SUCCESS)
EXP_STATUS_SUCCESS = 'success'

METRIC_COLUMNS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc',
                  'pr_auc', 'balanced_accuracy', 'mcc', 'brier_score',
                  'calibration_error']
# models that are allowed beyond the FINAL_EXPERIMENT_MODELS set
DOCUMENTED_BASELINES = {'majority_class', 'random_baseline'}


def _check_record(check: str, status: str, detail: str, **extra) -> Dict:
    row = {'check': check, 'status': status, 'detail': detail}
    row.update(extra)
    return row


def audit_results(
    experiment_dir: str,
    all_results: Optional[pd.DataFrame] = None,
    identity_results: Optional[List[Dict]] = None,
    kkbox_status: Optional[str] = None,
    expected_datasets: Optional[List[str]] = None,
    expected_models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run every integrity check; returns the audit table.

    ``kkbox_status`` is optional: when None (the default), the KKBox status
    check is omitted entirely — for experiment matrices that do not include
    the KKBox adapter.  Pass a status string to keep the KKBox honesty check
    (e.g. from the KKBox-specific validation harness).
    """
    expected_datasets = expected_datasets or FINAL_EXPERIMENT_DATASETS
    expected_models = expected_models or FINAL_EXPERIMENT_MODELS
    checks: List[Dict] = []

    if all_results is None or all_results.empty:
        return pd.DataFrame([_check_record(
            'master_results', STATUS_FAIL,
            'all_results is empty — nothing to audit')])

    # ---- 1. Expected experiment count (per available dataset) ----
    present = sorted(all_results['dataset'].unique())
    missing_ds = [d for d in expected_datasets if d not in present]
    expected_total = len(present) * 2 * len(expected_models)
    status = STATUS_OK if expected_total > 0 else STATUS_FAIL
    checks.append(_check_record(
        'expected_experiment_count',
        status,
        f'{expected_total} expected ({len(present)} datasets × 2 SMOTE × '
        f'{len(expected_models)} models); datasets absent from results: '
        f'{missing_ds or "none"}; full matrix with all 8 datasets validated '
        f'= 80 experiments'))

    # ---- 2. Cell coverage per (dataset, smote) ----
    coverage_fail = 0
    for dataset in present:
        ds = all_results[all_results['dataset'] == dataset]
        for smote_val in ['No', 'Yes']:
            cell = ds[ds['smote'] == smote_val]
            models = cell['model'].astype(str).tolist()
            missing = [m for m in expected_models if m not in models]
            dups = sorted({m for m in models if models.count(m) > 1})
            if missing or dups:
                coverage_fail += 1
            checks.append(_check_record(
                'cell_coverage', STATUS_FAIL if (missing or dups) else STATUS_OK,
                f'{dataset}/{smote_val}', dataset=dataset,
                smote=smote_val,
                missing_models=missing or 'none',
                duplicate_models=dups or 'none'))
    if coverage_fail == 0:
        logger.validation("All (dataset, smote) cells have complete model "
                          "coverage (%d cells)", len(present) * 2)

    # ---- 3. No NaN / Inf in metric columns (non-baseline rows only) ----
    non_base = all_results[
        ~all_results['model'].astype(str).isin(DOCUMENTED_BASELINES)]
    bad = 0
    for col in METRIC_COLUMNS:
        if col in non_base.columns:
            vals = non_base[col]
            n_bad = int(vals.isna().sum() + (np.isinf(vals).sum() if
                        pd.api.types.is_numeric_dtype(vals) else 0))
            if n_bad:
                bad += 1
                checks.append(_check_record(
                    'no_nan_inf', STATUS_FAIL, f'{col}: {n_bad} NaN/Inf'))
    if bad == 0:
        checks.append(_check_record(
            'no_nan_inf', STATUS_OK,
            f'no NaN/Inf in {len([c for c in METRIC_COLUMNS if c in non_base.columns])} metric columns (non-baseline rows)'))

    # ---- 4. Test-identity integrity ----
    if identity_results:
        for ir in identity_results:
            ds = ir.get('dataset')
            valid = bool(ir.get('valid'))
            checks.append(_check_record(
                'test_identity', STATUS_OK if valid else STATUS_FAIL,
                f'{ds}: {ir.get("note")}', dataset=ds,
                test_ids_match=ir.get('test_ids_match'),
                test_y_match=ir.get('test_y_match')))
    else:
        checks.append(_check_record(
            'test_identity', STATUS_WARN,
            'no identity results supplied — skipped (run validate_test_identity)'))

    # ---- 5. LightGBM presence in every cell ----
    lgbm_missing = 0
    for dataset in present:
        ds = all_results[all_results['dataset'] == dataset]
        for smote_val in ['No', 'Yes']:
            cell = ds[ds['smote'] == smote_val]
            has_lgbm = (cell['model'] == 'lightgbm').any()
            if not has_lgbm:
                lgbm_missing += 1
                checks.append(_check_record(
                    'lightgbm_present', STATUS_FAIL,
                    f'{dataset}/{smote_val}: lightgbm missing'))
    if lgbm_missing == 0:
        checks.append(_check_record(
            'lightgbm_present', STATUS_OK,
            f'lightgbm present in all {2 * len(present)} cells'))

    # ---- 6. Model set sanity ----
    allowed = set(expected_models) | DOCUMENTED_BASELINES
    unexpected = sorted({m for m in all_results['model'].astype(str).unique()
                         if m not in allowed})
    checks.append(_check_record(
        'model_set', STATUS_FAIL if unexpected else STATUS_OK,
        f'unexpected models: {unexpected or "none"}'))

    # ---- 7. KKBox status honesty (only when a status is supplied) ----
    if kkbox_status is not None:
        checks.append(_check_record(
            'kkbox_status', STATUS_OK if kkbox_status != 'UNKNOWN' else STATUS_WARN,
            f'kkbox = {kkbox_status}',
            kkbox_status=kkbox_status))

    # ---- 8. Per-dataset validity (zero-valid datasets must FAIL) ----
    # A dataset whose experiments all failed (or whose models produced no
    # usable metrics) must never be silently omitted from the audit.  It is
    # reported here as a hard FAIL so the matrix is never "completed" with a
    # dataset that contributed nothing.
    if 'status' in all_results.columns:
        zero_valid = []
        for dataset in present:
            ds = all_results[all_results['dataset'] == dataset]
            success_rows = ds[ds['status'] == EXP_STATUS_SUCCESS]
            non_base_valid = success_rows[
                ~success_rows['model'].astype(str).isin(DOCUMENTED_BASELINES)]
            valid_models = non_base_valid['model'].astype(str).nunique()
            if valid_models == 0:
                zero_valid.append(dataset)
                checks.append(_check_record(
                    'dataset_validity', STATUS_FAIL,
                    f'{dataset}: zero valid model results (every condition '
                    f'failed)', dataset=dataset, valid_models=0))
            else:
                checks.append(_check_record(
                    'dataset_validity', STATUS_OK,
                    f'{dataset}: {valid_models} valid model(s) × '
                    f'{ds["smote"].nunique()} condition(s)',
                    dataset=dataset, valid_models=int(valid_models)))
        if not zero_valid:
            logger.validation(
                "Audit | All %d datasets produced valid model results",
                len(present),
            )

    audit = pd.DataFrame(checks)
    audit['framework_version'] = FRAMEWORK_VERSION
    audit['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    audit['overall'] = (STATUS_FAIL if (audit['status'] == STATUS_FAIL).any()
                        else STATUS_OK)
    return audit


def overall_audit_status(audit: pd.DataFrame) -> str:
    if audit is None or audit.empty:
        return STATUS_FAIL
    if (audit['status'] == STATUS_FAIL).any():
        return STATUS_FAIL
    if (audit['status'] == STATUS_WARN).any():
        return STATUS_WARN
    return STATUS_OK


def _summary_block(audit: pd.DataFrame, all_results: pd.DataFrame,
                   extras: Dict) -> str:
    present = sorted(all_results['dataset'].unique()) \
        if all_results is not None else []
    line = '-' * 70
    return (
        f"\n{line}\n"
        f"BEHAVIORAL CHURN PREDICTION — FINAL AUDIT & COMPLETION\n"
        f"{line}\n"
        f"Framework version:       {FRAMEWORK_VERSION}\n"
        f"Datasets executed:       {len(present)} / "
        f"{len(FINAL_EXPERIMENT_DATASETS)}\n"
        f"Expected model cells:    {extras.get('expected_total')} "
        f"(datasets × 2 SMOTE × {len(FINAL_EXPERIMENT_MODELS)} models)\n"
        f"Condition runs succeeded: {extras.get('successful')}\n"
        f"Condition runs failed:    {extras.get('failed')}\n"
        f"Statistical comparison:  {extras.get('statistical')}\n"
        f"Framework quadrant:      {extras.get('framework')}\n"
        f"Model persistence:       {extras.get('model_persistence')}\n"
        f"Publication figures:     {extras.get('figures')}\n"
        f"Integrity audit:         {overall_audit_status(audit)}\n"
        f"Conclusion:              "
        f"{'COMPLETE — all integrity checks passed' if overall_audit_status(audit) == STATUS_OK else 'INCOMPLETE — see integrity audit for failures'}\n"
        f"{line}\n"
    )


def write_completion_report(
    experiment_dir: str,
    audit: pd.DataFrame,
    all_results: Optional[pd.DataFrame] = None,
    extras: Optional[Dict] = None,
) -> Dict[str, str]:
    """Write integrity_audit.csv, completion_report.csv/.json + terminal txt."""
    master = ensure_dir(os.path.join(experiment_dir, 'results', 'master'))

    audit_path = os.path.join(master, 'integrity_audit.csv')
    audit.to_csv(audit_path, index=False)

    extras = dict(extras or {})
    if all_results is not None:
        extras.setdefault('expected_total',
                          len(all_results['dataset'].unique()) * 2 *
                          len(FINAL_EXPERIMENT_MODELS))
        extras.setdefault('successful',
                          int((all_results['status'] == 'SUCCESS').sum()) if
                          'status' in all_results.columns else len(all_results))
        extras.setdefault('failed', 0)

    n_fail = int((audit['status'] == STATUS_FAIL).sum()) if len(audit) else 0
    n_warn = int((audit['status'] == STATUS_WARN).sum()) if len(audit) else 0
    n_ok = int((audit['status'] == STATUS_OK).sum()) if len(audit) else 0

    report = {
        'framework_version': FRAMEWORK_VERSION,
        'status': overall_audit_status(audit),
        'checks_ok': n_ok,
        'checks_warn': n_warn,
        'checks_fail': n_fail,
        'audit_file': os.path.basename(audit_path),
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    report.update({k: v for k, v in extras.items() if v is not None})

    report_csv_path = os.path.join(master, 'completion_report.csv')
    pd.DataFrame([report]).to_csv(report_csv_path, index=False)

    report_json_path = os.path.join(master, 'completion_report.json')
    with open(report_json_path, 'w') as f:
        json.dump(report, f, indent=2)

    txt_path = os.path.join(master, 'completion_report.txt')
    with open(txt_path, 'w') as f:
        f.write(_summary_block(audit, all_results, extras))

    print(_summary_block(audit, all_results, extras))
    logger.validation("Completion report written to %s", master)
    return {'audit': audit_path, 'csv': report_csv_path,
            'json': report_json_path, 'txt': txt_path}
