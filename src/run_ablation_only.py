"""
Ablation-only runner for the native-split datasets (Credit Card, Telco).

The main-model experiments for these two datasets are already valid and must
NOT be rerun (they are preserved as-is).  Only their ABLATION was invalid
(previous runs removed zero columns and emitted byte-identical ROC-AUC).
This runner rebuilds the same native stratified train matrix the main models
used and recomputes ONLY the ablation — each group now removes real columns
and is refit from scratch (no cached predictions).  The main model results,
metrics tables and figures are left untouched.

Usage
-----
    python -m src.run_ablation_only credit_card telco
    python -m src.run_ablation_only --with-smote credit_card   # ablation on SMOTE train matrix
"""
import sys

import pandas as pd

from src.config import RANDOM_SEED
from src.datasets import get_dataset
from src.run_context import set_run_scope, results_dir
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def run_ablation_only(dataset: str, use_smote: bool = False) -> pd.DataFrame:
    """Recompute the ablation for a native-split dataset.

    Builds the same train matrix used by the (already-valid) main run via
    ``build_native_modeling_data`` and reruns the group ablation with the
    dataset-aware group→column mapping.  The main experiment artefacts are
    not touched — only ``results/<dataset>/original/ablation/`` (and the
    matching figure) are refreshed.
    """
    set_seed(RANDOM_SEED)
    set_run_scope(dataset, use_smote=use_smote)

    adapter = get_dataset(dataset)
    if not hasattr(adapter, 'build_native_modeling_data'):
        raise RuntimeError(
            f"run_ablation_only supports native-split datasets only; "
            f"'{dataset}' does not implement build_native_modeling_data."
        )

    logger.info("── Ablation-only: %s ──", dataset)
    df = adapter.load_raw_data()
    df = adapter.preprocess(df)
    df = adapter.standardize_schema(df)
    X_train, X_test, y_train, y_test = adapter.build_native_modeling_data(df)

    if use_smote:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=RANDOM_SEED)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_train, columns=X_test.columns)
        y_train = pd.Series(y_train, name='churn')

    logger.validation(
        "Ablation-only | train: %d rows x %d cols | churn: %.2f%%",
        len(X_train), X_train.shape[1], y_train.mean() * 100,
    )

    from src.ablation import run_ablation
    ablation_df = run_ablation(
        X_train, y_train, dataset_name=dataset,
    )

    suffix = '_smote' if use_smote else ''
    out_csv = results_dir('ablation', f'ablation_results{suffix}.csv')
    ablation_df.to_csv(out_csv, index=False)
    logger.validation("Ablation-only results saved: %s", out_csv)

    try:
        from src.visualization import plot_ablation_results
        plot_ablation_results(ablation_df, suffix=suffix)
    except Exception as exc:
        logger.warning("Ablation-only plot failed: %s", exc)

    return ablation_df


def main(argv=None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    smote = '--with-smote' in args
    datasets = [a for a in args if not a.startswith('--')]
    if not datasets:
        datasets = ['credit_card', 'telco']
    for ds in datasets:
        run_ablation_only(ds, use_smote=smote)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())