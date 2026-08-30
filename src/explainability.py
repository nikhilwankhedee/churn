"""
SHAP-based model explainability.

Uses TreeExplainer for tree-based models and LinearExplainer for linear models.
Explainer selection is automatic based on the model type.  All SHAP calls are
wrapped in try/except so failures are logged but never halt the pipeline or
suppress metrics export.

Known behaviour:
    - Logistic Regression      -> LinearExplainer  (requires background sample)
    - Random Forest / XGBoost  -> TreeExplainer
    - LightGBM                 -> TreeExplainer
    - SVM (CalibratedClassifierCV) -> skipped (documented limitation)

Note: ``shap.summary_plot`` is called WITHOUT the ``ax=`` argument because the
Kaggle-bundled SHAP version routes it to ``summary_legacy()`` which does not
support that argument.  Figures are saved via ``plt.savefig`` on the active
current figure instead.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Any

from src.config import SHAP_SAMPLE_SIZE, RANDOM_SEED
from src.run_context import figures_dir
from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning("shap not installed — explainability disabled")


def get_shap_explainer(model: object, X_background: pd.DataFrame):
    """Return the appropriate SHAP explainer for a model, or None to skip.

    Model-type detection is name based and covers the models trained by the
    pipeline:
        - tree ensembles -> TreeExplainer
        - linear models   -> LinearExplainer (needs a background sample)
        - calibrated SVM  -> skipped
        - anything else   -> skipped gracefully
    """
    if not _SHAP_AVAILABLE:
        return None

    model_type = type(model).__name__.lower()

    # Tree-based models
    if any(name in model_type for name in [
        'forest', 'xgb', 'lgbm', 'lightgbm', 'gradient', 'tree', 'catboost',
    ]):
        return shap.TreeExplainer(model)

    # Linear models
    elif any(name in model_type for name in [
        'logistic', 'linear', 'ridge', 'lasso',
    ]):
        return shap.LinearExplainer(model, X_background)

    # CalibratedClassifierCV wrapping LinearSVC (SVM)
    elif 'calibrated' in model_type:
        return None

    # Fallback — skip gracefully
    else:
        return None


def compute_shap_values(
    model: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
):
    """Compute SHAP values for a model, or None when unsupported/failed."""
    if not _SHAP_AVAILABLE:
        logger.warning("SHAP not available — skipping %s", model_name)
        return None

    try:
        background = X_train.sample(min(100, len(X_train)), random_state=42)
        explainer = get_shap_explainer(model, background)

        if explainer is None:
            # Explicit SVM handling — documented behaviour, not an error.
            if 'calibrated' in type(model).__name__.lower():
                logger.info(
                    "SHAP skipped for SVM (CalibratedClassifierCV) — "
                    "TreeExplainer and LinearExplainer not compatible. "
                    "KernelExplainer excluded due to computational cost at "
                    "dataset scale."
                )
            else:
                logger.info(
                    "SHAP skipped for %s — unsupported model type", model_name
                )
            return None

        shap_values = explainer.shap_values(X_test)

        # Handle multi-class / list output from TreeExplainer.
        if isinstance(shap_values, list):
            # Prefer the positive (churned) class when binary.
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        return shap_values

    except Exception as exc:
        logger.warning(
            "SHAP computation failed for %s: %s", model_name, exc,
        )
        return None


def save_shap_plot(
    shap_values: Any,
    X: pd.DataFrame,
    output_path: str,
    model_name: str,
    plot_type: Optional[str] = None,
    max_display: int = 20,
) -> bool:
    """Render and save a SHAP summary plot with graceful degradation.

    Returns True on success, False otherwise.  Never raises.
    """
    if shap_values is None:
        logger.info(
            "No SHAP values available for %s — skipping plot", model_name,
        )
        return False

    ax_kwargs = {}
    if plot_type is not None:
        ax_kwargs['plot_type'] = plot_type
    if max_display is not None:
        ax_kwargs['max_display'] = max_display

    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X, show=False, **ax_kwargs,
        )
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close('all')
        logger.info("SHAP plot saved: %s", output_path)
        return True
    except Exception as exc:
        logger.warning(
            "SHAP plot failed for %s: %s", model_name, exc,
        )
        plt.close('all')
        return False


def shap_analysis(
    model: object,
    X_sample: pd.DataFrame,
    model_name: str,
    background: Optional[pd.DataFrame] = None,
    max_display: int = 20,
    suffix: str = '',
) -> Tuple[Optional[Any], Optional[Any]]:
    """Run SHAP analysis for a model and save all plots + return values.

    Gracefully degrades: any failure is logged and None is returned so the
    calling pipeline continues and metrics export is never suppressed.

    Parameters
    ----------
    model : object
        Trained classifier.
    X_sample : pd.DataFrame
        Test feature sample used for SHAP computation.
    model_name : str
        Model key used in file names and logs.
    background : pd.DataFrame, optional
        Background sample for LinearExplainer.  When None, a sample is drawn
        from ``X_sample``.
    max_display : int
        Number of top features to display.
    suffix : str
        Output file-name suffix (e.g. ``_smote``).

    Returns
    -------
    (shap_values, explainer) tuple, or (None, None) when SHAP is unavailable,
    unsupported, or the computation/plotting failed.
    """
    if not _SHAP_AVAILABLE:
        logger.warning("SHAP not available — skipping %s", model_name)
        return None, None

    # Explicit SVM skip — documented behaviour, not an error.
    if 'calibrated' in type(model).__name__.lower():
        logger.info(
            "SHAP skipped for SVM (CalibratedClassifierCV) — "
            "TreeExplainer and LinearExplainer not compatible. "
            "KernelExplainer excluded due to computational cost at "
            "dataset scale."
        )
        return None, None

    save_dir = figures_dir('shap_analysis')
    X_s = X_sample.copy()
    n_features = X_s.shape[1]

    # Use the provided background or derive one for linear models.
    if background is None:
        background = X_s.sample(
            min(100, len(X_s)), random_state=RANDOM_SEED,
        )

    try:
        explainer = get_shap_explainer(model, background)
        if explainer is None:
            logger.info(
                "SHAP skipped for %s — unsupported model type", model_name,
            )
            return None, None

        shap_values = explainer.shap_values(X_s)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(-1, 1)

        if shap_values.shape[1] != n_features:
            logger.warning(
                "SHAP shape mismatch: values %s vs features %d — truncating",
                shap_values.shape, n_features,
            )
            min_dim = min(shap_values.shape[1], n_features)
            shap_values = shap_values[:, :min_dim]
            X_s = X_s.iloc[:, :min_dim]

        # Summary bar plot
        save_shap_plot(
            shap_values, X_s,
            os.path.join(save_dir, f'{model_name}_shap_bar{suffix}.png'),
            model_name, plot_type="bar", max_display=max_display,
        )

        # Summary dot plot
        save_shap_plot(
            shap_values, X_s,
            os.path.join(save_dir, f'{model_name}_shap_summary{suffix}.png'),
            model_name, plot_type=None, max_display=max_display,
        )

        # Dependence plots for top 3 features (best-effort)
        try:
            top3 = np.argsort(np.abs(shap_values).mean(0))[-3:]
            for idx in top3:
                if idx >= X_s.shape[1]:
                    continue
                try:
                    plt.figure()
                    shap.dependence_plot(idx, shap_values, X_s, show=False)
                    fname = f'{model_name}_dependence_{X_s.columns[idx]}{suffix}.png'
                    plt.savefig(
                        os.path.join(save_dir, fname),
                        bbox_inches='tight', dpi=150,
                    )
                    plt.close('all')
                except Exception as exc:
                    logger.debug(
                        "SHAP dependence plot %s failed for %s: %s",
                        idx, model_name, exc,
                    )
                    plt.close('all')
        except Exception as exc:
            logger.debug(
                "SHAP dependence plots failed for %s: %s", model_name, exc,
            )

        logger.info("SHAP analysis complete for %s", model_name)
        return shap_values, explainer

    except Exception as exc:
        logger.warning("SHAP analysis failed for %s: %s", model_name, exc)
        plt.close('all')
        return None, None
