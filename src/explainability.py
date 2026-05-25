"""
SHAP-based model explainability.

Uses TreeExplainer for tree-based models and LinearExplainer for linear models.
All SHAP calls are wrapped in try/except so failures are logged but do not
block the pipeline.  No deprecated APIs are used.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Any

from src.config import FIGURES_DIR, SHAP_SAMPLE_SIZE, RANDOM_SEED
from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning("shap not installed — explainability disabled")


def shap_analysis(
    model: object,
    X_sample: pd.DataFrame,
    model_name: str,
    background: Optional[pd.DataFrame] = None,
    max_display: int = 20,
) -> Tuple[Optional[Any], Optional[Any]]:
    if not _SHAP_AVAILABLE:
        logger.warning("SHAP not available — skipping %s", model_name)
        return None, None

    save_dir = ensure_dir(os.path.join(FIGURES_DIR, 'shap_analysis'))
    X_s = X_sample.copy()
    n_features = X_s.shape[1]

    try:
        is_tree = hasattr(model, 'get_booster') or hasattr(model, 'feature_importances_')
        if is_tree:
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, 'coef_'):
            if background is None:
                background = X_s
            explainer = shap.LinearExplainer(model, background)
        else:
            explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_s)

        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
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
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_s, plot_type="bar", show=False,
            max_display=max_display, ax=ax,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(save_dir, f'{model_name}_shap_bar.png'),
            dpi=300, bbox_inches='tight',
        )
        plt.close(fig)

        # Summary dot plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_s, show=False,
            max_display=max_display, ax=ax,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(save_dir, f'{model_name}_shap_summary.png'),
            dpi=300, bbox_inches='tight',
        )
        plt.close(fig)

        # Dependence plots for top 3 features
        try:
            top3 = np.argsort(np.abs(shap_values).mean(0))[-3:]
            for idx in top3:
                if idx >= X_s.shape[1]:
                    continue
                fig, ax = plt.subplots()
                shap.dependence_plot(
                    idx, shap_values, X_s, show=False, ax=ax,
                )
                fig.tight_layout()
                fname = f'{model_name}_dependence_{X_s.columns[idx]}.png'
                fig.savefig(
                    os.path.join(save_dir, fname),
                    dpi=300, bbox_inches='tight',
                )
                plt.close(fig)
        except Exception as exc:
            logger.debug("SHAP dependence plots failed for %s: %s",
                          model_name, exc)

        logger.info("SHAP analysis complete for %s", model_name)
        return shap_values, explainer

    except Exception as exc:
        logger.warning("SHAP analysis failed for %s: %s", model_name, exc)
        return None, None
