"""
Calibration curve plotting with bootstrapped confidence intervals.

Gracefully handles extreme class imbalance and small bin sizes.
All figures are closed after writing to prevent memory leaks.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.utils import resample

from src.config import FIGURES_DIR, CALIBRATION_N_BINS, CALIBRATION_N_BOOTSTRAP
from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)


def plot_calibration_curves(
    prob_dict: dict,
    y_test: np.ndarray,
    n_bins: int = CALIBRATION_N_BINS,
    n_bootstrap: int = CALIBRATION_N_BOOTSTRAP,
    save_path: str = None,
    suffix: str = '',
) -> None:
    ensure_dir(os.path.join(FIGURES_DIR, 'calibration'))
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, 'calibration',
                                 f'calibration_curves{suffix}.png')

    fig, ax = plt.subplots(figsize=(8, 6))
    y_test_arr = np.asarray(y_test)

    for name, probs in prob_dict.items():
        probs = np.asarray(probs)
        try:
            prob_true, prob_pred = calibration_curve(
                y_test_arr, probs, n_bins=n_bins, strategy='uniform',
            )
        except Exception as exc:
            logger.warning("Calibration curve failed for %s: %s", name, exc)
            continue

        if len(prob_true) == 0:
            continue

        ax.plot(prob_pred, prob_true, marker='o', label=name)

        try:
            boot_true = []
            for _ in range(n_bootstrap):
                idx = resample(
                    np.arange(len(y_test_arr)), replace=True,
                    n_samples=len(y_test_arr),
                )
                bt, _ = calibration_curve(
                    y_test_arr[idx], probs[idx],
                    n_bins=n_bins, strategy='uniform',
                )
                # Pad shorter curves with NaN so stacking works
                if len(bt) < len(prob_true):
                    bt = np.pad(bt, (0, len(prob_true) - len(bt)),
                                constant_values=np.nan)
                elif len(bt) > len(prob_true):
                    bt = bt[:len(prob_true)]
                boot_true.append(bt)

            if boot_true:
                boot_arr = np.array(boot_true)
                with np.errstate(invalid='ignore'):
                    lower = np.nanpercentile(boot_arr, 2.5, axis=0)
                    upper = np.nanpercentile(boot_arr, 97.5, axis=0)
                ax.fill_between(prob_pred, lower, upper, alpha=0.15)
        except Exception as exc:
            logger.debug("Bootstrap CI failed for %s: %s", name, exc)

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Curves with 95% CI')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("Calibration curves saved to %s", save_path)
