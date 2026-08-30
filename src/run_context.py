"""
Active-run output scoping.

All artefact writers resolve their output directories through this module,
so that every dataset+mode combination lands in its own folder tree:

    results/<dataset>/<mode>/...
    figures/<dataset>/<mode>/...
    models/<dataset>/<mode>/...
    processed_data/<dataset>/<mode>/...

where ``mode`` is "original" or "smote".  Cross-dataset artefacts (the
master results table) stay level with the run scope so the original and
SMOTE sweeps produce two separate master tables.
"""
import os

from src.config import (
    RESULTS_DIR, FIGURES_DIR, MODELS_DIR, PROCESSED_DIR,
)
from src.utils import ensure_dir

_ACTIVE = {
    'dataset': 'none',
    'mode': 'original',
}


def set_run_scope(dataset: str, use_smote: bool) -> None:
    """Record the currently running dataset and mode ('original'|'smote')."""
    _ACTIVE['dataset'] = dataset
    _ACTIVE['mode'] = 'smote' if use_smote else 'original'


def active_dataset() -> str:
    return _ACTIVE['dataset']


def active_mode() -> str:
    return _ACTIVE['mode']


def _scoped(root: str, *parts: str) -> str:
    path = os.path.join(root, _ACTIVE['dataset'], _ACTIVE['mode'], *parts)
    return ensure_dir(path)


def results_dir(*parts: str) -> str:
    """Scoped results directory: results/<dataset>/<mode>/[parts]."""
    return _scoped(RESULTS_DIR, *parts)


def figures_dir(*parts: str) -> str:
    """Scoped figures directory: figures/<dataset>/<mode>/[parts]."""
    return _scoped(FIGURES_DIR, *parts)


def models_dir(*parts: str) -> str:
    """Scoped models directory: models/<dataset>/<mode>/[parts]."""
    return _scoped(MODELS_DIR, *parts)


def processed_dir(*parts: str) -> str:
    """Scoped processed-data directory:
    processed_data/<dataset>/<mode>/[parts]."""
    return _scoped(PROCESSED_DIR, *parts)


def master_results_path() -> str:
    """Path of the cross-dataset master table for the current mode."""
    d = ensure_dir(os.path.join(RESULTS_DIR, 'cross_dataset'))
    return os.path.join(d, f'master_results_{_ACTIVE["mode"]}.csv')