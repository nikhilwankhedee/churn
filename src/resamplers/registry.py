"""
Resampler registry.

Provides a centralized registry for resampling strategies, building on
the core PluginRegistry. Includes convenience functions for registering
built-in resamplers and retrieving them by name.

Usage:
    from src.resamplers.registry import get_resampler, list_resamplers

    resampler = get_resampler("smote")
    result = resampler.resample(X_train, y_train)
"""
from typing import Dict, List, Optional

from src.resamplers.base import Resampler, ResampleResult
from src.core.registry import registry


_CATEGORY = "resamplers"


def register_resampler(
    name: str,
    dotted_path: str,
    metadata: Optional[dict] = None,
) -> None:
    """Register a resampler by lazy dotted path."""
    registry.register(name, _CATEGORY, dotted_path, metadata)


def register_resampler_class(
    name: str,
    cls: type,
    metadata: Optional[dict] = None,
) -> None:
    """Register an already-imported resampler class."""
    registry.register_class(name, _CATEGORY, cls, metadata)


def get_resampler(name: str) -> Resampler:
    """Retrieve a registered resampler instance."""
    return registry.get_instance(name, _CATEGORY)


def list_resamplers() -> List[str]:
    """Return all registered resampler names."""
    return registry.list_registered(_CATEGORY)


def _register_builtins() -> None:
    """Register the built-in resamplers (lazy-loaded)."""
    _base = "src.resamplers"
    builtins = {
        "smote": f"{_base}.smote.SmoteResampler",
        "adasyn": f"{_base}.adasyn.AdasynResampler",
    }
    for name, path in builtins.items():
        if not registry.is_registered(name, _CATEGORY):
            registry.register(
                name, _CATEGORY, path,
                metadata={"builtin": True},
            )


# Auto-register builtins on import
_register_builtins()
