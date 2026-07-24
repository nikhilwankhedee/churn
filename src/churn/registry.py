"""
Churn strategy registry.

Provides a centralized registry for churn labeling strategies,
building on the core PluginRegistry. Includes convenience functions
for registering built-in strategies and retrieving them by name.

Usage:
    from src.churn.registry import get_churn_strategy, list_strategies

    strategy = get_churn_strategy("inactivity")
    result = strategy.label(df, cutoff_date)

    for name in list_strategies():
        print(name)
"""
from typing import List, Optional

from src.churn.base import ChurnStrategy, ChurnResult
from src.core.registry import registry


_CATEGORY = "churn_strategies"


def register_churn_strategy(
    name: str,
    dotted_path: str,
    metadata: Optional[dict] = None,
) -> None:
    """Register a churn strategy by lazy dotted path."""
    registry.register(name, _CATEGORY, dotted_path, metadata)


def register_churn_strategy_class(
    name: str,
    cls: type,
    metadata: Optional[dict] = None,
) -> None:
    """Register an already-imported churn strategy class."""
    registry.register_class(name, _CATEGORY, cls, metadata)


def get_churn_strategy(name: str) -> ChurnStrategy:
    """Retrieve a registered churn strategy instance."""
    return registry.get_instance(name, _CATEGORY)


def list_strategies() -> List[str]:
    """Return all registered churn strategy names."""
    return registry.list_registered(_CATEGORY)


def _register_builtins() -> None:
    """Register the three built-in churn strategies (lazy-loaded)."""
    _base = "src.churn"
    builtins = {
        "inactivity": f"{_base}.inactivity.InactivityStrategy",
        "subscription": f"{_base}.subscription.SubscriptionStrategy",
        "cadence": f"{_base}.cadence.CadenceStrategy",
    }
    for name, path in builtins.items():
        if not registry.is_registered(name, _CATEGORY):
            registry.register(
                name, _CATEGORY, path,
                metadata={"builtin": True},
            )


# Auto-register builtins on import
_register_builtins()
