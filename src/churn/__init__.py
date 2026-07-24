"""
Churn labeling strategies.

This package provides a pluggable churn labeling system with:
- Abstract base class (ChurnStrategy)
- Built-in strategies (inactivity, subscription, cadence)
- Plugin registry for custom strategies

Usage:
    from src.churn import get_churn_strategy, list_strategies

    strategy = get_churn_strategy("inactivity")
    result = strategy.label(df, cutoff_date)
"""
from src.churn.base import ChurnStrategy, ChurnResult
from src.churn.registry import (
    get_churn_strategy,
    list_strategies,
    register_churn_strategy,
    register_churn_strategy_class,
)

__all__ = [
    "ChurnStrategy",
    "ChurnResult",
    "get_churn_strategy",
    "list_strategies",
    "register_churn_strategy",
    "register_churn_strategy_class",
]
