"""
Resampler registry and built-in implementations.

This package provides a pluggable resampling system with:
- Abstract base class (Resampler)
- Built-in resamplers (SMOTE, ADASYN)
- Plugin registry for custom resamplers

Usage:
    from src.resamplers import get_resampler, list_resamplers

    resampler = get_resampler("smote")
    result = resampler.resample(X_train, y_train)
"""
from src.resamplers.base import Resampler, ResampleResult
from src.resamplers.registry import (
    get_resampler,
    list_resamplers,
    register_resampler,
    register_resampler_class,
)

__all__ = [
    "Resampler",
    "ResampleResult",
    "get_resampler",
    "list_resamplers",
    "register_resampler",
    "register_resampler_class",
]
