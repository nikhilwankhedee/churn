"""
SMOTE resampling with temporal safety.

This package provides controlled SMOTE application that:
- Only resamples training data (never test data)
- Validates no test data leakage via temporal assertions
- Is gated by configuration (disabled by default)

Usage:
    from src.smote import apply_smote

    result = apply_smote(X_train, y_train, X_test, y_test)
    X_resampled = result.X_resampled
    y_resampled = result.y_resampled
"""
from src.smote.resampler import apply_smote, SmoteResult

__all__ = ["apply_smote", "SmoteResult"]
