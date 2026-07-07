#!/usr/bin/env python3
"""
Example 05: Custom Model

Demonstrates how to create a custom model wrapper
that integrates with the framework's registry.

Usage:
    cd project_root
    python examples/05_custom_model.py
"""
import sys
import os
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.base import ModelWrapper, ModelResult
from src.models.registry import register_model_class


class SimpleMajorityClassifier(ModelWrapper):
    """A simple baseline that always predicts the majority class.

    This demonstrates how to create a custom model that integrates
    with the framework's registry and evaluation pipeline.
    """

    @property
    def name(self) -> str:
        return "majority_baseline"

    @property
    def default_hyperparameters(self) -> dict:
        return {}

    def _create_estimator(self, params: dict):
        """Return a simple majority class estimator."""
        return _MajorityEstimator()


class _MajorityEstimator:
    """Internal estimator that predicts the majority class."""

    def fit(self, X, y):
        self._majority_class = int(np.bincount(y.astype(int)).argmax())
        return self

    def predict(self, X):
        return np.full(len(X), self._majority_class)

    def predict_proba(self, X):
        proba = np.zeros((len(X), 2))
        proba[:, self._majority_class] = 1.0
        return proba


def main():
    print("Custom Model Example")
    print("=" * 60)

    # Create the model wrapper
    wrapper = SimpleMajorityClassifier()

    print(f"Model name: {wrapper.name}")
    print(f"Description: {wrapper.description}")
    print(f"Hyperparameters: {wrapper.default_hyperparameters}")

    # Use the framework's fit method
    print("\n--- Training ---")
    np.random.seed(42)
    import pandas as pd
    X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
    y_train = pd.Series(np.array([0] * 70 + [1] * 30))
    X_test = pd.DataFrame(np.random.randn(20, 5), columns=[f"f{i}" for i in range(5)])

    result = wrapper.fit(X_train, y_train)

    print(f"Result model name: {result.model_name}")
    print(f"Fitted model type: {type(result.fitted_model).__name__}")

    predictions = wrapper.predict(result, X_test)
    probabilities = wrapper.predict_proba(result, X_test)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample predictions: {predictions[:5]}")

    # Register with the framework
    print("\n--- Registering with Framework ---")
    register_model_class("majority_baseline", SimpleMajorityClassifier)
    print("Registered 'majority_baseline' with model registry")

    print("\nTo use this model in the pipeline:")
    print("1. Add it to src/models/ directory")
    print("2. Register it in src/models/__init__.py")
    print("3. It will appear in 'churn models'")


if __name__ == "__main__":
    main()
