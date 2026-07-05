"""Tests for churn strategies, models, and metrics registries."""
import os
import sys
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestChurnStrategies:
    """Tests for churn labeling strategies."""

    def test_list_strategies(self):
        from src.churn import list_strategies
        strats = list_strategies()
        assert isinstance(strats, list)
        assert len(strats) >= 3
        assert "inactivity" in strats
        assert "subscription" in strats
        assert "cadence" in strats

    def test_get_strategy(self):
        from src.churn import get_churn_strategy
        for name in ["inactivity", "subscription", "cadence"]:
            s = get_churn_strategy(name)
            assert hasattr(s, "description")
            assert hasattr(s, "required_columns")

    def test_strategy_has_required_columns(self):
        from src.churn import get_churn_strategy
        s = get_churn_strategy("inactivity")
        assert "customer_id" in s.required_columns
        assert "event_time" in s.required_columns


class TestModels:
    """Tests for model wrappers."""

    def test_list_models(self):
        from src.models import list_models
        models = list_models()
        assert isinstance(models, list)
        assert len(models) >= 3
        assert "logistic_regression" in models
        assert "random_forest" in models
        assert "xgboost" in models

    def test_get_model(self):
        from src.models import get_model
        for name in ["logistic_regression", "random_forest", "xgboost"]:
            m = get_model(name)
            assert hasattr(m, "description")


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_list_metrics(self):
        from src.metrics import list_metrics
        metrics = list_metrics()
        assert isinstance(metrics, list)
        assert len(metrics) >= 8

    def test_get_metric(self):
        from src.metrics import get_metric
        for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            m = get_metric(name)
            assert hasattr(m, "description")
            assert hasattr(m, "higher_is_better")

    def test_metric_properties(self):
        from src.metrics import get_metric
        acc = get_metric("accuracy")
        assert acc.higher_is_better is True
        brier = get_metric("brier_score")
        assert brier.higher_is_better is False

    def test_evaluate_with_all(self):
        import numpy as np
        from src.metrics import evaluate_with_all
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0])
        y_proba = np.array([0.1, 0.6, 0.9, 0.8, 0.2, 0.4, 0.3, 0.7, 0.85, 0.15])
        results = evaluate_with_all(y_true, y_pred, y_proba)
        assert isinstance(results, dict)
        assert "accuracy" in results
        assert "f1" in results
        # Results are MetricResult objects, access .value
        assert 0 <= results["accuracy"].value <= 1


class TestResamplers:
    """Tests for resampler registry."""

    def test_list_resamplers(self):
        from src.resamplers import list_resamplers
        resamplers = list_resamplers()
        assert isinstance(resamplers, list)
        assert "smote" in resamplers
        assert "adasyn" in resamplers

    def test_get_resampler(self):
        from src.resamplers import get_resampler
        smote = get_resampler("smote")
        assert hasattr(smote, "description")
        assert smote.requires_imbalanced_learn is True


class TestSmoteSingleClassGuard:
    """apply_smote must never call SMOTE with k_neighbors=-1 on a
    single-class training set; it returns the data unchanged instead."""

    def _run(self, n_pos, n_neg):
        import numpy as np
        import pandas as pd
        from src.smote.resampler import apply_smote
        X = pd.DataFrame({'f1': np.arange(n_pos + n_neg, dtype=float),
                          'f2': np.arange(n_pos + n_neg, dtype=float)})
        y = pd.Series([1] * n_pos + [0] * n_neg, dtype=int)
        return apply_smote(X, y, k_neighbors=5)

    def test_single_class_all_positive(self):
        res = self._run(n_pos=50, n_neg=0)
        assert res.n_synthetic == 0
        assert len(res.X_resampled) == 50

    def test_single_class_all_negative(self):
        res = self._run(n_pos=0, n_neg=50)
        assert res.n_synthetic == 0
        assert len(res.X_resampled) == 50

    def test_single_positive_sample(self):
        res = self._run(n_pos=1, n_neg=50)
        assert res.n_synthetic == 0
        assert len(res.X_resampled) == 51

    def test_two_classes_resamples(self):
        res = self._run(n_pos=10, n_neg=50)
        assert res.n_synthetic > 0
        assert len(res.X_resampled) > 60

    def test_k_neighbors_never_below_one(self):
        import numpy as np
        import pandas as pd
        from src.smote.resampler import apply_smote
        X = pd.DataFrame({'f1': np.arange(6.0), 'f2': np.arange(6.0)})
        y = pd.Series([1, 1, 1, 1, 1, 0], dtype=int)
        res = apply_smote(X, y, k_neighbors=5)
        assert res.n_synthetic >= 0
