"""Tests for the Python API (ChurnFramework)."""
import os
import sys
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestChurnFramework:
    """Tests for src.api.ChurnFramework."""

    def test_init(self):
        from src.api import ChurnFramework
        fw = ChurnFramework()
        assert fw is not None

    def test_list_datasets(self):
        from src.api import ChurnFramework
        fw = ChurnFramework()
        ds = fw.list_datasets()
        assert "olist" in ds
        assert len(ds) >= 6

    def test_list_models(self):
        from src.api import ChurnFramework
        fw = ChurnFramework()
        models = fw.list_models()
        assert "logistic_regression" in models

    def test_list_metrics(self):
        from src.api import ChurnFramework
        fw = ChurnFramework()
        metrics = fw.list_metrics()
        assert "accuracy" in metrics

    def test_list_strategies(self):
        from src.api import ChurnFramework
        fw = ChurnFramework()
        strats = fw.list_strategies()
        assert "inactivity" in strats

    def test_list_resamplers(self):
        from src.api import ChurnFramework
        fw = ChurnFramework()
        resamplers = fw.list_resamplers()
        assert "smote" in resamplers

    def test_list_reports(self):
        from src.api import ChurnFramework
        fw = ChurnFramework()
        reports = fw.list_reports()
        assert isinstance(reports, list)

    def test_doctor(self):
        from src.api import ChurnFramework
        fw = ChurnFramework()
        checks = fw.doctor()
        assert "core_registry" in checks
        assert checks["core_registry"]["ok"] is True
        assert "datasets" in checks
        assert checks["datasets"]["ok"] is True

    def test_validate_config(self, tmp_path):
        from src.api import ChurnFramework
        fw = ChurnFramework()
        cfg = tmp_path / "test.yaml"
        cfg.write_text("dataset:\n  name: test\n  ecosystem_type: transactional_marketplace\n")
        result = fw.validate_config(str(cfg))
        assert result["is_valid"] is True

    def test_list_experiments(self):
        from src.api import ChurnFramework
        fw = ChurnFramework()
        experiments = fw.list_experiments()
        assert isinstance(experiments, list)

    def test_profile_csv(self, tmp_path):
        import pandas as pd
        from src.api import ChurnFramework
        fw = ChurnFramework()

        df = pd.DataFrame({
            "customer_id": range(50),
            "order_date": pd.date_range("2023-01-01", periods=50),
            "amount": [10.0, 20.0, 15.0] * 16 + [10.0, 20.0],
        })
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)

        result = fw.profile_csv(str(path))
        assert result["n_rows"] == 50
        assert result["n_columns"] == 3
