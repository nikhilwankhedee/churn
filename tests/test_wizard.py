"""Tests for the wizard (dataset registration) module."""
import os
import sys
import pandas as pd
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV for testing."""
    df = pd.DataFrame({
        "user_id": range(100),
        "order_date": pd.date_range("2023-01-01", periods=100, freq="D"),
        "payment_amount": [10.5, 20.0, 15.3, 30.0, 5.0] * 20,
        "product_category": ["A", "B", "C", "A", "B"] * 20,
        "review_score": [4, 5, 3, 4, 5] * 20,
    })
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestInspector:
    """Tests for src.wizard.inspector."""

    def test_inspect_csv_returns_result(self, sample_csv):
        from src.wizard.inspector import inspect_csv
        result = inspect_csv(sample_csv)
        assert result.n_rows == 100
        assert result.n_columns == 5

    def test_infers_customer_id(self, sample_csv):
        from src.wizard.inspector import inspect_csv
        result = inspect_csv(sample_csv)
        assert result.inferred_customer_id == "user_id"

    def test_infers_timestamp(self, sample_csv):
        from src.wizard.inspector import inspect_csv
        result = inspect_csv(sample_csv)
        assert result.inferred_event_time == "order_date"

    def test_infers_transaction_value(self, sample_csv):
        from src.wizard.inspector import inspect_csv
        result = inspect_csv(sample_csv)
        assert result.inferred_transaction_value == "payment_amount"

    def test_columns_have_roles(self, sample_csv):
        from src.wizard.inspector import inspect_csv
        result = inspect_csv(sample_csv)
        roles = [c.inferred_role for c in result.columns if c.inferred_role]
        assert len(roles) >= 3  # at least customer_id, timestamp, value

    def test_user_hints_override(self, sample_csv):
        from src.wizard.inspector import inspect_csv
        result = inspect_csv(sample_csv, customer_id_hint="product_category")
        assert result.inferred_customer_id == "product_category"

    def test_dataset_name_guess(self, sample_csv):
        from src.wizard.inspector import inspect_csv
        result = inspect_csv(sample_csv)
        assert isinstance(result.suggested_dataset_name, str)
        assert len(result.suggested_dataset_name) > 0

    def test_warnings_for_missing_roles(self, tmp_path):
        """CSV with no obvious customer ID should produce a warning."""
        from src.wizard.inspector import inspect_csv
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = tmp_path / "no_id.csv"
        df.to_csv(path, index=False)
        result = inspect_csv(str(path))
        assert any("customer ID" in w for w in result.warnings)


class TestGenerator:
    """Tests for src.wizard.generator."""

    def test_generate_config(self, sample_csv):
        from src.wizard.inspector import inspect_csv
        from src.wizard.generator import generate_config
        inspection = inspect_csv(sample_csv)
        config = generate_config(inspection, dataset_name="test_ds")
        assert config.dataset_name == "test_ds"
        assert config.ecosystem_type == "transactional_marketplace"
        assert "purchase" in config.available_feature_groups
        assert "customer_id" in config.column_mapping.values()

    def test_to_yaml(self, sample_csv):
        from src.wizard.inspector import inspect_csv
        from src.wizard.generator import generate_config
        inspection = inspect_csv(sample_csv)
        config = generate_config(inspection, dataset_name="test_ds")
        yaml_str = config.to_yaml()
        assert "dataset:" in yaml_str
        assert "churn:" in yaml_str
        assert "test_ds" in yaml_str

    def test_ecosystem_override(self, sample_csv):
        from src.wizard.inspector import inspect_csv
        from src.wizard.generator import generate_config
        inspection = inspect_csv(sample_csv)
        config = generate_config(inspection, ecosystem_type="subscription")
        assert config.ecosystem_type == "subscription"
        assert config.churn_strategy == "subscription"
