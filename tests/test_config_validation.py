"""Tests for configuration validation."""
import os
import sys
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestConfigValidation:
    """Tests for src.config_validation."""

    def test_valid_config(self):
        from src.config_validation import validate_config
        config = {
            "dataset": {"name": "test", "ecosystem_type": "transactional_marketplace"},
            "churn": {"strategy": "inactivity", "prediction_window_days": 180},
            "features": {"available_groups": ["purchase", "monetary"]},
        }
        result = validate_config(config)
        assert result.is_valid
        assert result.error_count == 0

    def test_missing_dataset_section(self):
        from src.config_validation import validate_config
        result = validate_config({})
        assert not result.is_valid
        assert any("dataset" in e.path for e in result.errors)

    def test_invalid_ecosystem_type(self):
        from src.config_validation import validate_config
        config = {"dataset": {"name": "x", "ecosystem_type": "invalid_type"}}
        result = validate_config(config)
        assert result.is_valid  # warning, not error
        assert len(result.warnings) > 0

    def test_invalid_churn_strategy(self):
        from src.config_validation import validate_config
        config = {
            "dataset": {"name": "x"},
            "churn": {"strategy": "nonexistent"},
        }
        result = validate_config(config)
        assert not result.is_valid

    def test_negative_window(self):
        from src.config_validation import validate_config
        config = {
            "dataset": {"name": "x"},
            "churn": {"prediction_window_days": -10},
        }
        result = validate_config(config)
        assert not result.is_valid

    def test_invalid_feature_groups(self):
        from src.config_validation import validate_config
        config = {
            "dataset": {"name": "x"},
            "features": {"available_groups": ["purchase", "nonexistent_group"]},
        }
        result = validate_config(config)
        assert not result.is_valid

    def test_schema_mapping_missing_required(self):
        from src.config_validation import validate_config
        config = {
            "dataset": {"name": "x"},
            "schema": {"column_mapping": {"col_a": "transaction_value"}},
        }
        result = validate_config(config)
        # Should warn about missing customer_id and event_time
        assert len(result.warnings) > 0

    def test_validate_config_file(self, tmp_path):
        from src.config_validation import validate_config_file
        cfg = tmp_path / "test.yaml"
        cfg.write_text("dataset:\n  name: test\n")
        result = validate_config_file(str(cfg))
        assert result.is_valid

    def test_validate_nonexistent_file(self):
        from src.config_validation import validate_config_file
        result = validate_config_file("/nonexistent/path.yaml")
        assert not result.is_valid

    def test_validate_invalid_yaml(self, tmp_path):
        from src.config_validation import validate_config_file
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("{{invalid yaml: [")
        result = validate_config_file(str(cfg))
        assert not result.is_valid
