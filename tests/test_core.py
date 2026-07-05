"""Tests for core registry, context, and config modules."""
import os
import sys
import pytest

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestCoreRegistry:
    """Tests for src.core.registry."""

    def test_registry_import(self):
        from src.core.registry import registry
        assert registry is not None

    def test_register_and_get(self):
        from src.core.registry import PluginRegistry
        reg = PluginRegistry()

        reg.register("test_dummy", "test_cat", "builtins.int")
        assert reg.is_registered("test_dummy", "test_cat")
        cls = reg.get_class("test_dummy", "test_cat")
        assert cls is int

    def test_list_registered(self):
        from src.core.registry import PluginRegistry
        reg = PluginRegistry()

        reg.register("a", "things", "builtins.str")
        reg.register("b", "things", "builtins.float")
        names = reg.list_registered("things")
        assert "a" in names
        assert "b" in names

    def test_list_categories(self):
        from src.core.registry import PluginRegistry
        reg = PluginRegistry()

        reg.register("x", "cat1", "builtins.int")
        cats = reg.list_categories()
        assert "cat1" in cats

    def test_get_unregistered_raises_keyerror(self):
        from src.core.registry import PluginRegistry
        reg = PluginRegistry()
        with pytest.raises(KeyError):
            reg.get_class("nonexistent", "cat")


class TestPipelineContext:
    """Tests for src.core.context."""

    def test_context_creation(self):
        from src.core.context import PipelineContext
        ctx = PipelineContext(dataset="test")
        assert ctx.dataset == "test"

    def test_context_defaults(self):
        from src.core.context import PipelineContext
        ctx = PipelineContext(dataset="x")
        assert ctx.smote_enabled is False
        assert ctx.models is None


class TestConfig:
    """Tests for src.config."""

    def test_framework_version(self):
        from src.config import FRAMEWORK_VERSION
        assert isinstance(FRAMEWORK_VERSION, str)
        assert len(FRAMEWORK_VERSION) > 0

    def test_random_seed(self):
        from src.config import RANDOM_SEED
        assert RANDOM_SEED == 42

    def test_load_config(self, tmp_path):
        from src.config import load_config
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("dataset:\n  name: test\n")
        result = load_config(str(cfg_file))
        assert result["dataset"]["name"] == "test"

    def test_get_config_value(self, tmp_path):
        from src.config import load_config, get_config_value
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("churn:\n  strategy: inactivity\n  window: 180\n")
        load_config(str(cfg_file))
        assert get_config_value("churn.strategy") == "inactivity"
        assert get_config_value("churn.window") == 180
        assert get_config_value("churn.missing", "default") == "default"

    def test_feature_groups_defined(self):
        from src.config import STANDARD_FEATURE_GROUPS, FEATURE_GROUPS
        core_groups = [
            "purchase", "monetary", "inactivity", "review",
            "delivery", "payment", "engagement", "cadence",
        ]
        assert len(STANDARD_FEATURE_GROUPS) >= len(core_groups)
        for group in core_groups:
            assert group in STANDARD_FEATURE_GROUPS
        for group in STANDARD_FEATURE_GROUPS:
            assert group in FEATURE_GROUPS


class TestDatasetResolver:
    """Tests for src.dataset_resolver."""

    def test_explicit_data_dir(self, tmp_path):
        from src.dataset_resolver import resolve_dataset_directory
        import pandas as pd

        # Create a fake dataset directory
        ds_dir = tmp_path / "fake_dataset"
        ds_dir.mkdir()
        (ds_dir / "data.csv").write_text("a,b\n1,2\n")

        result = resolve_dataset_directory(
            "test", data_dir=str(ds_dir),
        )
        assert result == str(ds_dir)

    def test_nonexistent_data_dir_falls_back(self, tmp_path):
        from src.dataset_resolver import resolve_dataset_directory
        # Should not raise — falls back to other resolution methods
        # (will eventually raise FileNotFoundError if nothing works)
        result = resolve_dataset_directory(
            "nonexistent_dataset_xyz",
            data_dir=str(tmp_path / "does_not_exist"),
        )
        assert isinstance(result, str) and result

    def test_required_files_missing_raises(self, tmp_path):
        from src.dataset_resolver import resolve_dataset_directory
        ds_dir = tmp_path / "incomplete"
        ds_dir.mkdir()
        (ds_dir / "data.csv").write_text("a,b\n1,2\n")

        with pytest.raises(FileNotFoundError, match="Missing"):
            resolve_dataset_directory(
                "test",
                data_dir=str(ds_dir),
                required_files=["data.csv", "missing.csv"],
            )

    def test_required_files_found(self, tmp_path):
        from src.dataset_resolver import resolve_dataset_directory
        ds_dir = tmp_path / "complete"
        ds_dir.mkdir()
        (ds_dir / "orders.csv").write_text("a,b\n1,2\n")
        (ds_dir / "customers.csv").write_text("c,d\n3,4\n")

        result = resolve_dataset_directory(
            "test",
            data_dir=str(ds_dir),
            required_files=["orders.csv", "customers.csv"],
        )
        assert result == str(ds_dir)


class TestGetDataDir:
    """Tests for get_dataset with data_dir parameter."""

    def test_get_dataset_with_data_dir(self, tmp_path):
        from src.datasets import get_dataset
        import pandas as pd

        # Create adapter with explicit data_dir
        adapter = get_dataset("olist", data_dir=str(tmp_path))
        assert adapter.data_dir == str(tmp_path)
        assert adapter._resolved_data_dir == str(tmp_path)

    def test_get_dataset_without_data_dir(self):
        from src.datasets import get_dataset
        adapter = get_dataset("olist")
        assert adapter.dataset_name == "olist"

    def test_data_dir_setter(self, tmp_path):
        from src.datasets import get_dataset
        adapter = get_dataset("olist")
        adapter.data_dir = str(tmp_path)
        assert adapter.data_dir == str(tmp_path)
