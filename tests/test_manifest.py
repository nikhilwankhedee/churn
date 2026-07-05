"""
Tests for the manifest-driven dataset architecture.

Covers:
- GenericDatasetAdapter (loading, preprocessing, schema standardization)
- Manifest loading and validation
- Unified registry (built-in + manifest-driven datasets)
- End-to-end onboarding flow
- Multi-file merge logic
- Plugin hooks
"""
import os
import sys
import yaml
import pandas as pd
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV for testing."""
    df = pd.DataFrame({
        "cust_id": range(100),
        "purchase_date": pd.date_range("2023-01-01", periods=100, freq="D"),
        "amount": [10.5, 20.0, 15.3, 30.0, 5.0] * 20,
        "product_name": ["A", "B", "C", "A", "B"] * 20,
        "rating": [4, 5, 3, 4, 5] * 20,
    })
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sample_manifest(tmp_path, sample_csv):
    """Create a sample manifest YAML."""
    manifest = {
        "adapter": {"type": "generic", "plugin": None},
        "root_directory": str(tmp_path),
        "dataset": {
            "name": "test_dataset",
            "ecosystem_type": "transactional_marketplace",
        },
        "churn": {
            "strategy": "inactivity",
            "uses_native_churn_label": False,
            "prediction_window_days": 90,
        },
        "features": {
            "available_groups": ["purchase", "monetary", "inactivity"],
        },
        "schema": {
            "version": 2,
            "column_mapping": {
                "cust_id": "customer_id",
                "purchase_date": "event_time",
                "amount": "transaction_value",
            },
            "synthetic_columns": {},
        },
        "files": {
            "required": {"data": "test_data.csv"},
            "optional": {},
        },
        "preprocessing": {
            "timestamp_columns": ["purchase_date"],
        },
    }
    path = tmp_path / "manifest.yaml"
    with open(path, "w") as f:
        yaml.dump(manifest, f)
    return str(path)


@pytest.fixture
def multi_file_dataset(tmp_path):
    """Create a multi-file dataset for merge testing."""
    # Orders file
    orders = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "cust_id": ["A", "B", "A", "C", "B"],
        "order_date": pd.date_range("2023-01-01", periods=5),
    })
    orders.to_csv(tmp_path / "orders.csv", index=False)

    # Customers file
    customers = pd.DataFrame({
        "cust_id": ["A", "B", "C"],
        "name": ["Alice", "Bob", "Charlie"],
        "city": ["NYC", "LA", "SF"],
    })
    customers.to_csv(tmp_path / "customers.csv", index=False)

    # Payments file
    payments = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "amount": [100.0, 200.0, 50.0, 300.0, 150.0],
        "method": ["credit", "debit", "credit", "cash", "credit"],
    })
    payments.to_csv(tmp_path / "payments.csv", index=False)

    return tmp_path


@pytest.fixture
def multi_file_manifest(tmp_path, multi_file_dataset):
    """Create a manifest for multi-file dataset."""
    manifest = {
        "adapter": {"type": "generic", "plugin": None},
        "root_directory": str(multi_file_dataset),
        "dataset": {
            "name": "multi_file_test",
            "ecosystem_type": "transactional_marketplace",
        },
        "churn": {
            "strategy": "inactivity",
            "uses_native_churn_label": False,
            "prediction_window_days": 180,
        },
        "features": {
            "available_groups": ["purchase", "monetary", "payment"],
        },
        "schema": {
            "version": 2,
            "column_mapping": {
                "cust_id": "customer_id",
                "order_date": "event_time",
                "amount": "transaction_value",
            },
            "synthetic_columns": {},
        },
        "files": {
            "required": {
                "orders": "orders.csv",
                "customers": "customers.csv",
                "payments": "payments.csv",
            },
            "optional": {},
        },
        "preprocessing": {
            "timestamp_columns": ["order_date"],
        },
    }
    path = tmp_path / "manifest.yaml"
    with open(path, "w") as f:
        yaml.dump(manifest, f)
    return str(path)


@pytest.fixture
def native_churn_manifest(tmp_path):
    """Create a manifest with native churn label."""
    df = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4", "C5"],
        "monthly_charges": [29.99, 89.99, 49.99, 19.99, 99.99],
        "Churn": ["No", "Yes", "No", "Yes", "No"],
    })
    df.to_csv(tmp_path / "churn_data.csv", index=False)

    manifest = {
        "adapter": {"type": "generic", "plugin": None},
        "root_directory": str(tmp_path),
        "dataset": {
            "name": "native_churn_test",
            "ecosystem_type": "subscription",
        },
        "churn": {
            "strategy": "subscription",
            "uses_native_churn_label": True,
            "prediction_window_days": None,
        },
        "features": {
            "available_groups": ["purchase", "monetary"],
        },
        "schema": {
            "version": 2,
            "column_mapping": {
                "customer_id": "customer_id",
                "monthly_charges": "transaction_value",
            },
            "synthetic_columns": {
                "event_time": "2023-01-01",
                "event_type": "subscription",
            },
        },
        "files": {
            "required": {"data": "churn_data.csv"},
            "optional": {},
        },
        "preprocessing": {},
    }
    path = tmp_path / "manifest.yaml"
    with open(path, "w") as f:
        yaml.dump(manifest, f)
    return str(path)


# ── GenericDatasetAdapter Tests ────────────────────────────────────

class TestGenericAdapter:
    """Tests for GenericDatasetAdapter."""

    def test_init_from_manifest_path(self, sample_manifest):
        from src.datasets.generic import GenericDatasetAdapter
        adapter = GenericDatasetAdapter(manifest_path=sample_manifest)
        assert adapter.dataset_name == "test_dataset"
        assert adapter.ecosystem_type == "transactional_marketplace"

    def test_init_from_manifest_dict(self, sample_manifest):
        from src.datasets.generic import GenericDatasetAdapter
        with open(sample_manifest) as f:
            manifest = yaml.safe_load(f)
        adapter = GenericDatasetAdapter(manifest_dict=manifest)
        assert adapter.dataset_name == "test_dataset"

    def test_init_requires_manifest(self):
        from src.datasets.generic import GenericDatasetAdapter
        with pytest.raises(ValueError, match="requires either"):
            GenericDatasetAdapter()

    def test_churn_window(self, sample_manifest):
        from src.datasets.generic import GenericDatasetAdapter
        adapter = GenericDatasetAdapter(manifest_path=sample_manifest)
        assert adapter.churn_window_days == 90

    def test_native_churn_label(self, native_churn_manifest):
        from src.datasets.generic import GenericDatasetAdapter
        adapter = GenericDatasetAdapter(manifest_path=native_churn_manifest)
        assert adapter.uses_native_churn_label is True
        assert adapter.churn_window_days is None

    def test_feature_groups(self, sample_manifest):
        from src.datasets.generic import GenericDatasetAdapter
        adapter = GenericDatasetAdapter(manifest_path=sample_manifest)
        assert "purchase" in adapter.available_feature_groups
        assert "monetary" in adapter.available_feature_groups

    def test_required_files(self, sample_manifest):
        from src.datasets.generic import GenericDatasetAdapter
        adapter = GenericDatasetAdapter(manifest_path=sample_manifest)
        assert "test_data.csv" in adapter.required_files

    def test_metadata(self, sample_manifest):
        from src.datasets.generic import GenericDatasetAdapter
        adapter = GenericDatasetAdapter(manifest_path=sample_manifest)
        meta = adapter.metadata
        assert meta["dataset_name"] == "test_dataset"
        assert meta["ecosystem_type"] == "transactional_marketplace"

    def test_load_raw_data(self, sample_manifest):
        from src.datasets.generic import GenericDatasetAdapter
        adapter = GenericDatasetAdapter(manifest_path=sample_manifest)
        df = adapter.load_raw_data()
        assert len(df) == 100
        assert "cust_id" in df.columns

    def test_load_raw_data_sets_data_dir(self, sample_manifest):
        from src.datasets.generic import GenericDatasetAdapter
        adapter = GenericDatasetAdapter(manifest_path=sample_manifest)
        # Data dir comes from manifest root_directory
        df = adapter.load_raw_data()
        assert df is not None
        assert len(df) == 100

    def test_preprocess_applies_timestamp_parsing(self, sample_manifest):
        from src.datasets.generic import GenericDatasetAdapter
        adapter = GenericDatasetAdapter(manifest_path=sample_manifest)
        df = adapter.load_raw_data()
        df = adapter.preprocess(df)
        assert pd.api.types.is_datetime64_any_dtype(df["purchase_date"])

    def test_standardize_schema(self, sample_manifest):
        from src.datasets.generic import GenericDatasetAdapter
        adapter = GenericDatasetAdapter(manifest_path=sample_manifest)
        df = adapter.load_raw_data()
        df = adapter.preprocess(df)
        df = adapter.standardize_schema(df)
        assert "customer_id" in df.columns
        assert "event_time" in df.columns
        assert "transaction_value" in df.columns

    def test_full_pipeline_single_file(self, sample_manifest):
        """End-to-end: load -> preprocess -> standardize for single-file dataset."""
        from src.datasets.generic import GenericDatasetAdapter
        adapter = GenericDatasetAdapter(manifest_path=sample_manifest)
        df = adapter.load_raw_data()
        df = adapter.preprocess(df)
        df = adapter.standardize_schema(df)
        assert "customer_id" in df.columns
        assert "event_time" in df.columns
        assert "transaction_value" in df.columns
        assert len(df) == 100

    def test_native_churn_labels(self, native_churn_manifest):
        from src.datasets.generic import GenericDatasetAdapter
        adapter = GenericDatasetAdapter(manifest_path=native_churn_manifest)
        df = adapter.load_raw_data()
        df = adapter.preprocess(df)
        df = adapter.standardize_schema(df)
        cutoff = pd.Timestamp("2023-06-01")
        labels = adapter.get_native_churn_labels(df, cutoff)
        assert "customer_id" in labels.columns
        assert "churn" in labels.columns
        assert labels["churn"].isin([0, 1]).all()


# ── Multi-file Merge Tests ────────────────────────────────────────

class TestMultiFileMerge:
    """Tests for multi-file dataset merging."""

    def test_load_multi_file(self, multi_file_manifest):
        from src.datasets.generic import GenericDatasetAdapter
        adapter = GenericDatasetAdapter(manifest_path=multi_file_manifest)
        df = adapter.load_raw_data()
        assert len(df) == 5
        # Should have columns from all three files after merge
        assert "cust_id" in df.columns or "customer_id" in df.columns

    def test_merge_joins_on_common_key(self, multi_file_manifest):
        from src.datasets.generic import GenericDatasetAdapter
        adapter = GenericDatasetAdapter(manifest_path=multi_file_manifest)
        df = adapter.load_raw_data()
        # After merge, should have payment info
        assert "amount" in df.columns or "transaction_value" in df.columns

    def test_full_pipeline_multi_file(self, multi_file_manifest):
        """End-to-end for multi-file dataset."""
        from src.datasets.generic import GenericDatasetAdapter
        adapter = GenericDatasetAdapter(manifest_path=multi_file_manifest)
        df = adapter.load_raw_data()
        df = adapter.preprocess(df)
        df = adapter.standardize_schema(df)
        assert "customer_id" in df.columns
        assert "event_time" in df.columns
        assert len(df) == 5


# ── Manifest Loading Tests ─────────────────────────────────────────

class TestManifestLoading:
    """Tests for manifest loading functions."""

    def test_load_manifest_by_name(self):
        from src.datasets.generic import load_manifest
        manifest = load_manifest("olist")
        assert manifest["dataset"]["name"] == "olist"
        assert manifest["adapter"]["type"] == "olist"

    def test_load_manifest_missing_raises(self):
        from src.datasets.generic import load_manifest
        with pytest.raises(FileNotFoundError):
            load_manifest("nonexistent_dataset_xyz")

    def test_load_manifest_telco(self):
        from src.datasets.generic import load_manifest
        manifest = load_manifest("telco")
        assert manifest["churn"]["uses_native_churn_label"] is True
        assert manifest["adapter"]["type"] == "telco"

    def test_get_dataset_from_manifest(self):
        from src.datasets.generic import get_dataset_from_manifest
        adapter = get_dataset_from_manifest("olist")
        assert adapter.dataset_name == "olist"
        assert hasattr(adapter, "load_raw_data")


# ── Unified Registry Tests ─────────────────────────────────────────

class TestUnifiedRegistry:
    """Tests for the unified dataset registry."""

    def test_list_datasets_includes_builtin(self):
        from src.datasets import list_datasets
        datasets = list_datasets()
        assert "olist" in datasets
        assert "telco" in datasets
        assert "retailrocket" in datasets

    def test_list_datasets_includes_manifest(self):
        from src.datasets import list_datasets
        datasets = list_datasets()
        # All datasets with manifests should appear
        assert "olist" in datasets
        assert "rees46" in datasets

    def test_get_dataset_builtin(self):
        from src.datasets import get_dataset
        adapter = get_dataset("olist")
        assert adapter.dataset_name == "olist"
        assert hasattr(adapter, "load_raw_data")

    def test_get_dataset_with_data_dir(self):
        from src.datasets import get_dataset
        adapter = get_dataset("olist", data_dir="/tmp/test")
        assert adapter.dataset_name == "olist"
        assert adapter.data_dir == "/tmp/test"

    def test_get_dataset_unknown_raises(self):
        from src.datasets import list_datasets
        from src.datasets import get_dataset
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset("completely_nonexistent_dataset_xyz")

    def test_get_ecosystem_type_builtin(self):
        from src.datasets import get_ecosystem_type
        assert get_ecosystem_type("olist") == "transactional_marketplace"
        assert get_ecosystem_type("telco") == "subscription"

    def test_get_ecosystem_type_manifest(self):
        from src.datasets import get_ecosystem_type
        # Should read from manifest
        eco = get_ecosystem_type("olist")
        assert eco == "transactional_marketplace"


# ── Dataset Resolver Manifest Tests ────────────────────────────────

class TestResolverManifest:
    """Tests for resolver reading manifest root_directory."""

    def test_resolver_checks_manifest(self):
        from src.dataset_resolver import _get_manifest_root_directory
        # olist has root_directory: null, so should return None
        result = _get_manifest_root_directory("olist")
        # It returns None because root_directory is null in the manifest
        assert result is None


# ── No Hardcoded Paths Tests ──────────────────────────────────────

class TestNoHardcodedPaths:
    """Verify no hardcoded filesystem paths in the new architecture."""

    def test_generic_adapter_no_hardcoded_paths(self):
        """GenericDatasetAdapter should not reference /kaggle/input etc."""
        import inspect
        from src.datasets.generic import GenericDatasetAdapter
        source = inspect.getsource(GenericDatasetAdapter)
        assert "/kaggle/" not in source
        assert "DEFAULT_PATH" not in source

    def test_dataset_init_no_hardcoded_paths(self):
        """Dataset registry __init__ should not have hardcoded paths."""
        import inspect
        import src.datasets
        source = inspect.getsource(src.datasets)
        assert "/kaggle/input" not in source

    def test_resolver_manifest_function_no_hardcoded_paths(self):
        """Resolver manifest function should not have hardcoded paths."""
        import inspect
        from src.dataset_resolver import _get_manifest_root_directory
        source = inspect.getsource(_get_manifest_root_directory)
        assert "/kaggle/" not in source


# ── Edge Case Tests ────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_empty_manifest_raises(self, tmp_path):
        from src.datasets.generic import GenericDatasetAdapter
        manifest = {"adapter": {"type": "generic"}, "dataset": {"name": "empty"}}
        adapter = GenericDatasetAdapter(manifest_dict=manifest)
        adapter._resolved_data_dir = str(tmp_path)
        with pytest.raises(ValueError, match="no files defined"):
            adapter.load_raw_data()

    def test_missing_required_file_raises(self, tmp_path):
        from src.datasets.generic import GenericDatasetAdapter
        manifest = {
            "adapter": {"type": "generic"},
            "dataset": {"name": "missing"},
            "files": {"required": {"data": "nonexistent.csv"}, "optional": {}},
        }
        adapter = GenericDatasetAdapter(manifest_dict=manifest)
        adapter._resolved_data_dir = str(tmp_path)
        with pytest.raises(FileNotFoundError):
            adapter.load_raw_data()

    def test_missing_optional_file_skipped(self, tmp_path):
        """Missing optional files should be skipped, not raise."""
        from src.datasets.generic import GenericDatasetAdapter
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_csv(tmp_path / "main.csv", index=False)

        manifest = {
            "adapter": {"type": "generic"},
            "dataset": {"name": "partial"},
            "files": {
                "required": {"data": "main.csv"},
                "optional": {"extra": "missing_optional.csv"},
            },
            "schema": {"column_mapping": {}, "synthetic_columns": {}},
            "preprocessing": {},
        }
        adapter = GenericDatasetAdapter(manifest_dict=manifest)
        adapter._resolved_data_dir = str(tmp_path)
        result = adapter.load_raw_data()
        assert len(result) == 2

    def test_schema_version_2(self, sample_manifest):
        """Manifest should have schema version 2."""
        from src.datasets.generic import load_manifest
        manifest = load_manifest("olist")
        assert manifest.get("schema", {}).get("version") == 2

    def test_adapter_type_in_manifest(self, sample_manifest):
        """Manifest should have adapter.type field."""
        from src.datasets.generic import load_manifest
        manifest = load_manifest("olist")
        assert "adapter" in manifest
        assert "type" in manifest["adapter"]


# ── Manifest Validator Tests ────────────────────────────────────────

class TestManifestValidator:
    """Tests for the manifest validation utility."""

    def test_valid_manifest_passes(self, sample_manifest):
        """A well-formed manifest should pass validation."""
        from src.datasets.manifest_validator import validate_manifest_path
        result = validate_manifest_path(sample_manifest)
        assert result.valid
        assert result.error_count == 0

    def test_missing_name_fails(self, tmp_path):
        """Manifest without dataset.name should fail."""
        from src.datasets.manifest_validator import validate_manifest_path
        manifest = tmp_path / "bad.yaml"
        manifest.write_text(yaml.dump({
            "adapter": {"type": "generic"},
            "files": {"required": {"data": "test.csv"}},
            "schema": {"version": 2},
        }))
        result = validate_manifest_path(str(manifest))
        assert not result.valid
        assert any("name" in e.path or "dataset" in e.path for e in result.errors)

    def test_empty_file_fails(self, tmp_path):
        """Empty YAML file should fail."""
        from src.datasets.manifest_validator import validate_manifest_path
        manifest = tmp_path / "empty.yaml"
        manifest.write_text("")
        result = validate_manifest_path(str(manifest))
        assert not result.valid
        assert any("empty" in e.message.lower() for e in result.errors)

    def test_invalid_yaml_fails(self, tmp_path):
        """Malformed YAML should fail."""
        from src.datasets.manifest_validator import validate_manifest_path
        manifest = tmp_path / "bad.yaml"
        manifest.write_text("{{invalid yaml: [")
        result = validate_manifest_path(str(manifest))
        assert not result.valid

    def test_missing_file_fails(self):
        """Non-existent file should fail."""
        from src.datasets.manifest_validator import validate_manifest_path
        result = validate_manifest_path("/nonexistent/path.yaml")
        assert not result.valid

    def test_unknown_adapter_type_warns(self, tmp_path):
        """Unknown adapter.type should warn, not error."""
        from src.datasets.manifest_validator import validate_manifest_path
        manifest = tmp_path / "warn.yaml"
        manifest.write_text(yaml.dump({
            "dataset": {"name": "test"},
            "adapter": {"type": "unknown_adapter"},
            "files": {"required": {"data": "test.csv"}},
            "schema": {"version": 2, "columns": {}},
        }))
        result = validate_manifest_path(str(manifest))
        assert result.valid
        assert result.warning_count > 0
        assert any("adapter.type" in w.path for w in result.warnings)

    def test_missing_recommended_columns_warns(self, tmp_path):
        """Missing recommended schema columns should warn."""
        from src.datasets.manifest_validator import validate_manifest_path
        manifest = tmp_path / "warn2.yaml"
        manifest.write_text(yaml.dump({
            "dataset": {"name": "test"},
            "files": {"required": {"data": "test.csv"}},
            "schema": {"version": 2, "columns": {}},
        }))
        result = validate_manifest_path(str(manifest))
        assert result.valid
        assert any("customer_id" in w.message for w in result.warnings)

    def test_built_in_manifests_validate(self):
        """All built-in dataset manifests should validate."""
        from src.datasets.manifest_validator import validate_manifest_path
        from src.config import get_configs_dir
        configs_dir = get_configs_dir() / "datasets"
        for yaml_file in configs_dir.glob("*.yaml"):
            result = validate_manifest_path(str(yaml_file))
            assert result.valid, f"{yaml_file.name} failed: {result.report()}"
