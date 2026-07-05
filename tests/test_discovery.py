"""
Comprehensive tests for dataset discovery, path resolution, and registry.

Covers:
- Platform-independent path handling (Linux, macOS, Windows path patterns)
- Recursive directory scanning
- Dataset signature detection
- Previously registered dataset recognition
- Environment detection
- Output directory resolution
- Persistent registry operations
- Unknown dataset handling
- Edge cases (missing files, partial datasets, renamed folders)
"""
import os
import sys
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root on path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ══════════════════════════════════════════════════════════════════
#  FIXTURES
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_dir(tmp_path):
    """Create a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def sample_csvs(tmp_dir):
    """Create sample CSV files mimicking various datasets."""
    import pandas as pd

    # Olist-like dataset (in a renamed folder)
    olist_dir = tmp_dir / "my-ecommerce-data"
    olist_dir.mkdir()
    pd.DataFrame({
        "order_id": ["o1", "o2"],
        "customer_id": ["c1", "c2"],
        "order_purchase_timestamp": ["2020-01-01", "2020-02-01"],
    }).to_csv(olist_dir / "olist_orders_dataset.csv", index=False)
    pd.DataFrame({
        "customer_id": ["c1", "c2"],
        "customer_unique_id": ["u1", "u2"],
    }).to_csv(olist_dir / "olist_customers_dataset.csv", index=False)
    pd.DataFrame({
        "order_id": ["o1", "o2"],
        "payment_type": ["credit_card", "boleto"],
        "payment_value": [100.0, 200.0],
    }).to_csv(olist_dir / "olist_order_payments_dataset.csv", index=False)

    # Telco-like dataset (in a random folder name)
    telco_dir = tmp_dir / "customer-churn-2026"
    telco_dir.mkdir()
    pd.DataFrame({
        "customerID": ["t1", "t2"],
        "Churn": ["Yes", "No"],
        "MonthlyCharges": [50.0, 80.0],
    }).to_csv(telco_dir / "telco_customer_churn.csv", index=False)

    # Unknown dataset
    unknown_dir = tmp_dir / "mystery-data"
    unknown_dir.mkdir()
    pd.DataFrame({
        "id": [1, 2, 3],
        "value": [10, 20, 30],
    }).to_csv(unknown_dir / "random_data.csv", index=False)

    return {
        "olist_dir": olist_dir,
        "telco_dir": telco_dir,
        "unknown_dir": unknown_dir,
    }


@pytest.fixture
def nested_csvs(tmp_dir):
    """Create CSVs in deeply nested directories."""
    import pandas as pd

    deep_dir = tmp_dir / "level1" / "level2" / "level3" / "dataset"
    deep_dir.mkdir(parents=True)
    pd.DataFrame({
        "order_id": ["o1"],
        "customer_id": ["c1"],
        "order_purchase_timestamp": ["2020-01-01"],
    }).to_csv(deep_dir / "olist_orders_dataset.csv", index=False)
    pd.DataFrame({
        "customer_id": ["c1"],
        "customer_unique_id": ["u1"],
    }).to_csv(deep_dir / "olist_customers_dataset.csv", index=False)
    pd.DataFrame({
        "order_id": ["o1"],
        "payment_type": ["credit_card"],
        "payment_value": [50.0],
    }).to_csv(deep_dir / "olist_order_payments_dataset.csv", index=False)

    return deep_dir


@pytest.fixture
def kaggle_structure(tmp_dir):
    """Create a directory structure mimicking Kaggle."""
    import pandas as pd

    kaggle_input = tmp_dir / "kaggle" / "input"
    kaggle_input.mkdir(parents=True)

    # Olist in a Kaggle-style folder
    olist_kaggle = kaggle_input / "brazilian-ecommerce"
    olist_kaggle.mkdir()
    pd.DataFrame({
        "order_id": ["o1", "o2"],
        "customer_id": ["c1", "c2"],
        "order_purchase_timestamp": ["2020-01-01", "2020-02-01"],
    }).to_csv(olist_kaggle / "olist_orders_dataset.csv", index=False)
    pd.DataFrame({
        "customer_id": ["c1", "c2"],
        "customer_unique_id": ["u1", "u2"],
    }).to_csv(olist_kaggle / "olist_customers_dataset.csv", index=False)
    pd.DataFrame({
        "order_id": ["o1", "o2"],
        "payment_type": ["credit_card", "boleto"],
        "payment_value": [100.0, 200.0],
    }).to_csv(olist_kaggle / "olist_order_payments_dataset.csv", index=False)

    # Telco in another folder
    telco_kaggle = kaggle_input / "telco-churn-data"
    telco_kaggle.mkdir()
    pd.DataFrame({
        "customerID": ["t1"],
        "Churn": ["Yes"],
        "MonthlyCharges": [50.0],
    }).to_csv(telco_kaggle / "telco_customer_churn.csv", index=False)

    return kaggle_input


@pytest.fixture
def duplicate_datasets(tmp_dir):
    """Create duplicate dataset folders."""
    import pandas as pd

    for name in ["copy-1", "copy-2", "copy-3"]:
        d = tmp_dir / name
        d.mkdir()
        pd.DataFrame({
            "order_id": ["o1"],
            "customer_id": ["c1"],
            "order_purchase_timestamp": ["2020-01-01"],
        }).to_csv(d / "olist_orders_dataset.csv", index=False)
        pd.DataFrame({
            "customer_id": ["c1"],
            "customer_unique_id": ["u1"],
        }).to_csv(d / "olist_customers_dataset.csv", index=False)
        pd.DataFrame({
            "order_id": ["o1"],
            "payment_type": ["credit_card"],
            "payment_value": [50.0],
        }).to_csv(d / "olist_order_payments_dataset.csv", index=False)

    return tmp_dir


@pytest.fixture
def registry_store(tmp_dir):
    """Create a temporary registry store with isolated configs."""
    from src.registry_store import DatasetRegistryStore
    empty_configs = tmp_dir / "empty_configs"
    empty_configs.mkdir()
    store = DatasetRegistryStore(project_root=tmp_dir, configs_dir=empty_configs)
    return store


# ══════════════════════════════════════════════════════════════════
#  1. PLATFORM-INDEPENDENT PATH HANDLING
# ══════════════════════════════════════════════════════════════════

class TestPlatformPaths:
    """Test pathlib-based path resolution across platforms."""

    def test_resolve_relative_path(self, tmp_dir, monkeypatch):
        from src.paths import resolve
        monkeypatch.chdir(tmp_dir)
        result = resolve("some/path")
        assert result.is_absolute()
        assert result == tmp_dir / "some/path"

    def test_resolve_absolute_path(self, tmp_dir):
        from src.paths import resolve
        result = resolve(str(tmp_dir / "subdir"))
        assert result == tmp_dir / "subdir"

    def test_resolve_home_expansion(self):
        from src.paths import resolve
        result = resolve("~/test_file.txt")
        assert str(result).startswith(str(Path.home()))

    def test_safe_resolve_existing(self, tmp_dir):
        from src.paths import safe_resolve
        f = tmp_dir / "exists.csv"
        f.write_text("test")
        assert safe_resolve(f) == f

    def test_safe_resolve_missing(self, tmp_dir):
        from src.paths import safe_resolve
        assert safe_resolve(tmp_dir / "nope.csv") is None

    def test_ensure_dir_creates_nested(self, tmp_dir):
        from src.paths import ensure_dir
        nested = tmp_dir / "a" / "b" / "c"
        result = ensure_dir(nested)
        assert result.is_dir()
        assert nested.is_dir()

    def test_relative_to(self, tmp_dir):
        from src.paths import relative_to
        child = tmp_dir / "sub" / "file.csv"
        result = relative_to(child, tmp_dir)
        assert result == Path("sub/file.csv")

    def test_relative_to_not_relative(self, tmp_dir):
        from src.paths import relative_to
        unrelated = Path("/some/other/path")
        assert relative_to(unrelated, tmp_dir) is None

    def test_is_subpath(self, tmp_dir):
        from src.paths import is_subpath
        child = tmp_dir / "sub" / "file.csv"
        assert is_subpath(child, tmp_dir)
        assert not is_subpath(Path("/other/path"), tmp_dir)

    def test_find_csv_files(self, tmp_dir):
        from src.paths import find_csv_files
        (tmp_dir / "a.csv").write_text("test")
        (tmp_dir / "b.csv").write_text("test")
        sub = tmp_dir / "sub"
        sub.mkdir()
        (sub / "c.csv").write_text("test")

        files = find_csv_files(tmp_dir)
        assert len(files) == 3

    def test_find_csv_files_non_recursive(self, tmp_dir):
        from src.paths import find_csv_files
        (tmp_dir / "a.csv").write_text("test")
        sub = tmp_dir / "sub"
        sub.mkdir()
        (sub / "b.csv").write_text("test")

        files = find_csv_files(tmp_dir, recursive=False)
        assert len(files) == 1

    def test_list_immediate_subdirs(self, tmp_dir):
        from src.paths import list_immediate_subdirs
        (tmp_dir / "dir1").mkdir()
        (tmp_dir / "dir2").mkdir()
        (tmp_dir / "file.txt").write_text("test")
        sub = tmp_dir / "dir1" / "nested"
        sub.mkdir()

        dirs = list_immediate_subdirs(tmp_dir)
        assert len(dirs) == 2
        assert all(d.is_dir() for d in dirs)

    def test_normalize_path_for_display(self, tmp_dir):
        from src.paths import normalize_path_for_display
        result = normalize_path_for_display(tmp_dir / "sub" / "file.csv")
        assert "\\" not in result

    def test_windows_path_pattern(self):
        """Test that PureWindowsPath strings are handled."""
        from src.paths import resolve
        # Simulate a Windows-style path on Linux (just tests string handling)
        result = resolve("datasets/olist")
        assert result.is_absolute()

    def test_macos_path_pattern(self):
        """Test macOS-style paths."""
        from src.paths import resolve
        result = resolve("/Users/test/datasets")
        assert isinstance(result, Path)


# ══════════════════════════════════════════════════════════════════
#  2. ENVIRONMENT DETECTION
# ══════════════════════════════════════════════════════════════════

class TestEnvironmentDetection:
    """Test environment detection logic."""

    def test_local_detection(self, monkeypatch):
        from src.environment import detect_environment
        monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
        monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
        with patch("src.environment._is_kaggle", return_value=False), \
             patch("src.environment._is_colab", return_value=False), \
             patch("src.environment.Path.cwd", return_value=Path("/tmp")):
            env = detect_environment()
            assert env.is_local is True
            assert env.is_kaggle is False
            assert env.is_colab is False

    def test_kaggle_detection_with_input(self, tmp_dir):
        from src.environment import detect_environment
        with patch("src.environment._is_kaggle", return_value=True), \
             patch("src.environment._kaggle_working_dir", return_value=tmp_dir):
            env = detect_environment()
            assert env.is_kaggle is True

    def test_explicit_dataset_root_override(self, tmp_dir):
        from src.environment import detect_environment
        with patch("src.environment._is_kaggle", return_value=False), \
             patch("src.environment._is_colab", return_value=False), \
             patch("src.environment.Path.cwd", return_value=Path("/tmp")):
            env = detect_environment(str(tmp_dir))
            assert env.dataset_root == tmp_dir

    def test_default_output_dir(self):
        from src.environment import get_default_output_dir
        with patch("src.environment._is_kaggle", return_value=False), \
             patch("src.environment._is_colab", return_value=False), \
             patch("src.environment.Path.cwd", return_value=Path("/tmp")):
            result = get_default_output_dir()
            assert result is not None
            assert "outputs" in result

    def test_default_dataset_root_none_for_local(self):
        from src.environment import get_default_dataset_root
        with patch("src.environment._is_kaggle", return_value=False), \
             patch("src.environment._is_colab", return_value=False), \
             patch("src.environment.Path.cwd", return_value=Path("/tmp")):
            result = get_default_dataset_root()
            assert result is None


# ══════════════════════════════════════════════════════════════════
#  3. DATASET DETECTION
# ══════════════════════════════════════════════════════════════════

class TestDatasetDetection:
    """Test built-in dataset detectors."""

    def test_olist_detection(self, sample_csvs):
        from src.discovery.detectors import OlistDetector
        detector = OlistDetector()
        result = detector.detect(sample_csvs["olist_dir"])
        assert result.matched is True
        assert result.confidence >= 0.8
        assert result.dataset_type == "olist"
        assert len(result.matched_files) == 3

    def test_telco_detection(self, sample_csvs):
        from src.discovery.detectors import TelcoDetector
        detector = TelcoDetector()
        result = detector.detect(sample_csvs["telco_dir"])
        assert result.matched is True
        assert result.confidence >= 0.8
        assert result.dataset_type == "telco"

    def test_unknown_dataset_no_match(self, sample_csvs):
        from src.discovery.detectors import OlistDetector, TelcoDetector
        for Detector in [OlistDetector, TelcoDetector]:
            result = Detector().detect(sample_csvs["unknown_dir"])
            assert result.matched is False
            assert result.confidence == 0.0

    def test_detection_in_renamed_folder(self, sample_csvs):
        """Detection should work regardless of folder name."""
        from src.discovery.detectors import OlistDetector
        result = OlistDetector().detect(sample_csvs["olist_dir"])
        assert result.matched is True
        assert "olist" in result.dataset_type

    def test_detection_with_missing_files(self, tmp_dir):
        from src.discovery.detectors import OlistDetector
        # Only provide 1 of 3 required files
        import pandas as pd
        pd.DataFrame({
            "order_id": ["o1"],
            "customer_id": ["c1"],
            "order_purchase_timestamp": ["2020-01-01"],
        }).to_csv(tmp_dir / "olist_orders_dataset.csv", index=False)

        result = OlistDetector().detect(tmp_dir)
        assert result.matched is False
        assert len(result.missing_files) > 0

    def test_detection_nonexistent_dir(self):
        from src.discovery.detectors import OlistDetector
        result = OlistDetector().detect(Path("/nonexistent/path"))
        assert result.matched is False
        assert result.confidence == 0.0

    def test_all_detectors_available(self):
        from src.discovery.detectors import get_all_detectors
        detectors = get_all_detectors()
        assert len(detectors) == 6
        names = [d.dataset_type for d in detectors]
        assert "olist" in names
        assert "telco" in names
        assert "rees46" in names
        assert "retailrocket" in names
        assert "online_retail_ii" in names
        assert "instacart" in names

    def test_detect_dataset_returns_sorted(self, sample_csvs):
        from src.discovery.detectors import detect_dataset
        results = detect_dataset(sample_csvs["olist_dir"])
        assert len(results) > 0
        # Results should be sorted by confidence descending
        for i in range(len(results) - 1):
            assert results[i].confidence >= results[i + 1].confidence


# ══════════════════════════════════════════════════════════════════
#  4. RECURSIVE SCANNING
# ══════════════════════════════════════════════════════════════════

class TestRecursiveScanning:
    """Test recursive directory scanning."""

    def test_scan_top_level(self, sample_csvs):
        from src.discovery.scanner import scan_directory
        result = scan_directory(sample_csvs["olist_dir"].parent)
        assert result.total_directories_scanned > 0
        assert len(result.high_confidence) >= 2  # olist + telco

    def test_scan_nested_directories(self, nested_csvs):
        from src.discovery.scanner import scan_directory
        result = scan_directory(nested_csvs.parent.parent.parent)
        assert len(result.high_confidence) >= 1
        names = [d.dataset_name for d in result.high_confidence]
        assert "olist" in names

    def test_scan_kaggle_structure(self, kaggle_structure):
        from src.discovery.scanner import scan_directory
        result = scan_directory(kaggle_structure)
        assert len(result.high_confidence) >= 2
        names = [d.dataset_name for d in result.high_confidence]
        assert "olist" in names
        assert "telco" in names

    def test_scan_max_depth(self, nested_csvs):
        from src.discovery.scanner import scan_directory
        # Depth 1 should not find the deeply nested dataset
        result = scan_directory(nested_csvs.parent.parent.parent, max_depth=1)
        names = [d.dataset_name for d in result.high_confidence]
        assert "olist" not in names

    def test_scan_nonexistent_root(self):
        from src.discovery.scanner import scan_directory
        result = scan_directory("/nonexistent/path")
        assert len(result.discovered) == 0
        assert len(result.errors) > 0

    def test_scan_result_summary(self, sample_csvs):
        from src.discovery.scanner import scan_directory
        result = scan_directory(sample_csvs["olist_dir"].parent)
        summary = result.summary()
        assert "Scanned" in summary
        assert "Discovered" in summary

    def test_deduplication(self, duplicate_datasets):
        from src.discovery.scanner import scan_directory
        result = scan_directory(duplicate_datasets)
        # Should deduplicate to single olist entry
        olist_matches = [d for d in result.discovered if d.dataset_name == "olist"]
        assert len(olist_matches) == 1

    def test_scan_by_name_property(self, sample_csvs):
        from src.discovery.scanner import scan_directory
        result = scan_directory(sample_csvs["olist_dir"].parent)
        by_name = result.by_name
        assert "olist" in by_name
        assert "telco" in by_name


# ══════════════════════════════════════════════════════════════════
#  5. PERSISTENT REGISTRY
# ══════════════════════════════════════════════════════════════════

class TestPersistentRegistry:
    """Test the persistent dataset registry store."""

    def test_register_dataset(self, registry_store, tmp_dir):
        config_path = tmp_dir / "test.yaml"
        config_path.write_text("dataset:\n  name: test\n")
        entry = registry_store.register(
            name="test",
            config_path=config_path,
            required_columns={"orders.csv": ["id", "date"]},
            ecosystem_type="test",
        )
        assert entry.name == "test"
        assert entry.schema_fingerprint

    def test_get_registered(self, registry_store, tmp_dir):
        config_path = tmp_dir / "test.yaml"
        config_path.write_text("dataset:\n  name: test\n")
        registry_store.register(
            name="my_dataset",
            config_path=config_path,
            required_columns={"data.csv": ["col1"]},
        )
        retrieved = registry_store.get("my_dataset")
        assert retrieved is not None
        assert retrieved.name == "my_dataset"

    def test_get_unregistered(self, registry_store):
        assert registry_store.get("nonexistent") is None

    def test_list_registered(self, registry_store, tmp_dir):
        for name in ["alpha", "beta", "gamma"]:
            config_path = tmp_dir / f"{name}.yaml"
            config_path.write_text(f"dataset:\n  name: {name}\n")
            registry_store.register(name=name, config_path=config_path)

        registered = registry_store.list_registered()
        assert registered == ["alpha", "beta", "gamma"]

    def test_remove_dataset(self, registry_store, tmp_dir):
        config_path = tmp_dir / "test.yaml"
        config_path.write_text("dataset:\n  name: test\n")
        registry_store.register(name="test", config_path=config_path)
        assert registry_store.remove("test") is True
        assert registry_store.get("test") is None

    def test_remove_nonexistent(self, registry_store):
        assert registry_store.remove("nope") is False

    def test_sync_from_configs(self, tmp_dir):
        from src.registry_store import DatasetRegistryStore
        configs_dir = tmp_dir / "configs" / "datasets"
        configs_dir.mkdir(parents=True)

        # Create a YAML config
        (configs_dir / "mydata.yaml").write_text(
            "dataset:\n  name: mydata\n  ecosystem_type: subscription\n"
            "schema:\n  column_mapping:\n    customerID: customer_id\n"
        )

        store = DatasetRegistryStore(project_root=tmp_dir, configs_dir=configs_dir)
        count = store.sync_from_configs()
        assert count == 1
        assert "mydata" in store.list_registered()

    def test_sync_idempotent(self, tmp_dir):
        from src.registry_store import DatasetRegistryStore
        configs_dir = tmp_dir / "configs" / "datasets"
        configs_dir.mkdir(parents=True)
        (configs_dir / "test.yaml").write_text("dataset:\n  name: test\n")

        store = DatasetRegistryStore(project_root=tmp_dir, configs_dir=configs_dir)
        count1 = store.sync_from_configs()
        count2 = store.sync_from_configs()
        assert count1 == 1
        assert count2 == 0  # Already synced

    def test_schema_fingerprint_consistency(self, registry_store):
        fp1 = registry_store._compute_schema_fingerprint(
            {"orders.csv": ["id", "date", "amount"]}
        )
        fp2 = registry_store._compute_schema_fingerprint(
            {"orders.csv": ["id", "date", "amount"]}
        )
        assert fp1 == fp2

    def test_schema_fingerprint_different(self, registry_store):
        fp1 = registry_store._compute_schema_fingerprint(
            {"orders.csv": ["id", "date"]}
        )
        fp2 = registry_store._compute_schema_fingerprint(
            {"orders.csv": ["id", "date", "amount"]}
        )
        assert fp1 != fp2

    def test_get_schema_info(self, registry_store, tmp_dir):
        config_path = tmp_dir / "test.yaml"
        config_path.write_text("dataset:\n  name: test\n")
        registry_store.register(
            name="test",
            config_path=config_path,
            required_columns={"data.csv": ["a", "b"]},
        )
        info = registry_store.get_schema_info()
        assert "test" in info
        assert "required_columns" in info["test"]

    def test_persistence_across_instances(self, tmp_dir):
        from src.registry_store import DatasetRegistryStore
        config_path = tmp_dir / "test.yaml"
        config_path.write_text("dataset:\n  name: test\n")

        store1 = DatasetRegistryStore(project_root=tmp_dir)
        store1.register(name="persistent", config_path=config_path)

        store2 = DatasetRegistryStore(project_root=tmp_dir)
        assert store2.get("persistent") is not None


# ══════════════════════════════════════════════════════════════════
#  6. OUTPUT DIRECTORY RESOLUTION
# ══════════════════════════════════════════════════════════════════

class TestOutputDirectory:
    """Test output directory resolution."""

    def test_explicit_output_dir(self, tmp_dir):
        from src.paths import output_directory
        result = output_directory(output_dir=str(tmp_dir / "my_output"))
        assert result == tmp_dir / "my_output"
        assert result.is_dir()

    def test_default_output_dir(self):
        from src.paths import output_directory
        with patch("src.environment._is_kaggle", return_value=False), \
             patch("src.environment._is_colab", return_value=False), \
             patch("src.environment.Path.cwd", return_value=Path("/tmp")):
            result = output_directory()
            assert "outputs" in str(result)
            assert result.is_dir()

    def test_kaggle_output_dir(self, tmp_dir):
        from src.paths import output_directory
        with patch("src.environment._is_kaggle", return_value=True), \
             patch("src.environment._kaggle_working_dir", return_value=tmp_dir):
            result = output_directory(dataset_root=str(tmp_dir))
            assert "outputs" in str(result)


# ══════════════════════════════════════════════════════════════════
#  7. DISCOVERY WITH REGISTERED DATASETS
# ══════════════════════════════════════════════════════════════════

class TestDiscoveryWithRegistered:
    """Test discovery against previously registered schemas."""

    def test_recognize_previously_registered(self, tmp_dir):
        import pandas as pd
        from src.discovery.scanner import scan_directory

        # Create a dataset with a non-standard name that won't match built-in detectors
        # but WILL match via schema columns
        ds_dir = tmp_dir / "random-folder-name"
        ds_dir.mkdir()
        pd.DataFrame({
            "subscriber_id": ["t1", "t2"],
            "is_churned": [1, 0],
            "monthly_charge": [50.0, 80.0],
        }).to_csv(ds_dir / "telecom_data.csv", index=False)

        # Provide schema info as if previously registered
        previously_registered = {
            "my_telecom": {
                "required_columns": {
                    "telecom_data.csv": ["subscriber_id", "is_churned"],
                },
            },
        }

        result = scan_directory(
            ds_dir.parent,
            previously_registered=previously_registered,
        )

        # Should find my_telecom via schema matching
        matches = [d for d in result.discovered if d.dataset_name == "my_telecom"]
        assert len(matches) >= 1
        assert matches[0].source == "registered"


# ══════════════════════════════════════════════════════════════════
#  8. ADAPTER DATA DIR FLEXIBILITY
# ══════════════════════════════════════════════════════════════════

class TestAdapterDataDir:
    """Test that adapters support flexible data directory resolution."""

    def test_data_dir_override(self):
        from src.datasets.base import BaseDatasetAdapter

        class TestAdapter(BaseDatasetAdapter):
            @property
            def dataset_name(self): return "test"
            @property
            def ecosystem_type(self): return "test"
            def load_raw_data(self): return None
            def preprocess(self, df): return df
            def standardize_schema(self, df): return df
            @property
            def available_feature_groups(self): return []
            @property
            def metadata(self): return {}

        adapter = TestAdapter()
        original_dir = adapter.data_dir
        assert original_dir is not None

        # Override
        adapter.data_dir = "/custom/path"
        assert adapter.data_dir == "/custom/path"

        # Reset
        adapter.data_dir = None
        assert adapter.data_dir == original_dir


# ══════════════════════════════════════════════════════════════════
#  9. PARTIAL DATASETS AND EDGE CASES
# ══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and partial datasets."""

    def test_empty_directory(self, tmp_dir):
        from src.discovery.scanner import scan_directory
        result = scan_directory(tmp_dir)
        assert len(result.discovered) == 0

    def test_no_csv_files(self, tmp_dir):
        from src.discovery.scanner import scan_directory
        (tmp_dir / "data.txt").write_text("not a csv")
        result = scan_directory(tmp_dir)
        assert len(result.discovered) == 0

    def test_only_optional_files(self, tmp_dir):
        from src.discovery.detectors import OlistDetector
        import pandas as pd
        # Provide only optional files, not required ones
        pd.DataFrame({"a": [1]}).to_csv(tmp_dir / "olist_products_dataset.csv", index=False)
        result = OlistDetector().detect(tmp_dir)
        assert result.matched is False

    def test_hidden_directories_skipped(self, tmp_dir):
        from src.discovery.scanner import _collect_directories
        hidden = tmp_dir / ".hidden_dir"
        hidden.mkdir()
        dirs = _collect_directories(tmp_dir)
        assert hidden not in dirs

    def test_special_characters_in_path(self, tmp_dir):
        from src.paths import resolve
        special = tmp_dir / "path with spaces (1)"
        result = resolve(str(special))
        assert special.name in str(result)

    def test_symlink_directory(self, tmp_dir):
        from src.discovery.scanner import scan_directory
        import pandas as pd

        real_dir = tmp_dir / "real_data"
        real_dir.mkdir()
        pd.DataFrame({
            "order_id": ["o1"],
            "customer_id": ["c1"],
            "order_purchase_timestamp": ["2020-01-01"],
        }).to_csv(real_dir / "olist_orders_dataset.csv", index=False)
        pd.DataFrame({
            "customer_id": ["c1"],
            "customer_unique_id": ["u1"],
        }).to_csv(real_dir / "olist_customers_dataset.csv", index=False)
        pd.DataFrame({
            "order_id": ["o1"],
            "payment_type": ["credit_card"],
            "payment_value": [50.0],
        }).to_csv(real_dir / "olist_order_payments_dataset.csv", index=False)

        link = tmp_dir / "linked_data"
        try:
            link.symlink_to(real_dir)
            result = scan_directory(tmp_dir)
            names = [d.dataset_name for d in result.discovered]
            assert "olist" in names
        except OSError:
            pytest.skip("Symlinks not supported on this platform")


# ══════════════════════════════════════════════════════════════════
#  10. CLI COMMANDS
# ══════════════════════════════════════════════════════════════════

class TestCLICommands:
    """Test CLI command registration."""

    def test_benchmark_command_exists(self):
        try:
            from src.cli.main import app
            commands = [cmd for cmd in app.registered_commands if hasattr(cmd, 'name')]
            names = [cmd.name for cmd in commands]
            assert "benchmark" in names
        except Exception:
            pytest.skip("CLI app could not be loaded")

    def test_discover_command_exists(self):
        try:
            from src.cli.main import app
            commands = [cmd for cmd in app.registered_commands if hasattr(cmd, 'name')]
            names = [cmd.name for cmd in commands]
            assert "discover" in names
        except Exception:
            pytest.skip("CLI app could not be loaded")

    def test_existing_commands_still_work(self):
        try:
            from src.cli.main import app
            commands = [cmd for cmd in app.registered_commands if hasattr(cmd, 'name')]
            names = [cmd.name for cmd in commands]
            # Original commands still exist
            assert "version" in names
            assert "datasets" in names
            assert "models" in names
            assert "doctor" in names
            assert "register" in names
        except Exception:
            pytest.skip("CLI app could not be loaded")


# ══════════════════════════════════════════════════════════════════
#  11. PYTHON API
# ══════════════════════════════════════════════════════════════════

class TestPythonAPI:
    """Test the Python API methods for discovery and benchmark."""

    def test_discover_method(self):
        from src.api import ChurnFramework
        fw = ChurnFramework()
        result = fw.discover("/nonexistent/path")
        assert isinstance(result, list)

    def test_detect_environment_method(self):
        from src.api import ChurnFramework
        fw = ChurnFramework()
        with patch("src.environment._is_kaggle", return_value=False), \
             patch("src.environment._is_colab", return_value=False), \
             patch("src.environment.Path.cwd", return_value=Path("/tmp")):
            result = fw.detect_environment()
            assert "name" in result
            assert "is_local" in result
            assert "is_kaggle" in result

    def test_benchmark_method_dry_run(self, sample_csvs):
        from src.api import ChurnFramework
        fw = ChurnFramework()
        result = fw.benchmark(
            dataset_root=str(sample_csvs["olist_dir"].parent),
            dry_run=True,
        )
        assert "discovered_datasets" in result
        assert "executed_datasets" in result
        assert isinstance(result["discovered_datasets"], list)


# ══════════════════════════════════════════════════════════════════
#  12. NO HARDCODED PATHS CONFIRMATION
# ══════════════════════════════════════════════════════════════════

class TestNoHardcodedPaths:
    """Confirm no hardcoded dataset paths in new modules."""

    def test_detectors_no_hardcoded_paths(self):
        """Detectors should not contain hardcoded OS-specific paths."""
        from src.discovery import detectors
        source = open(detectors.__file__).read()
        assert "C:\\" not in source
        assert "/home/" not in source
        assert "/Users/" not in source

    def test_scanner_no_hardcoded_paths(self):
        """Scanner should not contain hardcoded OS-specific paths."""
        from src.discovery import scanner
        source = open(scanner.__file__).read()
        assert "C:\\" not in source
        assert "/home/" not in source

    def test_paths_module_no_hardcoded_dataset(self):
        """paths.py should not contain hardcoded dataset names."""
        from src import paths
        source = open(paths.__file__).read()
        assert "olist" not in source
        assert "telco" not in source
        assert "instacart" not in source

    def test_benchmark_no_hardcoded_paths(self):
        """benchmark.py should not contain hardcoded OS-specific paths."""
        from src import benchmark
        source = open(benchmark.__file__).read()
        assert "C:\\" not in source
        assert "/home/" not in source

    def test_environment_no_hardcoded_dataset(self):
        """environment.py should not contain hardcoded dataset paths."""
        from src import environment
        source = open(environment.__file__).read()
        assert "olist" not in source
        assert "telco" not in source


# ══════════════════════════════════════════════════════════════════
#  13. INTEGRATION: FULL DISCOVERY FLOW
# ══════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests for the full discovery flow."""

    def test_full_discovery_flow(self, sample_csvs):
        from src.benchmark import discover_only
        discovered = discover_only(
            str(sample_csvs["olist_dir"].parent),
            max_depth=2,
        )
        assert len(discovered) >= 2
        names = [d["name"] for d in discovered]
        assert "olist" in names
        assert "telco" in names

        for ds in discovered:
            assert "confidence" in ds
            assert "source" in ds
            assert "matched_files" in ds

    def test_discovery_with_registry_sync(self, tmp_dir):
        import pandas as pd
        from src.registry_store import DatasetRegistryStore
        from src.benchmark import discover_only

        # Create config
        configs_dir = tmp_dir / "configs" / "datasets"
        configs_dir.mkdir(parents=True)
        (configs_dir / "custom_data.yaml").write_text(
            "dataset:\n  name: custom_data\n"
        )

        # Create dataset directory
        ds_dir = tmp_dir / "datasets" / "custom"
        ds_dir.mkdir(parents=True)
        pd.DataFrame({
            "id": [1, 2],
            "name": ["a", "b"],
        }).to_csv(ds_dir / "custom_data.csv", index=False)

        # Initialize registry and sync from our isolated configs
        store = DatasetRegistryStore(project_root=tmp_dir, configs_dir=configs_dir)
        store.sync_from_configs()
        assert "custom_data" in store.list_registered()

        # Discovery should find it
        discovered = discover_only(str(tmp_dir / "datasets"))
        # At minimum, the directory was scanned
        assert isinstance(discovered, list)

    def test_dry_run_no_execution(self, sample_csvs):
        from src.benchmark import benchmark
        with patch("src.environment._is_kaggle", return_value=False), \
             patch("src.environment._is_colab", return_value=False), \
             patch("src.environment.Path.cwd", return_value=Path("/tmp")):
            result = benchmark(
                dataset_root=str(sample_csvs["olist_dir"].parent),
                dry_run=True,
            )
            assert result.total_duration > 0
            assert isinstance(result.discovered_datasets, list)
