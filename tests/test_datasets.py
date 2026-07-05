"""Tests for dataset adapters."""
import os
import sys
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestDatasetRegistry:
    """Tests for dataset registration and listing."""

    def test_list_datasets(self):
        from src.datasets import list_datasets
        ds = list_datasets()
        assert isinstance(ds, list)
        assert len(ds) >= 6
        assert "olist" in ds
        assert "rees46" in ds
        assert "telco" in ds

    def test_get_dataset(self):
        from src.datasets import get_dataset, list_datasets
        for name in list_datasets():
            adapter = get_dataset(name)
            assert adapter.dataset_name == name

    def test_get_ecosystem_type(self):
        from src.datasets import get_ecosystem_type
        eco = get_ecosystem_type("olist")
        assert isinstance(eco, str)
        assert len(eco) > 0

    def test_all_adapters_have_metadata(self):
        from src.datasets import get_dataset, list_datasets
        for name in list_datasets():
            adapter = get_dataset(name)
            meta = adapter.metadata
            assert "dataset_name" in meta
            assert "ecosystem_type" in meta

    def test_all_adapters_have_feature_groups(self):
        from src.datasets import get_dataset, list_datasets
        for name in list_datasets():
            adapter = get_dataset(name)
            groups = adapter.available_feature_groups
            assert isinstance(groups, list)
            assert len(groups) > 0


class TestBaseDatasetAdapter:
    """Tests for the base adapter interface."""

    def test_feature_group_mapping(self):
        from src.datasets import get_dataset
        adapter = get_dataset("olist")
        groups = adapter.get_feature_groups()
        assert "purchase" in groups
        assert "monetary" in groups


class TestAdapterFileFallbacks:
    """Kaggle-name alternates let the resolver find raw files."""

    def test_rees46_alternate_filenames(self):
        from src.datasets import get_dataset
        a = get_dataset("rees46")
        alts = a.alternate_filenames
        assert "rees46_events.csv" in alts
        assert "events.csv" in alts["rees46_events.csv"]

    def test_retailrocket_alternate_filenames(self):
        from src.datasets import get_dataset
        a = get_dataset("retailrocket")
        alts = a.alternate_filenames
        assert "retailrocket_events.csv" in alts
        assert "events.csv" in alts["retailrocket_events.csv"]

    def test_instacart_alternate_filenames(self):
        from src.datasets import get_dataset
        a = get_dataset("instacart")
        assert "orders.csv" in a.alternate_filenames["instacart_orders.csv"]

    def test_online_retail_ii_alternate_filenames(self):
        from src.datasets import get_dataset
        a = get_dataset("online_retail_ii")
        for required in a.required_files:
            assert "online_retail_II.xlsx" in a.alternate_filenames[required]

    def test_credit_card_alternate_filenames(self):
        from src.datasets import get_dataset
        a = get_dataset("credit_card")
        assert "BankChurners.csv" in a.alternate_filenames["credit_card_customers.csv"]


class TestRetailRocketLongFormatItems:
    """Kaggle item_properties_part1.csv is long-format (item×property rows);
    merging it with events must not blow up the row count."""

    def _adapter_with_dir(self, tmp_path):
        from src.datasets.retailrocket import RetailRocketAdapter
        a = RetailRocketAdapter()
        a.data_dir = str(tmp_path)
        return a

    def _write(self, tmp_path):
        pd = __import__('pandas')
        pd.DataFrame({
            'visitorid': ['1', '1', '2'],
            'timestamp': [0, 1, 2],
            'itemid': ['a', 'b', 'a'],
            'event': ['view', 'view', 'transaction'],
        }).to_csv(os.path.join(str(tmp_path), 'events.csv'), index=False)
        pd.DataFrame({
            'itemid': ['a', 'a', 'b', 'b', 'b'],
            'timestamp': [0, 0, 0, 0, 0],
            'property': ['cat', 'brand', 'cat', 'brand', 'price'],
            'value': ['x', 'y', 'z', 'w', '9'],
        }).to_csv(os.path.join(str(tmp_path), 'item_properties_part1.csv'),
                  index=False)

    def test_no_cartesian_explosion(self, tmp_path):
        self._write(tmp_path)
        a = self._adapter_with_dir(tmp_path)
        df = a.load_raw_data()
        assert len(df) == 3
        assert 'n_properties' in df.columns

    def test_wide_items_merge_unchanged(self, tmp_path):
        import pandas as pd
        self._write(tmp_path)
        pd.DataFrame({
            'itemid': ['a', 'b'],
            'category': ['x', 'z'],
        }).to_csv(os.path.join(str(tmp_path), 'item_properties_part1.csv'),
                  index=False)
        a = self._adapter_with_dir(tmp_path)
        df = a.load_raw_data()
        assert len(df) == 3
        assert 'category' in df.columns


class TestREES46CustomerModelRefused:
    """customer_model.csv has no per-event timestamps and must not silently
    substitute for the events file."""

    def test_customer_model_only_raises(self, tmp_path):
        import pandas as pd
        from src.datasets.rees46 import REES46Adapter
        pd.DataFrame({'user_id': ['1'], 'f0': [1]}).to_csv(
            os.path.join(str(tmp_path), 'customer_model.csv'), index=False)
        a = REES46Adapter()
        a.data_dir = str(tmp_path)
        with pytest.raises(FileNotFoundError):
            a.load_raw_data()

    def test_events_file_loads(self, tmp_path):
        import pandas as pd
        from src.datasets.rees46 import REES46Adapter
        pd.DataFrame({
            'user_id': ['1', '1'], 'timestamp': [0, 1],
            'event_type': ['view', 'purchase'], 'item_id': ['a', 'a'],
            'price': [0, 10],
        }).to_csv(os.path.join(str(tmp_path), 'events.csv'), index=False)
        a = REES46Adapter()
        a.data_dir = str(tmp_path)
        df = a.load_raw_data()
        assert len(df) == 2
