"""
Dataset registry — unified adapter resolution from manifests and built-in adapters.

Resolution order:
  1. Built-in adapter registry (_REGISTRY) — for known datasets with custom logic
  2. Manifest lookup — for datasets with configs/datasets/{name}.yaml
  3. GenericDatasetAdapter — for any dataset with a manifest

Usage:
    from src.datasets import get_dataset, list_datasets
    adapter = get_dataset("olist")
    adapter = get_dataset("my_custom_dataset")  # auto-routes to GenericDatasetAdapter
    df = adapter.load_raw_data()
"""
import os
from pathlib import Path
from typing import Dict, Optional, List

from src.utils import get_logger

logger = get_logger(__name__)

# ── Built-in adapter registry (backward-compatible) ────────────────
# These datasets have custom Python logic (multi-table joins, domain-specific
# preprocessing) that cannot be expressed declaratively. They remain as
# dedicated adapter classes but read their configuration from manifests.
_BUILTIN_ADAPTERS: Dict[str, str] = {
    "olist": "src.datasets.olist.OlistAdapter",
    "rees46": "src.datasets.rees46.REES46Adapter",
    "retailrocket": "src.datasets.retailrocket.RetailRocketAdapter",
    "online_retail_ii": "src.datasets.online_retail_ii.OnlineRetailIIAdapter",
    "instacart": "src.datasets.instacart.InstacartAdapter",
    "telco": "src.datasets.telco.TelcoAdapter",
    "credit_card": "src.datasets.credit_card.CreditCardAdapter",
    "lastfm": "src.datasets.lastfm.LastFMAdapter",
    "kkbox": "src.datasets.kkbox.KKBoxAdapter",
}

_ECOSYSTEM_TYPES: Dict[str, str] = {
    "olist": "transactional_marketplace",
    "rees46": "transactional_marketplace",
    "retailrocket": "clickstream_commerce",
    "online_retail_ii": "habitual_retail",
    "instacart": "habitual_retail",
    "telco": "subscription",
    "credit_card": "subscription",
    "lastfm": "media_streaming",
    "kkbox": "subscription",
}

_INSTANCE_CACHE: Dict[str, object] = {}


def _discover_manifest_datasets() -> List[str]:
    """Find all datasets that have manifest YAML files."""
    names = []
    try:
        from src.config import get_configs_dir
        configs_dir = get_configs_dir()
        datasets_dir = configs_dir / "datasets"
        if datasets_dir.is_dir():
            for yaml_file in datasets_dir.glob("*.yaml"):
                name = yaml_file.stem
                if name not in _BUILTIN_ADAPTERS:
                    names.append(name)
    except Exception:
        pass

    # Also check .dataset_registry/manifests/ for user-registered datasets
    try:
        from src.config import PROJECT_ROOT
        registry_dir = Path(PROJECT_ROOT) / ".dataset_registry" / "manifests"
        if registry_dir.is_dir():
            for yaml_file in registry_dir.glob("*.yaml"):
                name = yaml_file.stem
                if name not in _BUILTIN_ADAPTERS and name not in names:
                    names.append(name)
    except Exception:
        pass

    return sorted(names)


def _register_with_core_registry() -> None:
    """Register all adapters with the core PluginRegistry."""
    try:
        from src.core.registry import registry
        for name, dotted_path in _BUILTIN_ADAPTERS.items():
            if not registry.is_registered(name, "datasets"):
                metadata = {"ecosystem_type": _ECOSYSTEM_TYPES.get(name, "unknown")}
                registry.register(name, "datasets", dotted_path, metadata=metadata)
    except ImportError:
        pass


_register_with_core_registry()


def list_datasets() -> List[str]:
    """List all available datasets (built-in + manifest-discovered)."""
    all_names = set(_BUILTIN_ADAPTERS.keys())
    all_names.update(_discover_manifest_datasets())
    return sorted(all_names)


def get_dataset(name: str, data_dir: Optional[str] = None):
    """Get a dataset adapter by name.

    Resolution order:
      1. Built-in adapter (_REGISTRY) — for known datasets with custom logic
      2. Manifest file — creates GenericDatasetAdapter
      3. Raises ValueError if nothing found

    Parameters
    ----------
    name : str
        Dataset name (e.g. 'olist', 'retailrocket', 'my_custom').
    data_dir : str, optional
        Explicit directory containing raw data files.

    Returns
    -------
    BaseDatasetAdapter instance with data_dir configured.
    """
    name = name.lower().strip()

    # 1. Built-in adapter
    if name in _BUILTIN_ADAPTERS:
        if data_dir is not None:
            return _create_builtin_adapter(name, data_dir)
        if name in _INSTANCE_CACHE:
            return _INSTANCE_CACHE[name]
        instance = _create_builtin_adapter(name, data_dir=None)
        _INSTANCE_CACHE[name] = instance
        return instance

    # 2. Manifest-driven dataset → GenericDatasetAdapter
    manifest = _try_load_manifest(name)
    if manifest is not None:
        from src.datasets.generic import GenericDatasetAdapter
        adapter = GenericDatasetAdapter(manifest_dict=manifest)

        # Resolve data_dir from manifest or explicit param
        resolved_dir = data_dir or manifest.get("root_directory")
        if resolved_dir:
            adapter.data_dir = resolved_dir

        logger.info("Loaded manifest-driven dataset: %s", name)
        return adapter

    # 3. Nothing found
    all_available = list_datasets()
    raise ValueError(
        f"Unknown dataset '{name}'. Available: {all_available}"
    )


def _create_builtin_adapter(name: str, data_dir: Optional[str] = None):
    """Instantiate a built-in adapter."""
    module_path, class_name = _BUILTIN_ADAPTERS[name].rsplit(".", 1)
    try:
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        instance = cls()

        if data_dir is not None:
            instance.data_dir = data_dir

        logger.info("Loaded built-in adapter: %s", name)
        return instance
    except Exception as exc:
        logger.error("Failed to load adapter '%s': %s", name, exc)
        raise


def _try_load_manifest(name: str) -> Optional[dict]:
    """Try to load a manifest for the given dataset name."""
    try:
        import yaml

        # Try configs/datasets/
        from src.config import get_configs_dir
        configs_dir = get_configs_dir()
        manifest_path = configs_dir / "datasets" / f"{name}.yaml"
        if manifest_path.exists():
            with open(manifest_path) as f:
                return yaml.safe_load(f) or {}

        # Try .dataset_registry/manifests/
        from src.config import PROJECT_ROOT
        registry_path = Path(PROJECT_ROOT) / ".dataset_registry" / "manifests" / f"{name}.yaml"
        if registry_path.exists():
            with open(registry_path) as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return None


def get_ecosystem_type(name: str) -> str:
    """Get ecosystem type for a dataset."""
    name = name.lower().strip()

    # Check built-in registry
    if name in _ECOSYSTEM_TYPES:
        return _ECOSYSTEM_TYPES[name]

    # Check manifest
    manifest = _try_load_manifest(name)
    if manifest:
        return manifest.get("dataset", {}).get("ecosystem_type", "unknown")

    return "unknown"


def register_dataset(
    name: str,
    manifest_path: str,
    ecosystem_type: str = "unknown",
) -> None:
    """Register a new dataset from a manifest file.

    This makes the dataset discoverable via list_datasets() and get_dataset().

    Parameters
    ----------
    name : str
        Dataset name.
    manifest_path : str
        Path to the manifest YAML file.
    ecosystem_type : str
        Dataset ecosystem type.
    """
    # Copy manifest to the persistent registry location
    from src.config import PROJECT_ROOT
    registry_dir = Path(PROJECT_ROOT) / ".dataset_registry" / "manifests"
    registry_dir.mkdir(parents=True, exist_ok=True)

    target = registry_dir / f"{name}.yaml"
    import shutil
    shutil.copy2(manifest_path, target)

    logger.info("Registered dataset '%s' from %s", name, manifest_path)

    # Update ecosystem types cache
    try:
        import yaml
        with open(target) as f:
            manifest = yaml.safe_load(f) or {}
        eco = manifest.get("dataset", {}).get("ecosystem_type", ecosystem_type)
        _ECOSYSTEM_TYPES[name] = eco
    except Exception:
        _ECOSYSTEM_TYPES[name] = ecosystem_type
