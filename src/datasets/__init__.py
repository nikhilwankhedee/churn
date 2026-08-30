"""
Dataset registry — lazy-loaded adapters keyed by dataset name.

Usage:
    from src.datasets import get_dataset, list_datasets
    adapter = get_dataset("olist")
    df = adapter.load_raw_data()
"""
from typing import Dict, Type, Optional
from src.utils import get_logger

logger = get_logger(__name__)

_REGISTRY: Dict[str, str] = {
    "olist": "src.datasets.olist.OlistAdapter",
    "rees46": "src.datasets.rees46.REES46Adapter",
    "retailrocket": "src.datasets.retailrocket.RetailRocketAdapter",
    "online_retail_ii": "src.datasets.online_retail_ii.OnlineRetailIIAdapter",
    "instacart": "src.datasets.instacart.InstacartAdapter",
    "telco": "src.datasets.telco.TelcoAdapter",
    "lastfm": "src.datasets.lastfm.LastFMAdapter",
    "credit_card": "src.datasets.credit_card.CreditCardAdapter",
}

_ECOSYSTEM_TYPES = {
    "olist": "transactional_marketplace",
    "rees46": "transactional_marketplace",
    "retailrocket": "clickstream_commerce",
    "online_retail_ii": "habitual_retail",
    "instacart": "habitual_retail",
    "telco": "subscription",
    "lastfm": "music_streaming",
    "credit_card": "banking",
}

_INSTANCE_CACHE: Dict[str, object] = {}


def list_datasets() -> list:
    return sorted(_REGISTRY.keys())


def get_dataset(name: str):
    name = name.lower().strip()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list_datasets()}"
        )
    if name in _INSTANCE_CACHE:
        return _INSTANCE_CACHE[name]

    module_path, class_name = _REGISTRY[name].rsplit(".", 1)
    try:
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        instance = cls()
        _INSTANCE_CACHE[name] = instance
        logger.info("Loaded dataset adapter: %s", name)
        return instance
    except Exception as exc:
        logger.error("Failed to load dataset '%s': %s", name, exc)
        raise


def get_ecosystem_type(name: str) -> str:
    return _ECOSYSTEM_TYPES.get(name.lower(), "unknown")
