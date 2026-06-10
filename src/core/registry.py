"""
Universal plugin registry for the churn research framework.

Supports lazy loading via dotted module paths, instance caching,
and category-based organization. Every extensible component
(datasets, models, metrics, churn strategies, reports, feature groups,
explainability methods) registers through this single registry.

Usage:
    from src.core.registry import registry

    # Register a class by dotted path (lazy-loaded)
    registry.register("my_model", "models", "my_package.models.MyModel")

    # Register an instance directly
    registry.register_instance("my_metric", "metrics", my_metric_obj)

    # Retrieve
    model_cls = registry.get("my_model", "models")
    model = registry.get_instance("my_model", "models")

    # List
    registry.list_categories()
    registry.list_registered("models")
"""
import importlib
from typing import Any, Dict, List, Optional, Type


class PluginRegistry:
    """Central registry for all pluggable framework components."""

    def __init__(self):
        self._paths: Dict[str, Dict[str, str]] = {}
        self._instances: Dict[str, Dict[str, Any]] = {}
        self._classes: Dict[str, Dict[str, type]] = {}

    # ── Registration ──────────────────────────────────────────────

    def register(
        self,
        name: str,
        category: str,
        dotted_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a plugin by its dotted module.ClassPath.

        The class is not imported until first access (lazy loading).
        """
        cat = self._ensure_category(category)
        cat[name] = dotted_path
        if metadata:
            key = f"_meta_{category}_{name}"
            self._paths.setdefault("_metadata", {})[key] = metadata

    def register_class(
        self,
        name: str,
        category: str,
        cls: type,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register an already-imported class directly."""
        cat = self._ensure_class_category(category)
        cat[name] = cls
        if metadata:
            key = f"_meta_{category}_{name}"
            self._paths.setdefault("_metadata", {})[key] = metadata

    def register_instance(
        self,
        name: str,
        category: str,
        instance: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a pre-instantiated object directly."""
        cat = self._ensure_instance_category(category)
        cat[name] = instance
        if metadata:
            key = f"_meta_{category}_{name}"
            self._paths.setdefault("_metadata", {})[key] = metadata

    # ── Retrieval ─────────────────────────────────────────────────

    def get_class(self, name: str, category: str) -> type:
        """Retrieve a registered class (imports if lazy-loaded)."""
        # Check direct class registrations first
        if category in self._classes and name in self._classes[category]:
            return self._classes[category][name]

        # Check lazy-loaded paths
        if category in self._paths and name in self._paths[category]:
            dotted_path = self._paths[category][name]
            return self._import(dotted_path)

        raise KeyError(
            f"No class registered as '{name}' in category '{category}'. "
            f"Available: {self.list_registered(category)}"
        )

    def get_instance(self, name: str, category: str) -> Any:
        """Retrieve a registered instance (creates from class if needed)."""
        # Check direct instance registrations first
        if category in self._instances and name in self._instances[category]:
            return self._instances[category][name]

        # Try to create from registered class
        cls = self.get_class(name, category)
        instance = cls()
        self._ensure_instance_category(category)[name] = instance
        return instance

    def get_metadata(self, name: str, category: str) -> Dict[str, Any]:
        """Retrieve metadata for a registered plugin."""
        key = f"_meta_{category}_{name}"
        return self._paths.get("_metadata", {}).get(key, {})

    # ── Listing ───────────────────────────────────────────────────

    def list_categories(self) -> List[str]:
        """Return all registered category names."""
        cats = set()
        cats.update(self._paths.keys())
        cats.update(self._instances.keys())
        cats.update(self._classes.keys())
        cats.discard("_metadata")
        return sorted(cats)

    def list_registered(self, category: str) -> List[str]:
        """Return all registered names in a category."""
        names = set()
        if category in self._paths:
            names.update(self._paths[category].keys())
        if category in self._instances:
            names.update(self._instances[category].keys())
        if category in self._classes:
            names.update(self._classes[category].keys())
        return sorted(names)

    def is_registered(self, name: str, category: str) -> bool:
        """Check if a name is registered in a category."""
        if category in self._paths and name in self._paths[category]:
            return True
        if category in self._instances and name in self._instances[category]:
            return True
        if category in self._classes and name in self._classes[category]:
            return True
        return False

    # ── Internal ──────────────────────────────────────────────────

    def _ensure_category(self, category: str) -> Dict[str, str]:
        self._paths.setdefault(category, {})
        return self._paths[category]

    def _ensure_class_category(self, category: str) -> Dict[str, type]:
        self._classes.setdefault(category, {})
        return self._classes[category]

    def _ensure_instance_category(self, category: str) -> Dict[str, Any]:
        self._instances.setdefault(category, {})
        return self._instances[category]

    @staticmethod
    def _import(dotted_path: str) -> type:
        """Import a class from a dotted module.Class path."""
        module_path, class_name = dotted_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)


# Global singleton
registry = PluginRegistry()
