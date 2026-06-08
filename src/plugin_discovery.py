"""
Plugin discovery system.

Automatically discovers and registers plugins from:
1. Built-in plugins (registered in package __init__.py files)
2. Local plugins directory (configs/plugins/)
3. Installed packages with entry points (future)

Usage:
    from src.plugins import discover_plugins

    # Discover and register all plugins
    discover_plugins()

    # List discovered plugins
    from src.plugins import list_plugins
    plugins = list_plugins()
"""
import importlib
import importlib.metadata
import os
from typing import Any, Dict, List, Optional

from src.config import PROJECT_ROOT
from src.core.registry import registry
from src.utils import get_logger

logger = get_logger(__name__)

# Plugin categories that can be extended
PLUGIN_CATEGORIES = [
    "datasets",
    "churn_strategies",
    "models",
    "metrics",
    "resamplers",
    "reports",
]

# Local plugins directory
PLUGINS_DIR = os.path.join(PROJECT_ROOT, "plugins")


def discover_plugins(
    local_dir: Optional[str] = None,
    entry_point_group: str = "churn_framework.plugins",
) -> Dict[str, List[str]]:
    """Discover and register all available plugins.

    Parameters
    ----------
    local_dir : str, optional
        Path to local plugins directory. Default: configs/plugins/
    entry_point_group : str
        Entry point group name for installed package discovery.

    Returns
    -------
    Dict mapping category -> list of discovered plugin names.
    """
    discovered: Dict[str, List[str]] = {cat: [] for cat in PLUGIN_CATEGORIES}

    # 1. Discover from local plugins directory
    if local_dir is None:
        local_dir = PLUGINS_DIR

    if os.path.isdir(local_dir):
        local_plugins = _discover_local_plugins(local_dir)
        for cat, names in local_plugins.items():
            discovered[cat].extend(names)

    # 2. Discover from installed packages (entry points)
    entry_plugins = _discover_entry_points(entry_point_group)
    for cat, names in entry_plugins.items():
        discovered[cat].extend(names)

    # Log summary
    total = sum(len(v) for v in discovered.values())
    if total > 0:
        logger.info(
            "Plugin discovery complete — %d plugins found across %d categories",
            total, len([c for c in discovered if discovered[c]]),
        )

    return discovered


def _discover_local_plugins(plugins_dir: str) -> Dict[str, List[str]]:
    """Discover plugins from a local directory.

    Expected structure:
        plugins/
            my_plugin/
                __init__.py  (must define register_plugin(registry) function)

    Returns
    -------
    Dict mapping category -> list of registered plugin names.
    """
    discovered: Dict[str, List[str]] = {cat: [] for cat in PLUGIN_CATEGORIES}

    if not os.path.isdir(plugins_dir):
        return discovered

    for entry in os.listdir(plugins_dir):
        plugin_path = os.path.join(plugins_dir, entry)
        if not os.path.isdir(plugin_path):
            continue
        if entry.startswith("_") or entry.startswith("."):
            continue

        init_file = os.path.join(plugin_path, "__init__.py")
        if not os.path.isfile(init_file):
            continue

        try:
            # Import the plugin module
            module_name = f"plugins.{entry}"
            mod = importlib.import_module(module_name)

            # Call register_plugin if it exists
            if hasattr(mod, "register_plugin"):
                mod.register_plugin(registry)
                logger.info("Loaded local plugin: %s", entry)

                # Track what was registered
                for cat in PLUGIN_CATEGORIES:
                    new_names = registry.list_registered(cat)
                    # This is approximate — we don't track pre-existing
                    discovered[cat] = new_names

        except Exception as exc:
            logger.warning("Failed to load plugin '%s': %s", entry, exc)

    return discovered


def _discover_entry_points(group: str) -> Dict[str, List[str]]:
    """Discover plugins from installed package entry points.

    Packages can declare plugins via setup.cfg/pyproject.toml:
        [options.entry_points]
        churn_framework.plugins =
            my_plugin = my_package.plugin:register

    Returns
    -------
    Dict mapping category -> list of registered plugin names.
    """
    discovered: Dict[str, List[str]] = {cat: [] for cat in PLUGIN_CATEGORIES}

    try:
        eps = importlib.metadata.entry_points()
    except Exception:
        return discovered

    # Handle different Python versions
    if hasattr(eps, "select"):
        plugin_points = eps.select(group=group)
    elif isinstance(eps, dict):
        plugin_points = eps.get(group, [])
    else:
        plugin_points = []

    for ep in plugin_points:
        try:
            register_fn = ep.load()
            if callable(register_fn):
                register_fn(registry)
                logger.info("Loaded entry point plugin: %s", ep.name)
        except Exception as exc:
            logger.warning("Failed to load entry point '%s': %s", ep.name, exc)

    return discovered


def list_plugins() -> Dict[str, List[str]]:
    """List all currently registered plugins by category."""
    result = {}
    for cat in PLUGIN_CATEGORIES:
        names = registry.list_registered(cat)
        if names:
            result[cat] = names
    return result


def get_plugin_info(name: str, category: str) -> Dict[str, Any]:
    """Get detailed information about a registered plugin."""
    info = {
        "name": name,
        "category": category,
        "registered": registry.is_registered(name, category),
    }

    if registry.is_registered(name, category):
        # Check if it's lazy, class, or instance
        if category in registry._instances and name in registry._instances.get(category, {}):
            info["type"] = "instance"
        elif category in registry._classes and name in registry._classes.get(category, {}):
            info["type"] = "class"
        else:
            info["type"] = "lazy"
            info["dotted_path"] = registry._paths.get(category, {}).get(name, "")

        # Get metadata
        metadata = registry.get_metadata(name, category)
        if metadata:
            info["metadata"] = metadata

    return info
