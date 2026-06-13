"""Plugin scaffolding and discovery."""
from src.plugins.scaffold import create_plugin
from src.plugin_discovery import (
    discover_plugins,
    list_plugins,
    get_plugin_info,
)

__all__ = [
    "create_plugin",
    "discover_plugins",
    "list_plugins",
    "get_plugin_info",
]
