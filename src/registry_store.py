"""
Persistent dataset registry — stores and retrieves dataset registrations.

Maintains a local registry directory with YAML config files for each
registered dataset. Supports automatic recognition of previously
registered datasets regardless of folder name or environment.
"""
import os
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class RegisteredDataset:
    """A registered dataset entry."""
    name: str
    config_path: Path
    schema_fingerprint: str = ""
    required_columns: dict[str, list[str]] = field(default_factory=dict)
    ecosystem_type: str = "unknown"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "config_path": str(self.config_path),
            "schema_fingerprint": self.schema_fingerprint,
            "required_columns": self.required_columns,
            "ecosystem_type": self.ecosystem_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RegisteredDataset":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            config_path=Path(data["config_path"]),
            schema_fingerprint=data.get("schema_fingerprint", ""),
            required_columns=data.get("required_columns", {}),
            ecosystem_type=data.get("ecosystem_type", "unknown"),
            metadata=data.get("metadata", {}),
        )


class DatasetRegistryStore:
    """Persistent storage for dataset registrations.

    Maintains two stores:
    1. YAML configs in configs/datasets/ (existing, unchanged)
    2. Registry index in .dataset_registry/ (new, for auto-recognition)
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        configs_dir: Optional[Path] = None,
    ) -> None:
        if project_root is None:
            from src.config import PROJECT_ROOT
            project_root = Path(PROJECT_ROOT)

        self.project_root = Path(project_root)

        if configs_dir is not None:
            self.configs_dir = Path(configs_dir)
        else:
            # Resolve configs_dir: try src/configs first (installed), then project_root/configs (dev)
            pkg_configs = Path(__file__).resolve().parent / "configs" / "datasets"
            dev_configs = self.project_root / "configs" / "datasets"
            self.configs_dir = pkg_configs if pkg_configs.is_dir() else dev_configs

        self.registry_dir = self.project_root / ".dataset_registry"
        self.index_path = self.registry_dir / "index.json"

        # Ensure directories exist
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> dict[str, dict]:
        """Load the registry index."""
        if not self.index_path.exists():
            return {}
        try:
            with open(self.index_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load registry index: %s", exc)
            return {}

    def _save_index(self, index: dict[str, dict]) -> None:
        """Save the registry index."""
        try:
            with open(self.index_path, "w") as f:
                json.dump(index, f, indent=2)
        except OSError as exc:
            logger.error("Failed to save registry index: %s", exc)

    def _compute_schema_fingerprint(
        self, required_columns: dict[str, list[str]]
    ) -> str:
        """Compute a fingerprint from the schema for comparison."""
        parts = []
        for table in sorted(required_columns.keys()):
            cols = sorted(required_columns[table])
            parts.append(f"{table}:{','.join(cols)}")
        raw = "|".join(parts)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def register(
        self,
        name: str,
        config_path: Path,
        required_columns: Optional[dict[str, list[str]]] = None,
        ecosystem_type: str = "unknown",
        metadata: Optional[dict] = None,
    ) -> RegisteredDataset:
        """Register a dataset in the persistent registry.

        Parameters
        ----------
        name : str
            Dataset name.
        config_path : Path
            Path to the YAML configuration file.
        required_columns : dict, optional
            Mapping of table/filename -> list of required column names.
        ecosystem_type : str
            Dataset ecosystem type.
        metadata : dict, optional
            Additional metadata to store.

        Returns
        -------
        RegisteredDataset entry.
        """
        fingerprint = self._compute_schema_fingerprint(required_columns or {})

        entry = RegisteredDataset(
            name=name,
            config_path=Path(config_path),
            schema_fingerprint=fingerprint,
            required_columns=required_columns or {},
            ecosystem_type=ecosystem_type,
            metadata=metadata or {},
        )

        # Update index
        index = self._load_index()
        index[name] = entry.to_dict()
        self._save_index(index)

        logger.info(
            "Registered dataset '%s' (fingerprint: %s, config: %s)",
            name, fingerprint, config_path,
        )

        return entry

    def get(self, name: str) -> Optional[RegisteredDataset]:
        """Get a registered dataset by name."""
        index = self._load_index()
        if name not in index:
            return None
        return RegisteredDataset.from_dict(index[name])

    def list_registered(self) -> list[str]:
        """List all registered dataset names."""
        index = self._load_index()
        return sorted(index.keys())

    def get_all(self) -> dict[str, RegisteredDataset]:
        """Get all registered datasets."""
        index = self._load_index()
        return {
            name: RegisteredDataset.from_dict(data)
            for name, data in index.items()
        }

    def get_schema_info(self) -> dict[str, dict]:
        """Get schema information for all registered datasets.

        Returns dict suitable for passing to scan_directory's
        previously_registered parameter.
        """
        index = self._load_index()
        result = {}
        for name, data in index.items():
            result[name] = {
                "required_columns": data.get("required_columns", {}),
                "ecosystem_type": data.get("ecosystem_type", "unknown"),
                "schema_fingerprint": data.get("schema_fingerprint", ""),
            }
        return result

    def remove(self, name: str) -> bool:
        """Remove a dataset from the registry."""
        index = self._load_index()
        if name not in index:
            return False
        del index[name]
        self._save_index(index)
        logger.info("Removed dataset '%s' from registry", name)
        return True

    def sync_from_configs(self) -> int:
        """Synchronize registry from existing YAML configs in configs/datasets/.

        Scans configs/datasets/ for YAML files and registers any that
        are not yet in the registry index. Returns the number of new
        registrations.

        This ensures backward compatibility — existing YAML configs
        are automatically recognized.
        """
        if not self.configs_dir.is_dir():
            return 0

        index = self._load_index()
        count = 0

        for yaml_file in sorted(self.configs_dir.glob("*.yaml")):
            dataset_name = yaml_file.stem

            if dataset_name in index:
                continue  # Already registered

            # Parse minimal info from the YAML
            try:
                import yaml
                with open(yaml_file) as f:
                    config = yaml.safe_load(f) or {}
            except Exception:
                continue

            # Extract required columns from schema section
            required_columns = {}
            schema = config.get("schema", {})
            column_mapping = schema.get("column_mapping", {})
            if column_mapping:
                # Store as a synthetic "schema" entry
                required_columns["_schema_mapping"] = list(column_mapping.keys())

            # Extract ecosystem type
            dataset_info = config.get("dataset", {})
            ecosystem_type = dataset_info.get("ecosystem_type", "unknown")

            self.register(
                name=dataset_name,
                config_path=yaml_file,
                required_columns=required_columns,
                ecosystem_type=ecosystem_type,
                metadata=dataset_info,
            )
            count += 1
            logger.info("Synced config to registry: %s", dataset_name)

        return count

    def ensure_synced(self) -> None:
        """Ensure the registry is synced with existing YAML configs."""
        n = self.sync_from_configs()
        if n > 0:
            logger.info("Synced %d new dataset(s) from configs", n)


def get_registry_store(
    project_root: Optional[Path] = None,
) -> DatasetRegistryStore:
    """Get or create the dataset registry store."""
    return DatasetRegistryStore(project_root)
