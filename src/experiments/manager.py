"""
Experiment Manager: track, list, and reproduce pipeline runs.

Every pipeline execution automatically creates an experiment record.
Records are stored in .experiments/ as JSON files.
"""
import dataclasses
import datetime
import json
import os
import hashlib
from typing import Any, Dict, List, Optional

from src.utils import get_logger

logger = get_logger(__name__)

EXPERIMENTS_DIR = ".experiments"


@dataclasses.dataclass
class ExperimentRecord:
    """A single experiment record."""
    experiment_id: str
    dataset: str
    timestamp: str
    framework_version: str
    manifest_version: str = ""
    parameters: Dict[str, Any] = dataclasses.field(default_factory=dict)
    metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
    runtime_seconds: float = 0.0
    hardware: str = ""
    reports: List[str] = dataclasses.field(default_factory=list)
    model_artifacts: List[str] = dataclasses.field(default_factory=list)
    random_seed: int = 42
    status: str = "completed"
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ExperimentManager:
    """Manages experiment records for the framework."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or EXPERIMENTS_DIR
        os.makedirs(self.base_dir, exist_ok=True)

    def _generate_id(self, dataset: str) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"experiment_{dataset}_{ts}"

    def _get_hardware_info(self) -> str:
        import platform
        return f"{platform.machine()} | {platform.processor()}"

    def create_experiment(
        self,
        dataset: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ExperimentRecord:
        """Create a new experiment record."""
        from src.config import FRAMEWORK_VERSION, RANDOM_SEED

        exp_id = self._generate_id(dataset)
        record = ExperimentRecord(
            experiment_id=exp_id,
            dataset=dataset,
            timestamp=datetime.datetime.now().isoformat(),
            framework_version=FRAMEWORK_VERSION,
            parameters=parameters or {},
            hardware=self._get_hardware_info(),
            random_seed=RANDOM_SEED,
        )

        self._save(record)
        logger.info("Created experiment: %s", exp_id)
        return record

    def finish_experiment(
        self,
        record: ExperimentRecord,
        metrics: Optional[Dict[str, float]] = None,
        runtime_seconds: float = 0.0,
        reports: Optional[List[str]] = None,
        status: str = "completed",
        error: str = "",
    ) -> ExperimentRecord:
        """Update an experiment record with results."""
        if metrics:
            record.metrics.update(metrics)
        record.runtime_seconds = runtime_seconds
        record.status = status
        record.error = error
        if reports:
            record.reports.extend(reports)
        self._save(record)
        logger.info("Finished experiment: %s (%s)", record.experiment_id, status)
        return record

    def list_experiments(
        self,
        dataset: Optional[str] = None,
        limit: int = 50,
    ) -> List[ExperimentRecord]:
        """List experiment records, optionally filtered by dataset."""
        records = []
        if not os.path.isdir(self.base_dir):
            return records

        for fname in sorted(os.listdir(self.base_dir), reverse=True):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self.base_dir, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                record = ExperimentRecord.from_dict(data)
                if dataset and record.dataset != dataset:
                    continue
                records.append(record)
                if len(records) >= limit:
                    break
            except Exception:
                continue

        return records

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Get a specific experiment by ID."""
        path = os.path.join(self.base_dir, f"{experiment_id}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return ExperimentRecord.from_dict(data)
        except Exception:
            return None

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment record."""
        path = os.path.join(self.base_dir, f"{experiment_id}.json")
        if os.path.exists(path):
            os.remove(path)
            logger.info("Deleted experiment: %s", experiment_id)
            return True
        return False

    def get_leaderboard(
        self,
        dataset: Optional[str] = None,
        metric: str = "roc_auc",
    ) -> List[Dict[str, Any]]:
        """Get a leaderboard of experiments sorted by a metric."""
        records = self.list_experiments(dataset=dataset, limit=100)

        entries = []
        for r in records:
            if metric in r.metrics:
                entries.append({
                    "experiment_id": r.experiment_id,
                    "dataset": r.dataset,
                    "timestamp": r.timestamp,
                    "metric_value": r.metrics[metric],
                    "runtime": r.runtime_seconds,
                    "status": r.status,
                })

        entries.sort(key=lambda x: x["metric_value"], reverse=True)
        return entries

    def _save(self, record: ExperimentRecord) -> None:
        """Save an experiment record to disk."""
        path = os.path.join(self.base_dir, f"{record.experiment_id}.json")
        with open(path, "w") as f:
            json.dump(record.to_dict(), f, indent=2, default=str)
