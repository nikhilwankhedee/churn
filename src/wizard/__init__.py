"""
Dataset registration wizard.

Inspects a CSV file, infers column roles (customer_id, timestamp, monetary, etc.),
and generates a reusable YAML dataset configuration.
"""
from src.wizard.inspector import inspect_csv, ColumnInspection
from src.wizard.generator import generate_config, WizardConfig
from src.wizard.readiness import generate_readiness_report, ReadinessReport

__all__ = [
    "inspect_csv", "ColumnInspection",
    "generate_config", "WizardConfig",
    "generate_readiness_report", "ReadinessReport",
]
