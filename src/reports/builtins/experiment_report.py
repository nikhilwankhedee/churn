"""
Experiment Report.

Full experiment record including configuration, environment,
timestamps, and all captured metadata. Designed for reproducibility.
"""
from typing import Any, Dict

from src.config import FRAMEWORK_VERSION
from src.reports.base import Report, ReportOutput, ReportSection


class ExperimentReport(Report):
    """Full experiment record for reproducibility."""

    @property
    def name(self) -> str:
        return "experiment_report"

    @property
    def title(self) -> str:
        return "Experiment Report"

    @property
    def description(self) -> str:
        return (
            "Complete experiment record: configuration, environment, "
            "timestamps, and all captured metadata."
        )

    def generate(
        self,
        pipeline_result: Dict[str, Any],
        **kwargs: Any,
    ) -> ReportOutput:
        sections = []
        dataset = pipeline_result.get("dataset", "unknown")

        # Configuration
        config = (
            f"**Dataset**: {dataset}\n"
            f"**Ecosystem**: {pipeline_result.get('ecosystem_type', 'N/A')}\n"
            f"**Churn rate**: {pipeline_result.get('churn_rate', 0):.2%}\n"
            f"**Imbalance ratio**: {pipeline_result.get('imbalance_ratio', 0):.2f}:1\n"
            f"**Best model**: {pipeline_result.get('best_model', 'N/A')}\n"
            f"**Dominant feature group**: {pipeline_result.get('dominant_feature_group', 'N/A')}\n"
        )
        sections.append(ReportSection("Configuration", config))

        # Timing
        duration = pipeline_result.get("duration_seconds", 0)
        timing = (
            f"**Total pipeline duration**: {duration:.1f}s\n"
            f"**Framework version**: {FRAMEWORK_VERSION}\n"
        )
        sections.append(ReportSection("Timing", timing))

        # Validation Summary
        validation = (
            f"| Check | Result |\n"
            f"|-------|--------|\n"
            f"| Schema errors | {pipeline_result.get('schema_errors', 0)} |\n"
            f"| Schema warnings | {pipeline_result.get('schema_warnings', 0)} |\n"
            f"| Behavioral warnings | {pipeline_result.get('behavioral_warnings', 0)} |\n"
            f"| Missing output files | {pipeline_result.get('output_files_missing', 0)} |\n"
        )
        sections.append(ReportSection("Validation Summary", validation))

        # Reproducibility
        reproducibility = (
            "To reproduce this experiment:\n\n"
            "```bash\n"
            f"python -m src.cli run {dataset}\n"
            "```\n\n"
            "All random seeds are fixed (seed=42). "
            "Temporal splits are deterministic (quantile-based). "
            "The experiment log is saved in `results/experiments/experiment_log.csv`."
        )
        sections.append(ReportSection("Reproducibility", reproducibility))

        metadata = {
            "dataset": dataset,
            "report_type": "experiment",
            "duration_seconds": f"{duration:.1f}",
        }

        return ReportOutput(
            name=self.name,
            title=self.title,
            sections=sections,
            metadata=metadata,
        )
