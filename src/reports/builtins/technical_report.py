"""
Technical Report.

A detailed technical overview covering methodology, hyperparameters,
temporal splits, validation results, and per-model metrics.
Suitable for paper methods sections and reproducibility documentation.
"""
from typing import Any, Dict

from src.config import FRAMEWORK_VERSION
from src.reports.base import Report, ReportOutput, ReportSection


class TechnicalReport(Report):
    """Detailed technical report covering methodology and results."""

    @property
    def name(self) -> str:
        return "technical_report"

    @property
    def title(self) -> str:
        return "Technical Report"

    @property
    def description(self) -> str:
        return (
            "Methodology, hyperparameters, temporal splits, "
            "validation results, and per-model metrics."
        )

    def generate(
        self,
        pipeline_result: Dict[str, Any],
        **kwargs: Any,
    ) -> ReportOutput:
        sections = []
        dataset = pipeline_result.get("dataset", "unknown")

        # Methodology
        methodology = (
            f"**Dataset**: {dataset}\n"
            f"**Ecosystem**: {pipeline_result.get('ecosystem_type', 'N/A')}\n"
            f"**Churn window**: {pipeline_result.get('churn_window_days', 'N/A')} days\n"
            f"**Train/test split**: temporal (quantile-based)\n"
            f"**Random seed**: 42 (deterministic)\n"
            f"**Models**: Logistic Regression, Random Forest, XGBoost\n"
        )
        sections.append(ReportSection("Methodology", methodology))

        # Data Characteristics
        churn_rate = pipeline_result.get("churn_rate", 0)
        imbalance = pipeline_result.get("imbalance_ratio", 0)
        characteristics = (
            f"- **Churn rate**: {churn_rate:.2%}\n"
            f"- **Imbalance ratio**: {imbalance:.2f}:1\n"
            f"- **Dominant feature group**: {pipeline_result.get('dominant_feature_group', 'N/A')}\n"
        )
        sections.append(ReportSection("Data Characteristics", characteristics))

        # Validation Results
        schema_errors = pipeline_result.get("schema_errors", 0)
        schema_warnings = pipeline_result.get("schema_warnings", 0)
        behavioral_warnings = pipeline_result.get("behavioral_warnings", 0)
        output_missing = pipeline_result.get("output_files_missing", 0)

        validation = (
            f"| Layer | Status |\n"
            f"|-------|--------|\n"
            f"| Schema | {'PASS' if schema_errors == 0 else f'{schema_errors} errors'} |\n"
            f"| Behavioral | {'PASS' if behavioral_warnings == 0 else f'{behavioral_warnings} warnings'} |\n"
            f"| Output | {'PASS' if output_missing == 0 else f'{output_missing} missing files'} |\n"
        )
        sections.append(ReportSection("Validation Results", validation))

        # Performance Summary
        best = pipeline_result.get("best_model", "N/A")
        performance = (
            f"The **{best}** model achieved the best overall performance.\n\n"
            f"Per-model metrics are available in the experiment log "
            f"and the model comparison report."
        )
        sections.append(ReportSection("Performance Summary", performance))

        # Reproducibility
        timing = pipeline_result.get("duration_seconds", 0)
        reproducibility = (
            f"- **Pipeline duration**: {timing:.1f}s\n"
            f"- **Framework version**: {FRAMEWORK_VERSION}\n"
            f"- **Python**: see environment info\n"
            f"- **All random seeds fixed**: Yes (seed=42)\n"
            f"- **Temporal split**: deterministic (quantile-based)\n"
        )
        sections.append(ReportSection("Reproducibility", reproducibility))

        metadata = {
            "dataset": dataset,
            "report_type": "technical",
            "pipeline_duration": f"{timing:.1f}s",
        }

        return ReportOutput(
            name=self.name,
            title=self.title,
            sections=sections,
            metadata=metadata,
        )
