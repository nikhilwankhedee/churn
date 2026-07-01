"""
Calibration Report.

Analysis of model probability calibration — how well predicted
probabilities match actual outcomes. Includes ECE scores and
calibration curve interpretation.
"""
from typing import Any, Dict

from src.reports.base import Report, ReportOutput, ReportSection


class CalibrationReport(Report):
    """Model probability calibration analysis."""

    @property
    def name(self) -> str:
        return "calibration_report"

    @property
    def title(self) -> str:
        return "Calibration Report"

    @property
    def description(self) -> str:
        return (
            "Probability calibration analysis, ECE scores, "
            "and calibration curve interpretation."
        )

    def generate(
        self,
        pipeline_result: Dict[str, Any],
        **kwargs: Any,
    ) -> ReportOutput:
        sections = []
        dataset = pipeline_result.get("dataset", "unknown")

        # Overview
        overview = (
            f"Dataset: **{dataset}**\n\n"
            f"Calibration measures how well predicted probabilities "
            f"reflect actual churn rates. A perfectly calibrated model "
            f"has predicted probabilities equal to observed frequencies."
        )
        sections.append(ReportSection("Overview", overview))

        # Interpretation guide
        interpretation = (
            "## Interpreting Calibration\n\n"
            "- **ECE < 0.05**: Excellent calibration\n"
            "- **ECE 0.05-0.10**: Good calibration\n"
            "- **ECE 0.10-0.15**: Fair calibration\n"
            "- **ECE > 0.15**: Poor calibration — consider Platt scaling or isotonic regression\n\n"
            "Calibration curves are saved in the figures directory."
        )
        sections.append(ReportSection("Interpretation Guide", interpretation))

        # Methodology
        methodology = (
            "**Method**: Expected Calibration Error (ECE) with 10 uniform bins\n"
            "**Bootstrap**: 200 iterations for confidence intervals\n"
            "**Curves**: Reliability diagrams saved per model"
        )
        sections.append(ReportSection("Methodology", methodology))

        # Per-model calibration
        best = pipeline_result.get("best_model", "N/A")
        per_model = (
            f"Per-model calibration error values are recorded in the "
            f"experiment log. The **{best}** model is the primary candidate "
            f"for calibration analysis.\n\n"
            f"Calibration curves are saved in:\n"
            f"- `figures/calibration/`\n"
            f"- `results/model_metrics/`"
        )
        sections.append(ReportSection("Per-Model Calibration", per_model))

        metadata = {
            "dataset": dataset,
            "method": "ECE (10 bins, uniform)",
            "best_model": best,
        }

        return ReportOutput(
            name=self.name,
            title=self.title,
            sections=sections,
            metadata=metadata,
        )
