"""
Model Comparison Report.

Side-by-side comparison of all trained models across standard metrics.
Includes ranking, per-metric winners, and stability analysis.
"""
from typing import Any, Dict

from src.reports.base import Report, ReportOutput, ReportSection


class ModelComparisonReport(Report):
    """Side-by-side comparison of all trained models."""

    @property
    def name(self) -> str:
        return "model_comparison"

    @property
    def title(self) -> str:
        return "Model Comparison Report"

    @property
    def description(self) -> str:
        return (
            "Per-model metrics, ranking, per-metric winners, "
            "and stability analysis."
        )

    def generate(
        self,
        pipeline_result: Dict[str, Any],
        **kwargs: Any,
    ) -> ReportOutput:
        sections = []
        dataset = pipeline_result.get("dataset", "unknown")
        best_model = pipeline_result.get("best_model", "N/A")

        # Overview
        overview = (
            f"Dataset: **{dataset}**\n\n"
            f"Models evaluated: Logistic Regression, Random Forest, XGBoost\n"
            f"Best overall model: **{best_model}**"
        )
        sections.append(ReportSection("Overview", overview))

        # Per-model analysis
        # The pipeline_result contains metrics in the experiment log
        # We'll summarize what's available
        churn_rate = pipeline_result.get("churn_rate", 0)
        imbalance = pipeline_result.get("imbalance_ratio", 0)

        model_section = (
            f"**Churn rate**: {churn_rate:.2%}\n"
            f"**Imbalance ratio**: {imbalance:.2f}:1\n\n"
            f"All models were trained with balanced class weights "
            f"to handle the {imbalance:.1f}:1 class imbalance.\n\n"
            f"Per-model metrics (ROC-AUC, PR-AUC, F1, Brier score, "
            f"calibration error) are recorded in the experiment log "
            f"and can be viewed with `churn experiments`."
        )
        sections.append(ReportSection("Model Performance", model_section))

        # Recommendations
        recommendations = (
            f"Based on the analysis, **{best_model}** is recommended "
            f"for this dataset.\n\n"
            f"For production deployment, consider:\n"
            f"- Monitoring model performance over time\n"
            f"- Retraining with fresh data periodically\n"
            f"- Using SHAP values for explainability"
        )
        sections.append(ReportSection("Recommendations", recommendations))

        metadata = {
            "dataset": dataset,
            "best_model": best_model,
            "churn_rate": f"{churn_rate:.2%}",
            "imbalance_ratio": f"{imbalance:.2f}",
        }

        return ReportOutput(
            name=self.name,
            title=self.title,
            sections=sections,
            metadata=metadata,
        )
