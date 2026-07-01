"""
Explainability Report.

SHAP-based feature importance analysis — which features drive
churn predictions most strongly. Includes global and local explanations.
"""
from typing import Any, Dict

from src.reports.base import Report, ReportOutput, ReportSection


class ExplainabilityReport(Report):
    """SHAP-based feature importance and explanation analysis."""

    @property
    def name(self) -> str:
        return "explainability_report"

    @property
    def title(self) -> str:
        return "Explainability Report"

    @property
    def description(self) -> str:
        return (
            "SHAP-based feature importance, global explanations, "
            "and local interpretation patterns."
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
            f"This report summarizes SHAP-based explainability analysis "
            f"for the trained models. SHAP (SHapley Additive exPlanations) "
            f"provides both global feature importance and local predictions."
        )
        sections.append(ReportSection("Overview", overview))

        # Global Explanations
        dominant = pipeline_result.get("dominant_feature_group", "unknown")
        global_section = (
            f"The dominant feature group from ablation analysis was "
            f"**{dominant}**.\n\n"
            f"SHAP values provide a more granular view of feature importance "
            f"at the individual prediction level.\n\n"
            f"**Global feature importance** is computed as the mean |SHAP value| "
            f"across all test samples."
        )
        sections.append(ReportSection("Global Feature Importance", global_section))

        # Local Explanations
        local_section = (
            "**Local explanations** show which features drove each individual "
            "prediction. This is useful for:\n"
            "- Understanding why specific customers are flagged as high-risk\n"
            "- Identifying common churn patterns\n"
            "- Building trust with stakeholders through interpretable predictions"
        )
        sections.append(ReportSection("Local Explanations", local_section))

        # Output locations
        outputs = (
            "SHAP analysis outputs:\n"
            "- `figures/shap_analysis/` — SHAP summary plots\n"
            "- `results/shap_values/` — raw SHAP value matrices\n"
            "- `results/model_metrics/` — per-model feature importance rankings"
        )
        sections.append(ReportSection("Output Locations", outputs))

        metadata = {
            "dataset": dataset,
            "dominant_feature_group": dominant,
            "method": "SHAP (TreeExplainer for tree models, KernelExplainer for LR)",
        }

        return ReportOutput(
            name=self.name,
            title=self.title,
            sections=sections,
            metadata=metadata,
        )
