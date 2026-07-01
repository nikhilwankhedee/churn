"""
Executive Summary report.

A concise, high-level overview suitable for non-technical stakeholders
and paper abstracts. Covers dataset characteristics, key findings,
model performance, and business implications.
"""
from typing import Any, Dict

from src.reports.base import Report, ReportOutput, ReportSection


class ExecutiveSummaryReport(Report):
    """Concise overview of pipeline results for stakeholders."""

    @property
    def name(self) -> str:
        return "executive_summary"

    @property
    def title(self) -> str:
        return "Executive Summary"

    @property
    def description(self) -> str:
        return (
            "High-level overview of dataset, churn patterns, "
            "model performance, and key findings."
        )

    def generate(
        self,
        pipeline_result: Dict[str, Any],
        **kwargs: Any,
    ) -> ReportOutput:
        sections = []

        # Dataset Overview
        dataset = pipeline_result.get("dataset", "unknown")
        ecosystem = pipeline_result.get("ecosystem_type", "unknown")
        duration = pipeline_result.get("duration_seconds", 0)
        churn_rate = pipeline_result.get("churn_rate", 0)
        imbalance = pipeline_result.get("imbalance_ratio", 0)

        overview = (
            f"The pipeline analyzed the **{dataset}** dataset "
            f"({ecosystem} ecosystem) in {duration:.1f} seconds.\n\n"
            f"- **Churn rate**: {churn_rate:.1%}\n"
            f"- **Class imbalance ratio**: {imbalance:.1f}:1\n"
            f"- **Best model**: {pipeline_result.get('best_model', 'N/A')}\n"
        )
        sections.append(ReportSection("Dataset Overview", overview))

        # Key Findings
        best = pipeline_result.get("best_model", "N/A")
        dominant = pipeline_result.get("dominant_feature_group", "unknown")
        findings = (
            f"The **{best}** model achieved the best performance on this dataset.\n\n"
            f"The most impactful feature group was **{dominant}**, suggesting "
        )
        if dominant == "inactivity":
            findings += "that recency of interaction is the primary churn driver."
        elif dominant == "monetary":
            findings += "that spending patterns are the strongest churn predictor."
        elif dominant == "cadence":
            findings += "that purchase frequency changes signal churn risk."
        elif dominant == "review":
            findings += "that customer satisfaction indicators predict churn."
        else:
            findings += f"that {dominant} features are most informative."
        sections.append(ReportSection("Key Findings", findings))

        # Data Quality
        schema_errors = pipeline_result.get("schema_errors", 0)
        schema_warnings = pipeline_result.get("schema_warnings", 0)
        quality = (
            f"Data quality assessment: "
            f"**{schema_errors}** errors, **{schema_warnings}** warnings.\n\n"
        )
        if schema_errors == 0 and schema_warnings == 0:
            quality += "The dataset passed all quality checks with no issues."
        elif schema_errors == 0:
            quality += "No critical issues detected; minor warnings present."
        else:
            quality += "Some data quality issues were detected — review recommended."
        sections.append(ReportSection("Data Quality", quality))

        # Business Implications
        implications = (
            f"With a churn rate of {churn_rate:.1%}, "
        )
        if churn_rate > 0.5:
            implications += (
                "the majority of customers are churning. "
                "Retention interventions should target the full customer base."
            )
        elif churn_rate > 0.2:
            implications += (
                "a significant portion of customers are at risk. "
                "Targeted retention campaigns are recommended."
            )
        else:
            implications += (
                "churn is relatively low. Focus retention efforts "
                "on the highest-risk segments identified by the model."
            )
        sections.append(ReportSection("Business Implications", implications))

        metadata = {
            "dataset": dataset,
            "ecosystem_type": ecosystem,
            "churn_rate": f"{churn_rate:.2%}",
            "best_model": best,
            "dominant_feature_group": dominant,
        }

        return ReportOutput(
            name=self.name,
            title=self.title,
            sections=sections,
            metadata=metadata,
        )
