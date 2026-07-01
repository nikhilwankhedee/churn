"""
Data Quality Report.

Comprehensive data quality assessment covering completeness,
consistency, temporal validity, and outlier analysis.
"""
from typing import Any, Dict

from src.reports.base import Report, ReportOutput, ReportSection


class DataQualityReport(Report):
    """Comprehensive data quality assessment."""

    @property
    def name(self) -> str:
        return "data_quality_report"

    @property
    def title(self) -> str:
        return "Data Quality Report"

    @property
    def description(self) -> str:
        return (
            "Completeness, consistency, temporal validity, "
            "and outlier analysis."
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
            f"This report assesses data quality across four dimensions: "
            f"completeness, consistency, temporal validity, and outlier presence."
        )
        sections.append(ReportSection("Overview", overview))

        # Schema Quality
        schema_errors = pipeline_result.get("schema_errors", 0)
        schema_warnings = pipeline_result.get("schema_warnings", 0)

        if schema_errors == 0 and schema_warnings == 0:
            schema_status = "All schema checks passed. No missing required columns or structural issues."
        elif schema_errors == 0:
            schema_status = (
                f"Schema validation passed with {schema_warnings} warning(s). "
                f"Minor structural issues detected."
            )
        else:
            schema_status = (
                f"Schema validation found {schema_errors} error(s) and "
                f"{schema_warnings} warning(s). Review recommended."
            )
        sections.append(ReportSection("Schema Quality", schema_status))

        # Behavioral Quality
        behavioral_warnings = pipeline_result.get("behavioral_warnings", 0)
        if behavioral_warnings == 0:
            behavioral_status = (
                "Behavioral statistics are within expected ranges. "
                "No anomalies detected in temporal patterns or distributions."
            )
        else:
            behavioral_status = (
                f"{behavioral_warnings} behavioral warning(s) detected. "
                f"Some statistical properties deviate from expectations."
            )
        sections.append(ReportSection("Behavioral Quality", behavioral_status))

        # Output Quality
        output_missing = pipeline_result.get("output_files_missing", 0)
        if output_missing == 0:
            output_status = "All expected output files were generated successfully."
        else:
            output_status = (
                f"{output_missing} expected output file(s) missing. "
                f"Some pipeline steps may have failed silently."
            )
        sections.append(ReportSection("Output Completeness", output_status))

        # Recommendations
        recommendations = []
        if schema_errors > 0:
            recommendations.append("- Fix schema errors before running analysis")
        if behavioral_warnings > 0:
            recommendations.append("- Review behavioral warnings for data drift")
        if output_missing > 0:
            recommendations.append("- Check pipeline logs for failed steps")
        if not recommendations:
            recommendations.append("- No action required — data quality is good")

        rec_text = "\n".join(recommendations)
        sections.append(ReportSection("Recommendations", rec_text))

        metadata = {
            "dataset": dataset,
            "schema_errors": schema_errors,
            "schema_warnings": schema_warnings,
            "behavioral_warnings": behavioral_warnings,
            "output_files_missing": output_missing,
        }

        return ReportOutput(
            name=self.name,
            title=self.title,
            sections=sections,
            metadata=metadata,
        )
