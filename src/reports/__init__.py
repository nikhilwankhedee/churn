"""
Report generation framework.

This package provides a pluggable report system with:
- Abstract base class (Report)
- Built-in reports (7 types)
- Plugin registry for custom reports

Usage:
    from src.reports import get_report, list_reports, generate_all_reports

    # Generate a single report
    report = get_report("executive_summary")
    output = report.generate(pipeline_result)
    output.save("report.md")

    # Generate all reports
    reports = generate_all_reports(pipeline_result)
    for name, output in reports.items():
        output.save(f"reports/{name}.md")
"""
from src.reports.base import Report, ReportOutput, ReportSection
from src.reports.registry import (
    get_report,
    list_reports,
    register_report,
    register_report_class,
    generate_report,
    generate_all_reports,
)

__all__ = [
    "Report",
    "ReportOutput",
    "ReportSection",
    "get_report",
    "list_reports",
    "register_report",
    "register_report_class",
    "generate_report",
    "generate_all_reports",
]
