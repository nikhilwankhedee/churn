"""
Report registry.

Provides a centralized registry for report generators, building on
the core PluginRegistry. Includes convenience functions for registering
built-in reports and retrieving them by name.

Usage:
    from src.reports.registry import get_report, list_reports

    report = get_report("executive_summary")
    output = report.generate(pipeline_result)
    output.save("report.md")
"""
from typing import Any, Dict, List, Optional

from src.reports.base import Report, ReportOutput
from src.core.registry import registry


_CATEGORY = "reports"


def register_report(
    name: str,
    dotted_path: str,
    metadata: Optional[dict] = None,
) -> None:
    """Register a report generator by lazy dotted path."""
    registry.register(name, _CATEGORY, dotted_path, metadata)


def register_report_class(
    name: str,
    cls: type,
    metadata: Optional[dict] = None,
) -> None:
    """Register an already-imported report class."""
    registry.register_class(name, _CATEGORY, cls, metadata)


def get_report(name: str) -> Report:
    """Retrieve a registered report generator instance."""
    return registry.get_instance(name, _CATEGORY)


def list_reports() -> List[str]:
    """Return all registered report names."""
    return registry.list_registered(_CATEGORY)


def generate_report(
    name: str,
    pipeline_result: Dict[str, Any],
    **kwargs,
) -> ReportOutput:
    """Generate a single report by name."""
    report = get_report(name)
    return report.generate(pipeline_result, **kwargs)


def generate_all_reports(
    pipeline_result: Dict[str, Any],
    report_names: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, ReportOutput]:
    """Generate multiple reports.

    Parameters
    ----------
    pipeline_result : dict
        The metadata dict returned by run_pipeline().
    report_names : list of str, optional
        Specific reports to generate. Default: all registered.
    **kwargs : Any
        Additional context passed to each report.

    Returns
    -------
    Dict mapping report name -> ReportOutput.
    """
    names = report_names or list_reports()
    outputs = {}
    for name in names:
        try:
            outputs[name] = generate_report(name, pipeline_result, **kwargs)
        except Exception:
            continue
    return outputs


def _register_builtins() -> None:
    """Register the built-in report generators (lazy-loaded)."""
    _base = "src.reports.builtins"
    builtins = {
        "executive_summary": f"{_base}.executive_summary.ExecutiveSummaryReport",
        "technical_report": f"{_base}.technical_report.TechnicalReport",
        "data_quality_report": f"{_base}.data_quality_report.DataQualityReport",
        "model_comparison": f"{_base}.model_comparison.ModelComparisonReport",
        "calibration_report": f"{_base}.calibration_report.CalibrationReport",
        "explainability_report": f"{_base}.explainability_report.ExplainabilityReport",
        "experiment_report": f"{_base}.experiment_report.ExperimentReport",
    }
    for name, path in builtins.items():
        if not registry.is_registered(name, _CATEGORY):
            registry.register(
                name, _CATEGORY, path,
                metadata={"builtin": True},
            )


# Auto-register builtins on import
_register_builtins()
