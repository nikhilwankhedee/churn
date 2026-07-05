"""Tests for reports, batch, and explorer modules."""
import os
import sys
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestReports:
    """Tests for report generation."""

    def test_list_reports(self):
        from src.reports import list_reports
        reports = list_reports()
        assert isinstance(reports, list)
        assert len(reports) >= 7

    def test_get_report(self):
        from src.reports import get_report
        for name in ["executive_summary", "technical_report", "data_quality_report"]:
            r = get_report(name)
            assert hasattr(r, "description")

    def test_report_base_interface(self):
        from src.reports import get_report
        r = get_report("executive_summary")
        assert hasattr(r, "generate")

    def test_report_section_dataclass(self):
        from src.reports.base import ReportSection
        section = ReportSection(title="Test", content="Hello")
        assert section.title == "Test"
        assert section.content == "Hello"

    def test_report_output_dataclass(self):
        from src.reports.base import ReportOutput
        output = ReportOutput(
            name="test_report",
            title="Test Report",
            sections=[],
            metadata={"key": "value"},
        )
        md = output.to_markdown()
        assert "Test Report" in md


class TestBatch:
    """Tests for batch execution."""

    def test_batch_result_dataclass(self):
        from src.batch import BatchResult
        result = BatchResult()
        assert result.successful == []
        assert result.failed == []
        assert result.total_duration == 0.0

    def test_format_benchmark_table_empty(self):
        from src.batch import format_benchmark_table
        table = format_benchmark_table({})
        assert isinstance(table, str)


class TestExplorer:
    """Tests for experiment explorer."""

    def test_list_experiments(self):
        from src.explorer import list_experiments
        experiments = list_experiments()
        assert isinstance(experiments, list)

    def test_list_experiments_with_limit(self):
        from src.explorer import list_experiments
        experiments = list_experiments(limit=5)
        assert len(experiments) <= 5

    def test_compare_experiments_no_data(self):
        from src.explorer import compare_experiments
        result = compare_experiments(["nonexistent1", "nonexistent2"])
        # May return None if no experiments exist
        assert result is None or hasattr(result, "empty")


class TestPlugins:
    """Tests for plugin discovery."""

    def test_discover_plugins(self):
        from src.plugins import discover_plugins
        plugins = discover_plugins()
        assert isinstance(plugins, dict)
