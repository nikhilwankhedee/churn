"""
Abstract base class for report generation.

All reports implement a common interface: given pipeline context data,
produce structured content (initially Markdown) suitable for researchers.

Reports are designed to be:
- Composable: reports can include sections from other reports
- Exportable: Markdown now, HTML/PDF later via exporter plugins
- Narrative: every report includes concise natural-language summaries
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReportSection:
    """A single section within a report.

    Attributes
    ----------
    title : str
        Section heading.
    content : str
        Markdown content for this section.
    level : int
        Heading level (1 = top-level, 2 = subsection, etc.).
    """
    title: str
    content: str
    level: int = 2


@dataclass
class ReportOutput:
    """Container for a generated report.

    Attributes
    ----------
    name : str
        Report identifier.
    title : str
        Human-readable report title.
    sections : list of ReportSection
        Ordered sections composing the report.
    metadata : dict
        Report-specific metadata (dataset, timestamp, etc.).
    format : str
        Output format ('markdown' initially).
    """
    name: str
    title: str
    sections: List[ReportSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    format: str = "markdown"

    def to_markdown(self) -> str:
        """Render the full report as a Markdown string."""
        lines = [f"# {self.title}", ""]
        for section in self.sections:
            prefix = "#" * min(section.level + 1, 6)
            lines.append(f"{prefix} {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")
        if self.metadata:
            lines.append("---")
            lines.append("")
            lines.append("## Report Metadata")
            lines.append("")
            for k, v in self.metadata.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save the report to a file."""
        content = self.to_markdown()
        with open(path, "w") as f:
            f.write(content)


class Report(ABC):
    """Abstract base for report generators.

    Subclasses must implement:
        - name: report identifier
        - title: human-readable title
        - generate(pipeline_result, **kwargs) -> ReportOutput
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this report (e.g. 'executive_summary')."""
        ...

    @property
    @abstractmethod
    def title(self) -> str:
        """Human-readable report title."""
        ...

    @property
    def description(self) -> str:
        """Brief description of what this report covers."""
        return self.__class__.__doc__ or ""

    @abstractmethod
    def generate(
        self,
        pipeline_result: Dict[str, Any],
        **kwargs: Any,
    ) -> ReportOutput:
        """Generate the report from pipeline results.

        Parameters
        ----------
        pipeline_result : dict
            The metadata dict returned by run_pipeline().
        **kwargs : Any
            Additional context (e.g., experiment_log, comparison_data).

        Returns
        -------
        ReportOutput with sections and metadata.
        """
        ...

    def __repr__(self) -> str:
        return f"<Report: {self.name}>"
