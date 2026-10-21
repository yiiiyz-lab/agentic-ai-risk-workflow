
"""Structured schema for evidence-grounded AI risk reports."""

from dataclasses import asdict, dataclass, field
from typing import Any


REPORT_SCHEMA_VERSION = "1.0"


@dataclass
class EvidenceReference:
    evidence_index: int
    tool_name: str
    description: str = ""
    complete: bool = True


@dataclass
class ReportFinding:
    finding_id: str
    title: str
    narrative: str
    classification: str
    evidence_indices: list[int] = field(default_factory=list)


@dataclass
class ReportVisualization:
    visualization_id: str
    title: str
    chart_type: str
    data: list[dict[str, Any]]
    x_field: str = ""
    y_fields: list[str] = field(default_factory=list)
    unit: str = ""
    evidence_indices: list[int] = field(default_factory=list)
    notes: str = ""


@dataclass
class ReportTable:
    table_id: str
    title: str
    columns: list[str]
    rows: list[dict[str, Any]]
    evidence_indices: list[int] = field(default_factory=list)


@dataclass
class InvestigationReport:
    investigation_id: str
    portfolio_id: str
    as_of_date: str
    exception_id: str
    rule_id: str
    exception_status: str

    executive_summary: str = ""
    findings: list[ReportFinding] = field(default_factory=list)
    evidence_references: list[EvidenceReference] = field(
        default_factory=list
    )
    visualizations: list[ReportVisualization] = field(
        default_factory=list
    )
    tables: list[ReportTable] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    investigation_trace: list[dict[str, Any]] = field(
        default_factory=list
    )

    report_status: str = "DRAFT"
    review_status: str = "REQUIRES_REVIEW"
    schema_version: str = REPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
