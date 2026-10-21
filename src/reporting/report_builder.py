
"""Convert completed AI investigations into structured reports."""

import re

from src.reporting.report_schema import (
    EvidenceReference,
    InvestigationReport,
    ReportFinding,
)

from src.reporting.visualization_builder import (
    build_tables,
    build_visualizations,
)

from src.reporting.visualization_selector import (
    select_visualizations,
)

EVIDENCE_CITATION_PATTERN = re.compile(
    r"([a-z][a-z0-9_]*)\s*,\s*evidence\s+(\d+)",
    re.IGNORECASE,
)


def _extract_evidence_indices(text, evidence):
    """Extract only citations matching recorded evidence."""

    matches = EVIDENCE_CITATION_PATTERN.findall(
        text or ""
    )

    indices = []

    for tool_name, index_text in matches:
        index = int(index_text)

        if index >= len(evidence):
            continue

        actual_tool = evidence[index].get(
            "tool_name"
        )

        if actual_tool != tool_name:
            continue

        if index not in indices:
            indices.append(index)

    return indices


def _extract_sections(markdown):
    """Split a Markdown report into heading-based sections."""

    sections = []
    current_title = "Introduction"
    current_lines = []

    for line in markdown.splitlines():
        if line.startswith("#"):
            if current_lines:
                sections.append(
                    (
                        current_title,
                        "\n".join(current_lines).strip(),
                    )
                )

            current_title = line.lstrip("#").strip()
            current_lines = []

        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            (
                current_title,
                "\n".join(current_lines).strip(),
            )
        )

    return sections


def _classify_section(title):
    lower = title.lower()

    if "potential driver" in lower:
        return "INTERPRETATION"

    if "hypothes" in lower:
        return "HYPOTHESIS"

    return "OBSERVATION"


def build_report(investigation):
    """Build a structured report without new model calls."""

    if not isinstance(investigation, dict):
        raise ValueError(
            "Investigation must be a dictionary"
        )

    evidence = investigation.get(
        "evidence", []
    )

    draft = investigation.get(
        "draft_report", ""
    )

    report = InvestigationReport(
        investigation_id=str(
            investigation["investigation_id"]
        ),
        portfolio_id=str(
            investigation["portfolio_id"]
        ),
        as_of_date=str(
            investigation["as_of_date"]
        ),
        exception_id=str(
            investigation["exception_id"]
        ),
        rule_id=str(
            investigation["rule_id"]
        ),
        exception_status=str(
            investigation["exception_status"]
        ),
        investigation_trace=investigation.get(
            "trace", []
        ),
        limitations=list(
            investigation.get("limitations", [])
        ),
        report_status="DRAFT",
        review_status="REQUIRES_REVIEW",
    )

    for index, item in enumerate(evidence):
        report.evidence_references.append(
            EvidenceReference(
                evidence_index=index,
                tool_name=item["tool_name"],
                description=(
                    "Full analytical result retained "
                    "in the source investigation."
                ),
                complete=True,
            )
        )

    sections = _extract_sections(draft)

    for index, (title, narrative) in enumerate(
        sections,
        start=1,
    ):
        if not narrative:
            continue

        indices = _extract_evidence_indices(
            narrative,
            evidence,
        )

        report.findings.append(
            ReportFinding(
                finding_id=f"F{index:03d}",
                title=title,
                narrative=narrative,
                classification=_classify_section(
                    title
                ),
                evidence_indices=indices,
            )
        )

    for finding in report.findings:
        if (
            "exception and observed risk"
            in finding.title.lower()
        ):
            report.executive_summary = (
                finding.narrative
            )
            break

    if not report.executive_summary:
        report.executive_summary = (
            draft[:1000]
            if draft
            else "Investigation report unavailable."
        )

    available_charts = build_visualizations(investigation)

    selection = select_visualizations(
        charts=available_charts,
        rule_id=report.rule_id,
    )

    report.visualizations = selection.visualizations

    report.tables = build_tables(
        investigation
    )

    return report
