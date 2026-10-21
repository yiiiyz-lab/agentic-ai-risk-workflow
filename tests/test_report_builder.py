
from src.reporting.report_builder import (
    build_report,
)


def test_build_report_from_investigation():
    investigation = {
        "investigation_id": "test-001",
        "portfolio_id": "PORT_001",
        "as_of_date": "2026-09-30",
        "exception_id": (
            "2026-09-30_EQUITY_EXPOSURE_LIMIT"
        ),
        "rule_id": "EQUITY_EXPOSURE_LIMIT",
        "exception_status": "BREACH",
        "status": "REQUIRES_REVIEW",
        "trace": [],
        "evidence": [
            {
                "tool_name": "get_risk_metrics",
                "result": {
                    "equity_exposure_pct": 0.7046
                },
            },
            {
                "tool_name": "retrieve_risk_policy",
                "result": {
                    "breach_threshold": 0.65
                },
            },
        ],
        "limitations": [],
        "draft_report": (
            "### Exception and observed risk\n"
            "Equity exposure exceeded its limit. "
            "[get_risk_metrics, evidence 0; "
            "retrieve_risk_policy, evidence 1]\n\n"
            "### Potential drivers\n"
            "The portfolio has a high equity allocation."
        ),
    }

    report = build_report(
        investigation
    )

    assert report.portfolio_id == "PORT_001"
    assert report.report_status == "DRAFT"
    assert report.review_status == "REQUIRES_REVIEW"

    assert len(report.evidence_references) == 2
    assert len(report.findings) == 2

    assert (
        report.findings[0].evidence_indices
        == [0, 1]
    )

    assert report.executive_summary.startswith(
        "Equity exposure exceeded"
    )
