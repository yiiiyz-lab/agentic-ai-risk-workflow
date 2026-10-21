
from src.reporting.report_schema import (
    EvidenceReference,
    InvestigationReport,
    ReportFinding,
    ReportVisualization,
)


def test_report_schema_serialization():
    report = InvestigationReport(
        investigation_id="test-investigation",
        portfolio_id="PORT_001",
        as_of_date="2026-09-30",
        exception_id="2026-09-30_EQUITY_EXPOSURE_LIMIT",
        rule_id="EQUITY_EXPOSURE_LIMIT",
        exception_status="BREACH",
        executive_summary="Equity exposure exceeds its limit.",
        findings=[
            ReportFinding(
                finding_id="F001",
                title="Equity exposure breach",
                narrative="Observed exposure is above the limit.",
                classification="OBSERVATION",
                evidence_indices=[0, 2],
            )
        ],
        evidence_references=[
            EvidenceReference(
                evidence_index=0,
                tool_name="get_risk_metrics",
            ),
            EvidenceReference(
                evidence_index=2,
                tool_name="retrieve_risk_policy",
            ),
        ],
        visualizations=[
            ReportVisualization(
                visualization_id="exposure_limit",
                title="Exposure vs. limit",
                chart_type="bar",
                data=[
                    {
                        "metric": "Equity exposure",
                        "observed": 70.46,
                        "limit": 65.0,
                    }
                ],
                x_field="metric",
                y_fields=["observed", "limit"],
                unit="percent_nav",
                evidence_indices=[0, 2],
            )
        ],
    )

    result = report.to_dict()

    assert result["schema_version"] == "1.0"
    assert result["review_status"] == "REQUIRES_REVIEW"
    assert result["findings"][0]["classification"] == "OBSERVATION"
    assert result["visualizations"][0]["data"][0]["limit"] == 65.0
    assert result["visualizations"][0]["evidence_indices"] == [0, 2]
