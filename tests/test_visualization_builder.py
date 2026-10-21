
from src.reporting.visualization_builder import (
    build_tables,
    build_visualizations,
)


def test_visualizations_from_evidence():
    investigation = {
        "evidence": [
            {
                "tool_name": "get_risk_metrics",
                "result": {
                    "status": "PASS",
                    "data": {
                        "equity_exposure_pct": 0.7046
                    },
                },
            },
            {
                "tool_name": "retrieve_risk_policy",
                "result": {
                    "status": "PASS",
                    "rules": [{
                        "metric": "equity_exposure_pct",
                        "label": "Equity Exposure",
                        "warning_threshold": 0.60,
                        "breach_threshold": 0.65,
                        "unit": "pct_nav",
                    }],
                },
            },
            {
                "tool_name": "analyze_positions",
                "result": {
                    "status": "PASS",
                    "positions": [
                        {
                            "instrument_id": "QQQ",
                            "asset_class": "Equity",
                            "weight": 0.42,
                        },
                        {
                            "instrument_id": "SPY",
                            "asset_class": "Equity",
                            "weight": 0.28,
                        },
                        {
                            "instrument_id": "TLT",
                            "asset_class": "Rates",
                            "weight": 0.30,
                        },
                    ],
                },
            },
        ],
    }

    charts = build_visualizations(investigation)

    chart_ids = {
        chart.visualization_id
        for chart in charts
    }

    assert "risk_limit_comparison" in chart_ids
    assert "asset_class_allocation" in chart_ids
    assert "instrument_concentration" in chart_ids

    limit_chart = next(
        chart for chart in charts
        if chart.visualization_id == "risk_limit_comparison"
    )

    assert limit_chart.data[0]["observed"] == 70.46
    assert limit_chart.data[0]["breach"] == 65.0
    assert limit_chart.evidence_indices == [0, 1]


def test_missing_evidence_does_not_create_fake_charts():
    charts = build_visualizations({
        "evidence": []
    })

    assert charts == []


def test_data_quality_table():
    investigation = {
        "evidence": [{
            "tool_name": "check_data_quality",
            "result": {
                "overall_status": "PASS",
                "checks": [{
                    "check": "nav_reconciliation",
                    "status": "PASS",
                    "issue_count": 0,
                }],
            },
        }],
    }

    tables = build_tables(investigation)

    assert len(tables) == 1
    assert tables[0].table_id == "data_quality_checks"
    assert tables[0].rows[0]["status"] == "PASS"
