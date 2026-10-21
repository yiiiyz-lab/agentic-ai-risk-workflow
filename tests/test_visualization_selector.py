
from src.reporting.report_schema import ReportVisualization
from src.reporting.visualization_selector import (
    classify_exception,
    select_visualizations,
)


def make_chart(chart_id):
    return ReportVisualization(
        visualization_id=chart_id,
        title=chart_id,
        chart_type="bar",
        data=[{"x": "sample", "y": 1.0}],
        x_field="x",
        y_fields=["y"],
        evidence_indices=[0],
    )


def test_exception_classification():
    assert classify_exception(
        "EQUITY_EXPOSURE_LIMIT"
    ) == "equity_exposure"

    assert classify_exception(
        "SINGLE_INSTRUMENT_LIMIT",
        "largest_position_weight",
    ) == "concentration"

    assert classify_exception(
        "VAR_99_LIMIT"
    ) == "var_es"

    assert classify_exception(
        "ES_99_LIMIT"
    ) == "var_es"

    assert classify_exception(
        "DRAWDOWN_LIMIT"
    ) == "drawdown"


def test_equity_exposure_prioritization():
    charts = [
        make_chart("instrument_concentration"),
        make_chart("asset_class_allocation"),
        make_chart("risk_limit_comparison"),
    ]

    result = select_visualizations(
        charts,
        rule_id="EQUITY_EXPOSURE_LIMIT",
    )

    assert [
        chart.visualization_id
        for chart in result.visualizations
    ] == [
        "risk_limit_comparison",
        "asset_class_allocation",
        "instrument_concentration",
    ]


def test_concentration_prioritization():
    charts = [
        make_chart("asset_class_allocation"),
        make_chart("risk_limit_comparison"),
        make_chart("instrument_concentration"),
    ]

    result = select_visualizations(
        charts,
        rule_id="SINGLE_INSTRUMENT_LIMIT",
    )

    assert result.visualizations[0].visualization_id == (
        "instrument_concentration"
    )


def test_var_attribution_is_not_invented():
    charts = [
        make_chart("risk_limit_comparison"),
        make_chart("instrument_concentration"),
    ]

    result = select_visualizations(
        charts,
        rule_id="VAR_99_LIMIT",
    )

    ids = [
        chart.visualization_id
        for chart in result.visualizations
    ]

    assert "tail_risk_attribution" not in ids
    assert "tail_risk_attribution" in result.unavailable


def test_chart_limit():
    charts = [
        make_chart(f"chart_{index}")
        for index in range(10)
    ]

    result = select_visualizations(
        charts,
        rule_id="OTHER_RULE",
        max_charts=3,
    )

    assert len(result.visualizations) == 3
