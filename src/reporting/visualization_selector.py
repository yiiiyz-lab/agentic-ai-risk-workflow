
"""Select and prioritize visualizations for each risk exception."""

from dataclasses import dataclass, field

from src.reporting.report_schema import ReportVisualization


@dataclass
class VisualizationSelection:
    visualizations: list[ReportVisualization] = field(
        default_factory=list
    )
    unavailable: list[str] = field(default_factory=list)


# These are preferences, not mandatory charts.
# A chart is selected only when actual evidence produced it.
CHART_PRIORITIES = {
    "equity_exposure": [
        "risk_limit_comparison",
        "asset_class_allocation",
        "historical_exposure_change",
        "daily_exposure_attribution",
        "instrument_concentration",
    ],
    "concentration": [
        "instrument_concentration",
        "risk_limit_comparison",
        "historical_exposure_change",
        "daily_exposure_attribution",
        "asset_class_allocation",
    ],
    "var_es": [
        "risk_limit_comparison",
        "var_es_summary",
        "tail_risk_attribution",
        "historical_risk_trend",
        "instrument_concentration",
    ],
    "drawdown": [
        "risk_limit_comparison",
        "historical_drawdown",
        "pnl_attribution",
        "instrument_concentration",
    ],
    "volatility": [
        "risk_limit_comparison",
        "rolling_volatility",
        "risk_factor_comparison",
        "asset_class_allocation",
    ],
    "generic": [
        "risk_limit_comparison",
        "asset_class_allocation",
        "instrument_concentration",
    ],
}


def classify_exception(rule_id: str, metric: str = "") -> str:
    """Determine exception category from rule and metric identifiers."""

    if not isinstance(rule_id, str):
        rule_id = ""

    if not isinstance(metric, str):
        metric = ""

    identifier = f"{rule_id} {metric}".lower()

    if any(term in identifier for term in (
        "var_99", "es_99", "expected_shortfall",
        "value_at_risk",
    )):
        return "var_es"

    if "drawdown" in identifier:
        return "drawdown"

    if "volatility" in identifier or "vol_spike" in identifier:
        return "volatility"

    if any(term in identifier for term in (
        "concentration",
        "single_instrument",
        "largest_position",
    )):
        return "concentration"

    if any(term in identifier for term in (
        "equity_exposure",
        "equity_limit",
    )):
        return "equity_exposure"

    return "generic"


def select_visualizations(
    charts: list[ReportVisualization],
    rule_id: str,
    metric: str = "",
    max_charts: int = 4,
) -> VisualizationSelection:
    """Rank available charts for an exception without fabricating data."""

    if max_charts < 1:
        raise ValueError("max_charts must be positive")

    category = classify_exception(rule_id, metric)
    priorities = CHART_PRIORITIES[category]

    available = {
        chart.visualization_id: chart
        for chart in charts
    }

    selected = []
    selected_ids = set()

    for chart_id in priorities:
        if chart_id in available and len(selected) < max_charts:
            selected.append(available[chart_id])
            selected_ids.add(chart_id)

    # Include additional evidence-backed charts only if space remains.
    for chart in charts:
        if len(selected) >= max_charts:
            break

        if chart.visualization_id not in selected_ids:
            selected.append(chart)
            selected_ids.add(chart.visualization_id)

    unavailable = [
        chart_id
        for chart_id in priorities
        if chart_id not in available
    ]

    return VisualizationSelection(
        visualizations=selected,
        unavailable=unavailable,
    )
