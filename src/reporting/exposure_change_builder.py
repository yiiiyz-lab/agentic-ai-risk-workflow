
"""Evidence-grounded exposure change and attribution visualizations."""

import math

from src.reporting.report_schema import (
    ReportTable,
    ReportVisualization,
)


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    if not math.isfinite(value):
        return None

    return float(value)


def _percentage_points(value):
    return round(value * 100, 6)


def build_exposure_change(investigation):
    """Build daily weight-change charts and reconciliation tables.

    All weight effects are assumed to be decimal NAV fractions,
    consistent with analyze_positions methodology.
    """

    charts = []
    tables = []

    evidence = investigation.get("evidence", [])

    if not isinstance(evidence, list):
        return charts, tables

    for evidence_index, entry in enumerate(evidence):
        if not isinstance(entry, dict):
            continue

        if entry.get("tool_name") != "analyze_positions":
            continue

        result = entry.get("result", {})

        if not isinstance(result, dict):
            continue

        if result.get("status") in (
            "ERROR", "FAILED", "NOT_AVAILABLE"
        ):
            continue

        analysis = result.get("analysis", {})

        if not isinstance(analysis, dict):
            continue

        bridge = analysis.get("daily_weight_bridge", {})

        if not isinstance(bridge, dict) or not bridge:
            continue

        chart_rows = []
        table_rows = []

        for instrument, values in bridge.items():
            if not isinstance(values, dict):
                continue

            required_fields = (
                "previous_weight",
                "pre_trade_weight",
                "current_weight",
                "market_effect",
                "trade_effect",
                "total_change",
            )

            parsed = {
                field: _finite_number(values.get(field))
                for field in required_fields
            }

            if any(
                value is None
                for value in parsed.values()
            ):
                continue

            previous = parsed["previous_weight"]
            pre_trade = parsed["pre_trade_weight"]
            current = parsed["current_weight"]
            market = parsed["market_effect"]
            trade = parsed["trade_effect"]
            change = parsed["total_change"]

            # Verify the bridge before displaying attribution.
            tolerance = 1e-8

            if not math.isclose(
                previous + market,
                pre_trade,
                rel_tol=0,
                abs_tol=tolerance,
            ):
                continue

            if not math.isclose(
                pre_trade + trade,
                current,
                rel_tol=0,
                abs_tol=tolerance,
            ):
                continue

            if not math.isclose(
                market + trade,
                change,
                rel_tol=0,
                abs_tol=tolerance,
            ):
                continue

            chart_rows.append({
                "instrument": str(instrument),
                "market_effect": _percentage_points(market),
                "trade_effect": _percentage_points(trade),
                "total_change": _percentage_points(change),
            })

            table_rows.append({
                "instrument": str(instrument),
                "previous_weight_pct": _percentage_points(
                    previous
                ),
                "pre_trade_weight_pct": _percentage_points(
                    pre_trade
                ),
                "current_weight_pct": _percentage_points(
                    current
                ),
                "market_effect_pp": _percentage_points(
                    market
                ),
                "trade_effect_pp": _percentage_points(
                    trade
                ),
                "total_change_pp": _percentage_points(
                    change
                ),
            })

        if chart_rows:
            chart_rows.sort(
                key=lambda row: -abs(row["total_change"])
            )

            charts.append(
                ReportVisualization(
                    visualization_id="daily_exposure_attribution",
                    title="Daily Exposure Change: Market vs. Trading",
                    chart_type="bar",
                    data=chart_rows,
                    x_field="instrument",
                    y_fields=[
                        "market_effect",
                        "trade_effect",
                    ],
                    unit="percentage_points",
                    evidence_indices=[evidence_index],
                    notes=(
                        "One-day change in instrument NAV weights. "
                        "Market and trading effects are calculated "
                        "by the portfolio analytics engine. "
                        "Each displayed instrument bridge passed "
                        "arithmetic reconciliation."
                    ),
                )
            )

            tables.append(
                ReportTable(
                    table_id="daily_weight_bridge",
                    title="Daily Portfolio Weight Reconciliation",
                    columns=[
                        "instrument",
                        "previous_weight_pct",
                        "pre_trade_weight_pct",
                        "current_weight_pct",
                        "market_effect_pp",
                        "trade_effect_pp",
                        "total_change_pp",
                    ],
                    rows=table_rows,
                    evidence_indices=[evidence_index],
                )
            )

        # Use the first valid analyze_positions result.
        break

    return charts, tables


def build_historical_exposure_change(investigation):
    """Compare instrument weight changes across historical windows.

    Source values are decimal NAV-weight changes.
    Output values are percentage points, not percentage returns.
    """

    charts = []
    evidence = investigation.get("evidence", [])

    if not isinstance(evidence, list):
        return charts

    for evidence_index, entry in enumerate(evidence):
        if not isinstance(entry, dict):
            continue

        if entry.get("tool_name") != "analyze_positions":
            continue

        result = entry.get("result", {})

        if not isinstance(result, dict):
            continue

        if result.get("status") in (
            "ERROR", "FAILED", "NOT_AVAILABLE"
        ):
            continue

        analysis = result.get("analysis", {})

        if not isinstance(analysis, dict):
            continue

        history = analysis.get("historical_weight_changes", {})

        if not isinstance(history, dict):
            continue

        five_day = history.get("5d", {})
        twenty_day = history.get("20d", {})

        if not isinstance(five_day, dict):
            five_day = {}

        if not isinstance(twenty_day, dict):
            twenty_day = {}

        instruments = sorted(
            set(five_day) & set(twenty_day)
        )

        rows = []

        for instrument in instruments:
            change_5d = _finite_number(
                five_day[instrument]
            )
            change_20d = _finite_number(
                twenty_day[instrument]
            )

            if change_5d is None or change_20d is None:
                continue

            rows.append({
                "instrument": str(instrument),
                "change_5d": _percentage_points(
                    change_5d
                ),
                "change_20d": _percentage_points(
                    change_20d
                ),
            })

        rows.sort(
            key=lambda row: -abs(row["change_20d"])
        )

        if rows:
            charts.append(
                ReportVisualization(
                    visualization_id="historical_exposure_change",
                    title="Historical Portfolio Weight Changes",
                    chart_type="bar",
                    data=rows,
                    x_field="instrument",
                    y_fields=[
                        "change_5d",
                        "change_20d",
                    ],
                    unit="percentage_points",
                    evidence_indices=[evidence_index],
                    notes=(
                        "Changes in instrument weights over "
                        "5 and 20 trading days, expressed in "
                        "percentage points. These are endpoint "
                        "changes, not a daily time series or "
                        "standalone instrument returns."
                    ),
                )
            )

        break

    return charts
