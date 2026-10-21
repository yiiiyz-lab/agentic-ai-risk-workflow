
"""Build evidence-grounded charts and tables for investigation reports."""

import math

from src.reporting.report_schema import (
    ReportTable,
    ReportVisualization,
)

from src.reporting.exposure_change_builder import (
    build_exposure_change,
    build_historical_exposure_change,
)

def _number(value):
    """Return a finite numeric value, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _percent(value):
    """Convert a decimal NAV weight to a percentage."""
    number = _number(value)
    return None if number is None else round(number * 100, 4)


def _evidence_map(investigation):
    """Index successful tool results by tool name and evidence index."""
    mapping = {}

    for index, entry in enumerate(investigation.get("evidence", [])):
        if not isinstance(entry, dict):
            continue

        name = entry.get("tool_name")
        result = entry.get("result")

        if (
            isinstance(name, str)
            and isinstance(result, dict)
            and result.get("status", result.get("overall_status"))
            not in ("ERROR", "FAILED", "NOT_AVAILABLE")
        ):
            mapping.setdefault(name, []).append((index, result))

    return mapping


def _first(mapping, name):
    entries = mapping.get(name, [])
    return entries[0] if entries else None


def build_visualizations(investigation):
    """Return chart specifications derived from stored tool evidence."""
    evidence = _evidence_map(investigation)
    charts = []

    metrics_entry = _first(evidence, "get_risk_metrics")
    policy_entry = _first(evidence, "retrieve_risk_policy")
    positions_entry = _first(evidence, "analyze_positions")

    # 1. Observed risk metric against its configured policy thresholds.
    if metrics_entry and policy_entry:
        metric_index, metrics_result = metrics_entry
        policy_index, policy_result = policy_entry

        metrics = metrics_result.get("data", {})
        rules = policy_result.get("rules", [])

        if not isinstance(metrics, dict):
            metrics = {}
        if not isinstance(rules, list):
            rules = []

        rows = []

        for rule in rules:
            if not isinstance(rule, dict):
                continue

            metric_name = rule.get("metric")
            if not isinstance(metric_name, str):
                continue

            observed = _number(metrics.get(metric_name))
            warning = _number(rule.get("warning_threshold"))
            breach = _number(rule.get("breach_threshold"))

            if None in (observed, warning, breach):
                continue

            unit = rule.get("unit")

            # Only percentage-of-NAV rules with decimal inputs
            # are supported by this chart.
            if unit not in ("pct_nav", "fraction", "decimal"):
                continue

            rows.append({
                "metric": rule.get("label") or metric_name,
                "observed": _percent(observed),
                "warning": _percent(warning),
                "breach": _percent(breach),
            })

        if rows:
            charts.append(
                ReportVisualization(
                    visualization_id="risk_limit_comparison",
                    title="Observed Risk vs. Policy Thresholds",
                    chart_type="bar",
                    data=rows,
                    x_field="metric",
                    y_fields=["observed", "warning", "breach"],
                    unit="percent",
                    evidence_indices=[
                        metric_index,
                        policy_index,
                    ],
                    notes=(
                        "Observed values and thresholds are "
                        "converted from decimal fractions to "
                        "percentages. Configured policy is "
                        "illustrative."
                    ),
                )
            )

    # 2. Asset-class composition from actual positions.
    if positions_entry:
        positions_index, positions_result = positions_entry
        positions = positions_result.get("positions", [])

        if isinstance(positions, list):
            allocation = {}

            for position in positions:
                if not isinstance(position, dict):
                    continue

                asset_class = position.get("asset_class")
                weight = _number(position.get("weight"))

                if not isinstance(asset_class, str) or weight is None:
                    continue

                allocation[asset_class] = (
                    allocation.get(asset_class, 0.0) + weight
                )

            if allocation:
                rows = [
                    {
                        "asset_class": name,
                        "weight": _percent(weight),
                    }
                    for name, weight in sorted(
                        allocation.items(),
                        key=lambda item: -item[1],
                    )
                ]

                # Pie charts require nonnegative slices and a
                # meaningful positive total.
                if (
                    all(row["weight"] >= 0 for row in rows)
                    and sum(row["weight"] for row in rows) > 0
                ):
                    charts.append(
                        ReportVisualization(
                            visualization_id="asset_class_allocation",
                            title="Portfolio Asset-Class Allocation",
                            chart_type="pie",
                            data=rows,
                            x_field="asset_class",
                            y_fields=["weight"],
                            unit="percent_nav",
                            evidence_indices=[positions_index],
                            notes=(
                                "Weights are based on recorded "
                                "positions. Cash may be excluded "
                                "from the displayed allocation."
                            ),
                        )
                    )

            # 3. Largest instrument weights.
            instrument_rows = []

            for position in positions:
                if not isinstance(position, dict):
                    continue

                symbol = position.get("instrument_id")
                weight = _number(position.get("weight"))

                if not isinstance(symbol, str) or weight is None:
                    continue

                instrument_rows.append({
                    "instrument": symbol,
                    "weight": _percent(weight),
                })

            instrument_rows.sort(
                key=lambda item: -abs(item["weight"])
            )

            if instrument_rows:
                charts.append(
                    ReportVisualization(
                        visualization_id="instrument_concentration",
                        title="Instrument Concentration",
                        chart_type="bar",
                        data=instrument_rows[:10],
                        x_field="instrument",
                        y_fields=["weight"],
                        unit="percent_nav",
                        evidence_indices=[positions_index],
                        notes=(
                            "Top 10 instruments by absolute NAV "
                            "weight; signed weights are preserved."
                        ),
                    )
                )

    exposure_charts, _ = build_exposure_change(
        investigation
    )

    charts.extend(exposure_charts)

    historical_charts = build_historical_exposure_change(
        investigation
    )

    charts.extend(historical_charts)

    return charts


def build_tables(investigation):
    """Build evidence-grounded analytical tables."""
    evidence = _evidence_map(investigation)
    tables = []

    quality_entry = _first(evidence, "check_data_quality")

    if quality_entry:
        index, result = quality_entry
        checks = result.get("checks", [])

        if isinstance(checks, list) and checks:
            rows = [
                {
                    "check": item.get("check"),
                    "status": item.get("status"),
                    "issue_count": item.get("issue_count"),
                }
                for item in checks
                if isinstance(item, dict)
            ]

            if rows:
                tables.append(
                    ReportTable(
                        table_id="data_quality_checks",
                        title="Data Quality and Reconciliation",
                        columns=[
                            "check",
                            "status",
                            "issue_count",
                        ],
                        rows=rows,
                        evidence_indices=[index],
                    )
                )

    transaction_entry = _first(
        evidence, "analyze_transactions"
    )

    if transaction_entry:
        index, result = transaction_entry
        summary = result.get("summary", {})
        window = result.get("analysis_window", {})

        if isinstance(summary, dict) and isinstance(window, dict):
            trade_count = summary.get("trade_count")

            if _number(trade_count) is not None:
                tables.append(
                    ReportTable(
                        table_id="transaction_summary",
                        title="Transaction Activity",
                        columns=[
                            "start_date",
                            "end_date",
                            "trade_count",
                            "gross_notional_usd",
                        ],
                        rows=[{
                            "start_date": window.get("start_date"),
                            "end_date": window.get("end_date"),
                            "trade_count": trade_count,
                            "gross_notional_usd": summary.get(
                                "gross_notional_usd"
                            ),
                        }],
                        evidence_indices=[index],
                    )
                )

    _, exposure_tables = build_exposure_change(
        investigation
    )

    tables.extend(exposure_tables)

    return tables
