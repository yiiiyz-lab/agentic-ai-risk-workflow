
"""Tests for daily and historical exposure change visualizations."""

from src.reporting.exposure_change_builder import (
    build_exposure_change,
    build_historical_exposure_change,
)


def make_investigation():
    """Create a sample investigation with daily weight attribution."""
    return {
        "evidence": [
            {
                "tool_name": "analyze_positions",
                "result": {
                    "status": "PASS",
                    "analysis": {
                        "daily_weight_bridge": {
                            "QQQ": {
                                "previous_weight": 0.40,
                                "pre_trade_weight": 0.42,
                                "current_weight": 0.43,
                                "market_effect": 0.02,
                                "trade_effect": 0.01,
                                "total_change": 0.03,
                            },
                            "SPY": {
                                "previous_weight": 0.30,
                                "pre_trade_weight": 0.29,
                                "current_weight": 0.29,
                                "market_effect": -0.01,
                                "trade_effect": 0.0,
                                "total_change": -0.01,
                            },
                        }
                    },
                },
            }
        ]
    }


def test_daily_attribution():
    """Daily market and trading effects should reconcile."""
    charts, tables = build_exposure_change(
        make_investigation()
    )

    assert len(charts) == 1
    assert len(tables) == 1

    chart = charts[0]

    assert chart.visualization_id == (
        "daily_exposure_attribution"
    )
    assert chart.chart_type == "bar"
    assert chart.unit == "percentage_points"
    assert chart.evidence_indices == [0]

    qqq = next(
        row
        for row in chart.data
        if row["instrument"] == "QQQ"
    )

    assert qqq["market_effect"] == 2.0
    assert qqq["trade_effect"] == 1.0
    assert qqq["total_change"] == 3.0

    spy = next(
        row
        for row in chart.data
        if row["instrument"] == "SPY"
    )

    assert spy["market_effect"] == -1.0
    assert spy["trade_effect"] == 0.0
    assert spy["total_change"] == -1.0

    table = tables[0]

    assert table.table_id == "daily_weight_bridge"
    assert table.evidence_indices == [0]
    assert len(table.rows) == 2

    qqq_table = next(
        row
        for row in table.rows
        if row["instrument"] == "QQQ"
    )

    assert qqq_table["previous_weight_pct"] == 40.0
    assert qqq_table["pre_trade_weight_pct"] == 42.0
    assert qqq_table["current_weight_pct"] == 43.0


def test_inconsistent_bridge_is_excluded():
    """An inconsistent weight bridge must not be visualized."""
    investigation = make_investigation()

    bridge = investigation["evidence"][0][
        "result"
    ]["analysis"]["daily_weight_bridge"]

    bridge["QQQ"]["total_change"] = 0.50

    charts, tables = build_exposure_change(
        investigation
    )

    assert len(charts) == 1
    assert len(tables) == 1

    assert len(charts[0].data) == 1
    assert charts[0].data[0]["instrument"] == "SPY"

    assert len(tables[0].rows) == 1
    assert tables[0].rows[0]["instrument"] == "SPY"


def test_missing_bridge():
    """No daily attribution should be created without evidence."""
    charts, tables = build_exposure_change({
        "evidence": []
    })

    assert charts == []
    assert tables == []


def test_historical_exposure_change():
    """Historical changes should be converted to percentage points."""
    investigation = {
        "evidence": [
            {
                "tool_name": "analyze_positions",
                "result": {
                    "status": "PASS",
                    "analysis": {
                        "historical_weight_changes": {
                            "5d": {
                                "QQQ": 0.0026130767052739423,
                                "SPY": -0.001,
                            },
                            "20d": {
                                "QQQ": 0.015118898294669003,
                                "SPY": -0.002,
                            },
                        }
                    },
                },
            }
        ]
    }

    charts = build_historical_exposure_change(
        investigation
    )

    assert len(charts) == 1

    chart = charts[0]

    assert chart.visualization_id == (
        "historical_exposure_change"
    )
    assert chart.chart_type == "bar"
    assert chart.unit == "percentage_points"
    assert chart.evidence_indices == [0]

    assert chart.x_field == "instrument"
    assert chart.y_fields == [
        "change_5d",
        "change_20d",
    ]

    assert len(chart.data) == 2

    qqq = next(
        row
        for row in chart.data
        if row["instrument"] == "QQQ"
    )

    assert qqq["change_5d"] == 0.261308
    assert qqq["change_20d"] == 1.51189

    spy = next(
        row
        for row in chart.data
        if row["instrument"] == "SPY"
    )

    assert spy["change_5d"] == -0.1
    assert spy["change_20d"] == -0.2


def test_historical_change_requires_both_windows():
    """Only instruments with both windows should be displayed."""
    investigation = {
        "evidence": [
            {
                "tool_name": "analyze_positions",
                "result": {
                    "status": "PASS",
                    "analysis": {
                        "historical_weight_changes": {
                            "5d": {"QQQ": 0.01},
                            "20d": {"SPY": 0.02},
                        }
                    },
                },
            }
        ]
    }

    charts = build_historical_exposure_change(
        investigation
    )

    assert charts == []


def test_historical_change_excludes_invalid_values():
    """Invalid or non-finite values must not enter charts."""
    investigation = {
        "evidence": [
            {
                "tool_name": "analyze_positions",
                "result": {
                    "status": "PASS",
                    "analysis": {
                        "historical_weight_changes": {
                            "5d": {
                                "QQQ": float("nan"),
                                "SPY": 0.005,
                            },
                            "20d": {
                                "QQQ": 0.02,
                                "SPY": 0.01,
                            },
                        }
                    },
                },
            }
        ]
    }

    charts = build_historical_exposure_change(
        investigation
    )

    assert len(charts) == 1
    assert len(charts[0].data) == 1
    assert charts[0].data[0]["instrument"] == "SPY"


def test_historical_change_missing_evidence():
    """No historical chart should be produced without evidence."""
    charts = build_historical_exposure_change({
        "evidence": []
    })

    assert charts == []


def test_historical_change_rejects_failed_tool():
    """Failed analytical tools must not supply chart data."""
    investigation = {
        "evidence": [
            {
                "tool_name": "analyze_positions",
                "result": {
                    "status": "FAILED",
                    "analysis": {
                        "historical_weight_changes": {
                            "5d": {"QQQ": 0.01},
                            "20d": {"QQQ": 0.02},
                        }
                    },
                },
            }
        ]
    }

    charts = build_historical_exposure_change(
        investigation
    )

    assert charts == []
