
import json

import pytest

from src.agent.tools import (
    get_risk_metrics,
    analyze_positions,
)


def test_get_risk_metrics():
    result = get_risk_metrics(
        "PORT_001",
        "2026-09-30",
    )

    assert result["tool"] == "get_risk_metrics"
    assert result["portfolio_id"] == "PORT_001"
    assert result["data"]["nav_usd"] > 0

    assert result["data"]["var_99_pct_nav"] > 0

    # Output must be JSON serializable.
    json.dumps(result, allow_nan=False)


def test_analyze_positions():
    result = analyze_positions(
        "PORT_001",
        "2026-09-30",
    )

    assert result["tool"] == "analyze_positions"
    assert result["nav_usd"] > 0

    largest = result["largest_position"]

    assert largest["instrument_id"] == "QQQ"
    assert largest["weight"] > 0.35

    assert len(result["positions"]) == 6

    json.dumps(result, allow_nan=False)


def test_missing_date_raises_error():
    with pytest.raises(ValueError):
        get_risk_metrics(
            "PORT_001",
            "2035-01-01",
        )


def test_wrong_portfolio_raises_error():
    with pytest.raises(ValueError):
        analyze_positions(
            "UNKNOWN_PORTFOLIO",
            "2026-09-30",
        )
