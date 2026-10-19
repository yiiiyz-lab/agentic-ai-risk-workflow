
import json

import pytest

from src.tools.analyze_transactions import analyze_transactions


PORTFOLIO_ID = "PORT_001"


def test_2024_rebalance():
    result = analyze_transactions(
        PORTFOLIO_ID,
        "2024-08-02",
        lookback_days=1,
    )

    assert result["summary"]["trade_count"] == 3

    assert result["summary"]["buy_notional_usd"] == pytest.approx(
        10_000_000, abs=0.10
    )

    assert result["summary"]["sell_notional_usd"] == pytest.approx(
        10_000_000, abs=0.10
    )

    assert result["summary"]["net_cash_flow_usd"] == pytest.approx(
        0, abs=0.10
    )

    assert result["daily_reconciliations"][0][
        "reconciliation_status"
    ] == "PASS"


def test_2024_trade_effect_directions():
    result = analyze_transactions(
        PORTFOLIO_ID,
        "2024-08-02",
        lookback_days=1,
    )

    effects = result["daily_reconciliations"][0][
        "position_weight_effects"
    ]

    assert effects["QQQ"]["trade_effect"] < 0
    assert effects["HYG"]["trade_effect"] < 0
    assert effects["IEF"]["trade_effect"] > 0


def test_no_trade_day():
    result = analyze_transactions(
        PORTFOLIO_ID,
        "2026-09-30",
        lookback_days=1,
    )

    assert result["summary"]["trade_count"] == 0
    assert result["transactions"] == []
    assert result["daily_reconciliations"] == []


def test_no_future_transactions():
    result = analyze_transactions(
        PORTFOLIO_ID,
        "2022-06-10",
        lookback_days=20,
    )

    assert all(
        trade["trade_date"] <= "2022-06-10"
        for trade in result["transactions"]
    )


def test_json_serializable():
    result = analyze_transactions(
        PORTFOLIO_ID,
        "2024-08-02",
        lookback_days=1,
    )

    json.dumps(result, allow_nan=False)


def test_invalid_lookback():
    with pytest.raises(ValueError):
        analyze_transactions(
            PORTFOLIO_ID,
            "2024-08-02",
            lookback_days=0,
        )


def test_missing_date():
    with pytest.raises(ValueError):
        analyze_transactions(
            PORTFOLIO_ID,
            "2035-01-01",
        )
