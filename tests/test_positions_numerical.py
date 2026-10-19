
import numpy as np
import pandas as pd
import pytest

from src.tools.analyze_positions import analyze_positions


PORTFOLIO_ID = "PORT_001"


def get_bridge(date):
    result = analyze_positions(PORTFOLIO_ID, date)
    return result["analysis"]["daily_weight_bridge"]


def test_market_only_day_has_no_trade_effect():
    """No transactions on 2026-09-30."""
    bridge = get_bridge("2026-09-30")

    assert bridge is not None

    for symbol, values in bridge.items():
        assert values["trade_effect"] == pytest.approx(
            0.0, abs=1e-9
        ), f"Unexpected trading effect for {symbol}"


@pytest.mark.parametrize(
    "trade_date",
    [
        "2022-06-10",
        "2023-03-10",
        "2024-08-02",
    ],
)
def test_transaction_day_has_nonzero_trade_effect(trade_date):
    """Portfolio rebalancing must affect at least one position."""
    bridge = get_bridge(trade_date)

    assert bridge is not None

    assert any(
        abs(values["trade_effect"]) > 1e-5
        for values in bridge.values()
    )


@pytest.mark.parametrize(
    "date",
    [
        "2022-06-10",
        "2023-03-10",
        "2024-08-02",
        "2026-09-30",
    ],
)
def test_daily_weight_bridge_reconciles(date):
    """Market effect + trade effect = total weight change."""
    bridge = get_bridge(date)

    for symbol, values in bridge.items():
        calculated_change = (
            values["market_effect"]
            + values["trade_effect"]
        )

        assert calculated_change == pytest.approx(
            values["total_change"],
            abs=1e-10,
        )


def test_2024_rebalance_direction():
    """
    On 2024-08-02:
    SELL QQQ and HYG, BUY IEF.

    Trading effects should reflect these directions.
    """
    bridge = get_bridge("2024-08-02")

    assert bridge["QQQ"]["trade_effect"] < 0
    assert bridge["HYG"]["trade_effect"] < 0
    assert bridge["IEF"]["trade_effect"] > 0


def test_2022_rebalance_direction():
    """
    On 2022-06-10:
    SELL IEF, BUY QQQ.
    """
    bridge = get_bridge("2022-06-10")

    assert bridge["IEF"]["trade_effect"] < 0
    assert bridge["QQQ"]["trade_effect"] > 0


def test_concentration_metrics_reconcile():
    """Concentration metrics must match underlying holdings."""
    result = analyze_positions(
        PORTFOLIO_ID,
        "2026-09-30",
    )

    positions = result["positions"]
    concentration = result["analysis"]["concentration"]

    weights = np.array(
        [position["weight"] for position in positions]
    )

    assert concentration["largest_weight"] == pytest.approx(
        weights.max()
    )

    assert concentration["top_two_weight"] == pytest.approx(
        np.sort(weights)[-2:].sum()
    )

    assert concentration["hhi"] == pytest.approx(
        np.square(weights).sum()
    )


def test_asset_class_exposures_reconcile():
    """Asset-class exposures must reconcile to invested assets."""
    result = analyze_positions(
        PORTFOLIO_ID,
        "2026-09-30",
    )

    exposures = result["analysis"]["asset_class_exposures"]

    total_exposure = sum(
        item["market_value_usd"]
        for item in exposures
    )

    total_positions = sum(
        item["market_value_usd"]
        for item in result["positions"]
    )

    assert total_exposure == pytest.approx(
        total_positions, abs=0.01
    )


def test_weight_changes_are_percentage_point_differences():
    """Historical weight changes must be decimal differences."""
    result = analyze_positions(
        PORTFOLIO_ID,
        "2026-09-30",
    )

    changes = result["analysis"]["historical_weight_changes"]

    assert changes["5d"] is not None
    assert changes["20d"] is not None

    for lookback in ("5d", "20d"):
        assert set(changes[lookback]) == {
            "SPY", "QQQ", "IEF", "TLT", "LQD", "HYG"
        }

        assert all(
            np.isfinite(value)
            for value in changes[lookback].values()
        )
