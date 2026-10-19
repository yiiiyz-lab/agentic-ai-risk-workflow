
import json

import pytest

from src.tools.get_pnl_attribution import get_pnl_attribution


PORTFOLIO_ID = "PORT_001"


def test_single_day_pnl_reconciles():
    result = get_pnl_attribution(
        PORTFOLIO_ID, "2026-09-30"
    )

    assert result["summary"][
        "reconciliation_status"
    ] == "PASS"

    assert result["summary"]["portfolio_pnl_usd"] == (
        pytest.approx(-69912.797474, abs=0.10)
    )


def test_instrument_pnl_sums_to_portfolio():
    result = get_pnl_attribution(
        PORTFOLIO_ID, "2026-09-30"
    )

    total = sum(
        item["pnl_usd"]
        for item in result["instrument_attribution"]
    )

    assert total == pytest.approx(
        result["summary"]["portfolio_pnl_usd"],
        abs=0.01,
    )


def test_asset_class_pnl_sums_to_portfolio():
    result = get_pnl_attribution(
        PORTFOLIO_ID, "2026-09-30"
    )

    total = sum(
        item["pnl_usd"]
        for item in result["asset_class_attribution"]
    )

    assert total == pytest.approx(
        result["summary"]["portfolio_pnl_usd"],
        abs=0.01,
    )


@pytest.mark.parametrize(
    "date",
    [
        "2022-06-10",
        "2023-03-10",
        "2024-08-02",
    ],
)
def test_rebalance_day_pnl_reconciles(date):
    result = get_pnl_attribution(
        PORTFOLIO_ID, date
    )

    assert result["summary"][
        "reconciliation_status"
    ] == "PASS"


def test_five_day_pnl_reconciles():
    result = get_pnl_attribution(
        PORTFOLIO_ID,
        "2026-09-30",
        lookback_days=5,
    )

    assert result["analysis_window"]["trading_days"] == 5
    assert len(result["daily_attribution"]) == 5
    assert result["summary"][
        "reconciliation_status"
    ] == "PASS"


def test_twenty_day_pnl_reconciles():
    result = get_pnl_attribution(
        PORTFOLIO_ID,
        "2026-09-30",
        lookback_days=20,
    )

    assert result["analysis_window"]["trading_days"] == 20
    assert result["summary"][
        "reconciliation_status"
    ] == "PASS"


def test_explicit_date_range():
    result = get_pnl_attribution(
        PORTFOLIO_ID,
        "2024-08-05",
        start_date="2024-08-01",
    )

    assert result["analysis_window"][
        "start_date"
    ] == "2024-08-01"

    assert result["analysis_window"][
        "end_date"
    ] == "2024-08-05"


def test_no_future_data():
    result = get_pnl_attribution(
        PORTFOLIO_ID,
        "2022-06-10",
        lookback_days=5,
    )

    assert all(
        item["date"] <= "2022-06-10"
        for item in result["daily_attribution"]
    )


def test_json_serializable():
    result = get_pnl_attribution(
        PORTFOLIO_ID,
        "2026-09-30",
        lookback_days=5,
    )

    json.dumps(result, allow_nan=False)


def test_invalid_lookback():
    with pytest.raises(ValueError):
        get_pnl_attribution(
            PORTFOLIO_ID,
            "2026-09-30",
            lookback_days=0,
        )


def test_missing_date():
    with pytest.raises(ValueError):
        get_pnl_attribution(
            PORTFOLIO_ID,
            "2035-01-01",
        )


def test_invalid_date_range():
    with pytest.raises(ValueError):
        get_pnl_attribution(
            PORTFOLIO_ID,
            "2026-09-30",
            start_date="2026-10-01",
        )
