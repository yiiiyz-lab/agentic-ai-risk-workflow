
import json

import numpy as np
import pandas as pd
import pytest

from src.tools.check_data_quality import check_data_quality


@pytest.fixture
def sample_paths(tmp_path):
    dates = pd.bdate_range(
        end="2026-09-30",
        periods=260,
    )

    prices = pd.DataFrame({
        "date": dates,
        "SPY": np.linspace(100, 120, len(dates)),
        "IEF": np.linspace(100, 105, len(dates)),
    })

    portfolio = pd.DataFrame({
        "date": dates,
        "portfolio_id": "TEST_PORT",
        "nav_usd": [1000.0] * len(dates),
        "cash_usd": [100.0] * len(dates),
        "daily_pnl_usd": [0.0] * len(dates),
    })

    positions = pd.DataFrame([
        {
            "date": "2026-09-30",
            "portfolio_id": "TEST_PORT",
            "instrument_id": "SPY",
            "quantity": 5.0,
            "adjusted_price": 120.0,
            "market_value_usd": 600.0,
        },
        {
            "date": "2026-09-30",
            "portfolio_id": "TEST_PORT",
            "instrument_id": "IEF",
            "quantity": 300.0 / 105.0,
            "adjusted_price": 105.0,
            "market_value_usd": 300.0,
        },
    ])

    paths = {
        "prices_path": tmp_path / "prices.csv",
        "positions_path": tmp_path / "positions.csv",
        "portfolio_path": tmp_path / "portfolio.csv",
    }

    prices.to_csv(paths["prices_path"], index=False)
    positions.to_csv(paths["positions_path"], index=False)
    portfolio.to_csv(paths["portfolio_path"], index=False)

    return paths


def execute(paths, **kwargs):
    return check_data_quality(
        portfolio_id="TEST_PORT",
        as_of_date="2026-09-30",
        **paths,
        **kwargs,
    )


def issue_codes(result):
    return {
        issue["code"]
        for issue in result["issues"]
    }


def test_clean_data_passes(sample_paths):
    result = execute(sample_paths)

    assert result["overall_status"] == "PASS"
    assert result["summary"]["issue_count"] == 0


def test_nav_reconciliation_passes(sample_paths):
    result = execute(sample_paths)

    assert result["reconciliation"]["nav"][
        "status"
    ] == "PASS"


def test_daily_pnl_reconciliation_passes(sample_paths):
    result = execute(sample_paths)

    assert result["reconciliation"]["daily_pnl"][
        "status"
    ] == "PASS"


def test_history_coverage_passes(sample_paths):
    result = execute(sample_paths)

    assert result["coverage"]["SPY"][
        "available_return_observations"
    ] == 252


def test_duplicate_positions_detected(sample_paths):
    df = pd.read_csv(
        sample_paths["positions_path"]
    )

    df = pd.concat(
        [df, df.iloc[[0]]],
        ignore_index=True,
    )

    df.to_csv(
        sample_paths["positions_path"],
        index=False,
    )

    result = execute(sample_paths)

    assert "DUPLICATE_POSITIONS" in issue_codes(result)
    assert result["overall_status"] == "FAIL"


def test_missing_price_column_detected(sample_paths):
    df = pd.read_csv(
        sample_paths["prices_path"]
    )

    df = df.drop(columns=["IEF"])
    df.to_csv(
        sample_paths["prices_path"],
        index=False,
    )

    result = execute(sample_paths)

    assert "MISSING_PRICE_COLUMNS" in issue_codes(result)


def test_invalid_price_detected(sample_paths):
    df = pd.read_csv(
        sample_paths["prices_path"]
    )

    df.loc[df.index[-5], "SPY"] = -1.0

    df.to_csv(
        sample_paths["prices_path"],
        index=False,
    )

    result = execute(sample_paths)

    assert "INVALID_ADJUSTED_PRICE" in issue_codes(result)
    assert result["overall_status"] == "FAIL"


def test_missing_price_detected(sample_paths):
    df = pd.read_csv(
        sample_paths["prices_path"]
    )

    df.loc[df.index[-5], "SPY"] = np.nan

    df.to_csv(
        sample_paths["prices_path"],
        index=False,
    )

    result = execute(sample_paths)

    assert "INVALID_ADJUSTED_PRICE" in issue_codes(result)


def test_duplicate_price_dates_detected(sample_paths):
    df = pd.read_csv(
        sample_paths["prices_path"]
    )

    df = pd.concat(
        [df, df.iloc[[-2]]],
        ignore_index=True,
    )

    df.to_csv(
        sample_paths["prices_path"],
        index=False,
    )

    result = execute(sample_paths)

    assert "DUPLICATE_PRICE_DATES" in issue_codes(result)


def test_position_valuation_mismatch(sample_paths):
    df = pd.read_csv(
        sample_paths["positions_path"]
    )

    df.loc[
        df["instrument_id"] == "SPY",
        "market_value_usd",
    ] = 650.0

    df.to_csv(
        sample_paths["positions_path"],
        index=False,
    )

    result = execute(sample_paths)

    assert (
        "POSITION_VALUATION_MISMATCH"
        in issue_codes(result)
    )


def test_nav_mismatch(sample_paths):
    df = pd.read_csv(
        sample_paths["portfolio_path"]
    )

    df.loc[df.index[-1], "nav_usd"] = 1100.0

    df.to_csv(
        sample_paths["portfolio_path"],
        index=False,
    )

    result = execute(sample_paths)

    assert (
        "NAV_RECONCILIATION_FAILURE"
        in issue_codes(result)
    )


def test_daily_pnl_mismatch(sample_paths):
    df = pd.read_csv(
        sample_paths["portfolio_path"]
    )

    df.loc[
        df.index[-1],
        "daily_pnl_usd",
    ] = 50.0

    df.to_csv(
        sample_paths["portfolio_path"],
        index=False,
    )

    result = execute(sample_paths)

    assert (
        "DAILY_PNL_RECONCILIATION_FAILURE"
        in issue_codes(result)
    )


def test_insufficient_history(sample_paths):
    result = execute(
        sample_paths,
        lookback_days=300,
    )

    assert (
        "INSUFFICIENT_RETURN_HISTORY"
        in issue_codes(result)
    )

    assert result["overall_status"] == "WARNING"


def test_future_rows_not_used(sample_paths):
    df = pd.read_csv(
        sample_paths["prices_path"]
    )

    future = pd.DataFrame([{
        "date": "2026-10-01",
        "SPY": -100.0,
        "IEF": -100.0,
    }])

    df = pd.concat(
        [df, future],
        ignore_index=True,
    )

    df.to_csv(
        sample_paths["prices_path"],
        index=False,
    )

    result = execute(sample_paths)

    assert result["overall_status"] == "PASS"


def test_missing_source_returns_failure(sample_paths):
    sample_paths["prices_path"].unlink()

    result = execute(sample_paths)

    assert result["overall_status"] == "FAIL"
    assert "SOURCE_UNAVAILABLE" in issue_codes(result)


def test_json_serializable(sample_paths):
    result = execute(sample_paths)

    json.dumps(result, allow_nan=False)
