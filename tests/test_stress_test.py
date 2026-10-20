
import json

import pandas as pd
import pytest

from src.tools.run_stress_test import run_stress_test


PORTFOLIO_ID = "TEST_PORT"
AS_OF_DATE = "2026-09-30"

HYPOTHETICAL = {
    "id": "TEST_RISK_OFF",
    "name": "Illustrative Risk-Off",
    "type": "hypothetical",
    "shocks": {
        "SPY": -0.10,
        "IEF": 0.05,
        "HYG": -0.08,
    },
}

HISTORICAL = {
    "id": "TEST_HISTORICAL",
    "name": "Historical ETF Movement",
    "type": "historical",
    "start_date": "2026-09-28",
    "end_date": "2026-09-29",
}


@pytest.fixture
def sample_paths(tmp_path):
    positions = pd.DataFrame([
        {
            "date": AS_OF_DATE,
            "portfolio_id": PORTFOLIO_ID,
            "instrument_id": "SPY",
            "market_value_usd": 400.0,
        },
        {
            "date": AS_OF_DATE,
            "portfolio_id": PORTFOLIO_ID,
            "instrument_id": "IEF",
            "market_value_usd": 300.0,
        },
        {
            "date": AS_OF_DATE,
            "portfolio_id": PORTFOLIO_ID,
            "instrument_id": "HYG",
            "market_value_usd": 200.0,
        },
    ])

    portfolio = pd.DataFrame([
        {
            "date": AS_OF_DATE,
            "portfolio_id": PORTFOLIO_ID,
            "nav_usd": 1000.0,
            "cash_usd": 100.0,
        }
    ])

    prices = pd.DataFrame([
        {
            "date": "2026-09-28",
            "SPY": 100.0,
            "IEF": 100.0,
            "HYG": 100.0,
        },
        {
            "date": "2026-09-29",
            "SPY": 90.0,
            "IEF": 105.0,
            "HYG": 92.0,
        },
        {
            "date": AS_OF_DATE,
            "SPY": 95.0,
            "IEF": 104.0,
            "HYG": 94.0,
        },
    ])

    positions_path = tmp_path / "positions.csv"
    portfolio_path = tmp_path / "portfolio.csv"
    prices_path = tmp_path / "prices.csv"

    positions.to_csv(positions_path, index=False)
    portfolio.to_csv(portfolio_path, index=False)
    prices.to_csv(prices_path, index=False)

    return {
        "positions_path": positions_path,
        "portfolio_path": portfolio_path,
        "prices_path": prices_path,
    }


def execute(paths, scenarios=None, **kwargs):
    if scenarios is None:
        scenarios = [HYPOTHETICAL]

    return run_stress_test(
        portfolio_id=PORTFOLIO_ID,
        as_of_date=AS_OF_DATE,
        scenarios=scenarios,
        **paths,
        **kwargs,
    )


def test_success(sample_paths):
    result = execute(sample_paths)
    assert result["status"] == "SUCCESS"


def test_hypothetical_portfolio_pnl(sample_paths):
    result = execute(sample_paths)
    scenario = result["scenarios"][0]

    assert scenario[
        "portfolio_stress_pnl_usd"
    ] == pytest.approx(-41.0)


def test_hypothetical_stressed_nav(sample_paths):
    result = execute(sample_paths)
    scenario = result["scenarios"][0]

    assert scenario[
        "stressed_nav_usd"
    ] == pytest.approx(959.0)


def test_hypothetical_loss_pct_nav(sample_paths):
    result = execute(sample_paths)
    scenario = result["scenarios"][0]

    assert scenario[
        "portfolio_stress_loss_pct_nav"
    ] == pytest.approx(0.041)


def test_instrument_attribution(sample_paths):
    result = execute(sample_paths)
    items = {
        item["instrument_id"]: item
        for item in result["scenarios"][0][
            "instrument_attribution"
        ]
    }

    assert items["SPY"][
        "stress_pnl_usd"
    ] == pytest.approx(-40.0)

    assert items["IEF"][
        "stress_pnl_usd"
    ] == pytest.approx(15.0)

    assert items["HYG"][
        "stress_pnl_usd"
    ] == pytest.approx(-16.0)


def test_asset_class_attribution(sample_paths):
    result = execute(sample_paths)
    items = {
        item["asset_class"]: item
        for item in result["scenarios"][0][
            "asset_class_attribution"
        ]
    }

    assert items["Equity"][
        "stress_pnl_usd"
    ] == pytest.approx(-40.0)

    assert items["Rates"][
        "stress_pnl_usd"
    ] == pytest.approx(15.0)

    assert items["Credit"][
        "stress_pnl_usd"
    ] == pytest.approx(-16.0)


def test_reconciliation(sample_paths):
    result = execute(sample_paths)
    reconciliation = result["scenarios"][0][
        "reconciliation"
    ]

    assert reconciliation["status"] == "PASS"
    assert reconciliation[
        "asset_class_pnl_difference_usd"
    ] == pytest.approx(0.0)
    assert reconciliation[
        "stressed_nav_difference_usd"
    ] == pytest.approx(0.0)


def test_historical_scenario(sample_paths):
    result = execute(
        sample_paths,
        scenarios=[HISTORICAL],
    )

    scenario = result["scenarios"][0]

    assert scenario[
        "portfolio_stress_pnl_usd"
    ] == pytest.approx(-41.0)

    assert scenario[
        "scenario_source"
    ]["shock_source"] == (
        "historical_adjusted_prices"
    )


def test_multiple_scenarios_and_worst(sample_paths):
    mild = {
        "id": "MILD",
        "name": "Mild Sell-Off",
        "type": "hypothetical",
        "shocks": {
            "SPY": -0.01,
            "IEF": 0.0,
            "HYG": 0.0,
        },
    }

    result = execute(
        sample_paths,
        scenarios=[mild, HYPOTHETICAL],
    )

    assert result["summary"][
        "scenario_count"
    ] == 2

    assert result["summary"][
        "worst_scenario_id"
    ] == "TEST_RISK_OFF"


def test_future_historical_scenario_rejected(sample_paths):
    future = {
        "id": "FUTURE",
        "name": "Future Scenario",
        "type": "historical",
        "start_date": "2026-09-29",
        "end_date": "2026-10-01",
    }

    with pytest.raises(ValueError):
        execute(
            sample_paths,
            scenarios=[future],
        )


def test_as_of_historical_end_rejected(sample_paths):
    scenario = {
        "id": "AS_OF",
        "name": "As-of Scenario",
        "type": "historical",
        "start_date": "2026-09-28",
        "end_date": AS_OF_DATE,
    }

    with pytest.raises(ValueError):
        execute(
            sample_paths,
            scenarios=[scenario],
        )


def test_missing_hypothetical_shock_rejected(sample_paths):
    scenario = {
        "id": "MISSING",
        "name": "Missing Shock",
        "type": "hypothetical",
        "shocks": {
            "SPY": -0.10,
            "IEF": 0.05,
        },
    }

    with pytest.raises(ValueError):
        execute(
            sample_paths,
            scenarios=[scenario],
        )


def test_below_minus_100_pct_rejected(sample_paths):
    scenario = {
        "id": "INVALID",
        "name": "Invalid Shock",
        "type": "hypothetical",
        "shocks": {
            "SPY": -1.10,
            "IEF": 0.05,
            "HYG": -0.08,
        },
    }

    with pytest.raises(ValueError):
        execute(
            sample_paths,
            scenarios=[scenario],
        )


def test_duplicate_scenario_ids_rejected(sample_paths):
    with pytest.raises(ValueError):
        execute(
            sample_paths,
            scenarios=[
                HYPOTHETICAL,
                HYPOTHETICAL,
            ],
        )


def test_unknown_portfolio_rejected(sample_paths):
    with pytest.raises(ValueError):
        run_stress_test(
            portfolio_id="UNKNOWN",
            as_of_date=AS_OF_DATE,
            scenarios=[HYPOTHETICAL],
            **sample_paths,
        )


def test_no_scenarios_rejected(sample_paths):
    with pytest.raises(ValueError):
        execute(
            sample_paths,
            scenarios=[],
        )


def test_nav_reconciliation_failure(sample_paths):
    portfolio = pd.read_csv(
        sample_paths["portfolio_path"]
    )
    portfolio["nav_usd"] = 1100.0
    portfolio.to_csv(
        sample_paths["portfolio_path"],
        index=False,
    )

    with pytest.raises(ValueError):
        execute(sample_paths)


def test_duplicate_positions_rejected(sample_paths):
    positions = pd.read_csv(
        sample_paths["positions_path"]
    )

    positions = pd.concat(
        [positions, positions.iloc[[0]]],
        ignore_index=True,
    )

    positions.to_csv(
        sample_paths["positions_path"],
        index=False,
    )

    with pytest.raises(ValueError):
        execute(sample_paths)


def test_json_serializable(sample_paths):
    result = execute(
        sample_paths,
        scenarios=[
            HYPOTHETICAL,
            HISTORICAL,
        ],
    )

    json.dumps(result, allow_nan=False)


def test_hypothetical_warning(sample_paths):
    result = execute(sample_paths)

    codes = {
        item["code"]
        for item in result["governance_warnings"]
    }

    assert (
        "ILLUSTRATIVE_HYPOTHETICAL_SHOCKS"
        in codes
    )
