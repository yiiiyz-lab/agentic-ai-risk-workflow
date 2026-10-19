
import json

import numpy as np
import pandas as pd
import pytest

from src.tools.analyze_risk_factors import (
    analyze_risk_factors,
    _detect_risk_signals,
)


PORTFOLIO_ID = "PORT_001"
AS_OF_DATE = "2026-09-30"

TAIL_DATES = [
    "2026-06-05",
    "2025-10-10",
    "2026-06-23",
]


@pytest.fixture(scope="module")
def result():
    """Calculate risk-factor analytics once per test module."""
    return analyze_risk_factors(
        PORTFOLIO_ID,
        AS_OF_DATE,
        scenario_dates=TAIL_DATES,
    )


# ============================================================
# 1. Basic output and historical window validation
# ============================================================

def test_success(result):
    assert result["status"] == "SUCCESS"


def test_default_windows(result):
    assert result["summary"]["evaluated_windows"] == [
        20, 63, 252
    ]


def test_six_instruments(result):
    assert result["summary"]["instrument_count"] == 6


def test_window_observation_counts(result):
    for window in result["window_statistics"]:
        assert window["status"] == "SUCCESS"

        assert (
            window["available_observations"]
            == window["window"]
        )


# ============================================================
# 2. Numerical validation
# ============================================================

def test_annualized_volatility_independent():
    prices = pd.read_csv(
        "data/sample/adjusted_prices.csv"
    ).set_index("date")

    selected = prices.loc[:AS_OF_DATE]

    returns = selected["QQQ"].pct_change(
        fill_method=None
    ).dropna()

    expected = (
        returns.tail(20).std(ddof=1)
        * np.sqrt(252)
    )

    output = analyze_risk_factors(
        PORTFOLIO_ID,
        AS_OF_DATE,
        windows=(20,),
    )

    qqq = next(
        item
        for item in output[
            "window_statistics"
        ][0]["instruments"]
        if item["instrument_id"] == "QQQ"
    )

    assert qqq["annualized_volatility"] == (
        pytest.approx(expected, abs=1e-12)
    )


def test_cumulative_return_independent():
    prices = pd.read_csv(
        "data/sample/adjusted_prices.csv"
    ).set_index("date")

    selected = prices.loc[:AS_OF_DATE]

    returns = selected["SPY"].pct_change(
        fill_method=None
    ).dropna()

    expected = (
        (1 + returns.tail(63)).prod() - 1
    )

    output = analyze_risk_factors(
        PORTFOLIO_ID,
        AS_OF_DATE,
        windows=(63,),
    )

    spy = next(
        item
        for item in output[
            "window_statistics"
        ][0]["instruments"]
        if item["instrument_id"] == "SPY"
    )

    assert spy["cumulative_return"] == (
        pytest.approx(expected, abs=1e-12)
    )


def test_correlation_matrix_symmetric(result):
    for window in result["window_statistics"]:
        matrix = window["correlation_matrix"]

        for left in matrix:
            for right in matrix:
                assert matrix[left][right] == (
                    pytest.approx(
                        matrix[right][left],
                        abs=1e-12,
                    )
                )


def test_correlation_diagonal(result):
    for window in result["window_statistics"]:
        matrix = window["correlation_matrix"]

        for symbol in matrix:
            assert matrix[symbol][symbol] == (
                pytest.approx(
                    1.0,
                    abs=1e-12,
                )
            )


def test_correlation_bounds(result):
    for window in result["window_statistics"]:
        for row in window[
            "correlation_matrix"
        ].values():
            for value in row.values():
                if value is not None:
                    assert -1.0 <= value <= 1.0


# ============================================================
# 3. Equity / Treasury ETF correlations
# ============================================================

def test_equity_rates_pairs(result):
    for item in result[
        "equity_rates_correlations"
    ]:
        assert len(item["pairs"]) == 4


def test_volatility_change(result):
    assert result[
        "volatility_change"
    ]["status"] == "SUCCESS"

    assert len(
        result[
            "volatility_change"
        ]["instruments"]
    ) == 6


# ============================================================
# 4. Historical tail-scenario diagnostics
# ============================================================

def test_tail_scenario_dates(result):
    returned = {
        item["historical_date"]
        for item in result[
            "historical_scenarios"
        ]
    }

    assert returned == set(TAIL_DATES)


def test_tail_scenario_returns_independent(result):
    prices = pd.read_csv(
        "data/sample/adjusted_prices.csv"
    ).set_index("date")

    expected = (
        prices.loc["2026-06-05", "QQQ"]
        / prices.shift(1).loc[
            "2026-06-05", "QQQ"
        ]
        - 1
    )

    scenario = next(
        item
        for item in result[
            "historical_scenarios"
        ]
        if item["historical_date"] == "2026-06-05"
    )

    assert scenario[
        "instrument_returns"
    ]["QQQ"] == pytest.approx(
        expected,
        abs=1e-12,
    )


# ============================================================
# 5. Point-in-time controls and input validation
# ============================================================

def test_no_future_scenario():
    with pytest.raises(ValueError):
        analyze_risk_factors(
            PORTFOLIO_ID,
            AS_OF_DATE,
            scenario_dates=["2026-10-01"],
        )


def test_as_of_scenario_rejected():
    with pytest.raises(ValueError):
        analyze_risk_factors(
            PORTFOLIO_ID,
            AS_OF_DATE,
            scenario_dates=[AS_OF_DATE],
        )


def test_unknown_portfolio():
    with pytest.raises(ValueError):
        analyze_risk_factors(
            "UNKNOWN_PORTFOLIO",
            AS_OF_DATE,
        )


def test_missing_as_of_date():
    with pytest.raises(ValueError):
        analyze_risk_factors(
            PORTFOLIO_ID,
            "2030-01-01",
        )


def test_invalid_window():
    with pytest.raises(ValueError):
        analyze_risk_factors(
            PORTFOLIO_ID,
            AS_OF_DATE,
            windows=(0,),
        )


def test_insufficient_history():
    output = analyze_risk_factors(
        PORTFOLIO_ID,
        "2020-02-03",
        windows=(252,),
    )

    assert output[
        "window_statistics"
    ][0]["status"] == "NOT_EVALUATED"


def test_json_serializable(result):
    json.dumps(result, allow_nan=False)


# ============================================================
# 6. Market-risk signal summary
# ============================================================

def test_risk_signal_summary_reconciles(result):
    summary = result["risk_signal_summary"]

    assert summary["total_signals"] == len(
        result["risk_signals"]
    )

    assert summary["total_signals"] == (
        summary["correlation_signals"]
        + summary["volatility_signals"]
        + summary["scenario_signals"]
    )


def test_risk_signals_are_not_policy_breaches(result):
    for signal in result["risk_signals"]:
        assert signal["severity"] == "WATCH"

        assert signal["category"] in {
            "CORRELATION",
            "VOLATILITY",
            "SCENARIO",
        }


# ============================================================
# 7. Correlation increase signals
# ============================================================

def test_correlation_increase_detected(result):
    matching = [
        signal
        for signal in result["risk_signals"]
        if signal["code"]
        == "EQUITY_RATES_CORRELATION_INCREASE"
    ]

    assert len(matching) >= 1

    for signal in matching:
        assert (
            signal["recent_20d_correlation"]
            >= 0.50
        )

        assert (
            signal["correlation_change"]
            >= 0.20
        )


# ============================================================
# 8. Cross-asset joint-loss signals
# ============================================================

def test_joint_loss_detected_on_june_5(result):
    matching = [
        signal
        for signal in result["risk_signals"]
        if (
            signal["code"]
            == "CROSS_ASSET_JOINT_LOSS"
            and signal["historical_date"]
            == "2026-06-05"
        )
    ]

    assert len(matching) == 1

    assert all(
        value < 0
        for value in matching[0][
            "instrument_returns"
        ].values()
    )


def test_joint_loss_not_detected_on_october_10(result):
    matching = [
        signal
        for signal in result["risk_signals"]
        if (
            signal["code"]
            == "CROSS_ASSET_JOINT_LOSS"
            and signal["historical_date"]
            == "2025-10-10"
        )
    ]

    assert matching == []


# ============================================================
# 9. Volatility increase signal — synthetic numerical test
# ============================================================

def test_volatility_signal_synthetic():
    volatility_change = {
        "status": "SUCCESS",
        "instruments": [
            {
                "instrument_id": "QQQ",
                "previous_20d_annualized_volatility": 0.16,
                "recent_20d_annualized_volatility": 0.24,
                "relative_change": 0.50,
            },
            {
                "instrument_id": "SPY",
                "previous_20d_annualized_volatility": 0.12,
                "recent_20d_annualized_volatility": 0.13,
                "relative_change": 0.0833333333,
            },
        ],
    }

    signals = _detect_risk_signals(
        window_results=[],
        volatility_change=volatility_change,
        scenario_results=[],
    )

    assert len(signals) == 1

    assert signals[0]["code"] == (
        "VOLATILITY_INCREASE"
    )

    assert signals[0]["instruments"] == [
        "QQQ"
    ]


# ============================================================
# 10. Empty inputs and JSON compatibility
# ============================================================

def test_no_risk_signals_when_inputs_empty():
    signals = _detect_risk_signals(
        window_results=[],
        volatility_change={
            "status": "NOT_EVALUATED",
        },
        scenario_results=[],
    )

    assert signals == []


def test_risk_signals_json_serializable(result):
    json.dumps(
        result["risk_signals"],
        allow_nan=False,
    )
