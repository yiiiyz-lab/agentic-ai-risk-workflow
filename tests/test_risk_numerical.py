
import numpy as np
import pandas as pd
import pytest

from src.risk.calculator import calculate_risk_metrics


SYMBOLS = ["SPY", "QQQ", "IEF", "TLT", "LQD", "HYG"]
WINDOW = 252
TRADE_DATES = ["2022-06-10", "2023-03-10", "2024-08-02"]


def independent_historical_var_es(
    historical_returns, position_values
):
    """
    Independent reference implementation.

    Calculate scenario losses using explicit matrix multiplication.
    Compute VaR by sorting losses and applying linear interpolation.
    ES is the mean of losses at or above VaR.
    """
    scenario_pnl = (
        historical_returns.to_numpy(dtype=float)
        @ position_values.to_numpy(dtype=float)
    )
    losses = np.sort(-scenario_pnl)

    rank = 0.99 * (len(losses) - 1)
    lower = int(np.floor(rank))
    upper = int(np.ceil(rank))
    fraction = rank - lower

    var = (
        losses[lower] * (1 - fraction)
        + losses[upper] * fraction
    )
    es = losses[losses >= var].mean()

    return max(0.0, var), max(0.0, es)


@pytest.fixture(scope="module")
def market_data():
    prices = pd.read_csv(
        "data/sample/adjusted_prices.csv",
        parse_dates=["date"],
    ).set_index("date")

    positions = pd.read_csv(
        "data/sample/daily_positions.csv",
        parse_dates=["date"],
    )

    transactions = pd.read_csv(
        "data/sample/transactions.csv",
        parse_dates=["trade_date"],
    )

    metrics = calculate_risk_metrics().set_index("date")

    return prices, positions, transactions, metrics


def test_known_returns_and_positions(tmp_path):
    """
    Use a deterministic artificial market with known returns
    and constant holdings. Independently calculate VaR and ES.
    """
    dates = pd.bdate_range("2020-01-01", periods=260)

    # Deterministic, varied daily returns, including tail events.
    base_returns = np.array([
        -0.08, -0.05, -0.03, -0.02, -0.01,
        0.00, 0.005, 0.01, 0.02, 0.03,
    ])

    returns = np.resize(base_returns, len(dates) - 1)

    prices = pd.DataFrame(index=dates)

    for j, symbol in enumerate(SYMBOLS):
        symbol_returns = returns * (1 + j * 0.1)
        prices[symbol] = 100.0 * np.cumprod(
            np.r_[1.0, 1.0 + symbol_returns]
        )

    quantities = pd.Series(
        [1000, 800, 1200, 700, 900, 600],
        index=SYMBOLS,
        dtype=float,
    )

    position_values = prices.mul(quantities, axis=1)
    nav = position_values.sum(axis=1)

    portfolio = pd.DataFrame({
        "date": dates,
        "portfolio_id": "PORT_TEST",
        "nav_usd": nav.to_numpy(),
        "daily_pnl_usd": nav.diff().fillna(0).to_numpy(),
        "cash_usd": 0.0,
        "trade_count": 0,
    })

    position_rows = []

    for date in dates:
        for symbol in SYMBOLS:
            position_rows.append({
                "date": date,
                "portfolio_id": "PORT_TEST",
                "instrument_id": symbol,
                "quantity": quantities[symbol],
                "adjusted_price": prices.loc[date, symbol],
                "market_value_usd": position_values.loc[
                    date, symbol
                ],
                "weight": (
                    position_values.loc[date, symbol]
                    / nav.loc[date]
                ),
            })

    prices.reset_index(names="date").to_csv(
        tmp_path / "adjusted_prices.csv", index=False
    )
    portfolio.to_csv(
        tmp_path / "daily_portfolio.csv", index=False
    )
    pd.DataFrame(position_rows).to_csv(
        tmp_path / "daily_positions.csv", index=False
    )

    actual = calculate_risk_metrics(
        data_dir=tmp_path
    ).set_index("date")

    # Use the first valid VaR date.
    date = dates[253]

    historical_returns = prices.pct_change(
        fill_method=None
    ).iloc[1:253]

    current_values = position_values.loc[date]

    expected_var, expected_es = (
        independent_historical_var_es(
            historical_returns, current_values
        )
    )

    assert actual.loc[date, "var_99_usd"] == pytest.approx(
        expected_var, rel=1e-10, abs=1e-6
    )
    assert actual.loc[date, "es_99_usd"] == pytest.approx(
        expected_es, rel=1e-10, abs=1e-6
    )


@pytest.mark.parametrize("trade_date", TRADE_DATES)
def test_transaction_changes_holdings(
    market_data, trade_date
):
    prices, positions, transactions, metrics = market_data

    date = pd.Timestamp(trade_date)
    previous_date = prices.index[
        prices.index.get_loc(date) - 1
    ]

    before = (
        positions.loc[
            positions["date"] == previous_date
        ]
        .set_index("instrument_id")["quantity"]
        .reindex(SYMBOLS)
    )

    after = (
        positions.loc[
            positions["date"] == date
        ]
        .set_index("instrument_id")["quantity"]
        .reindex(SYMBOLS)
    )

    trades = transactions.loc[
        transactions["trade_date"] == date
    ]

    expected_change = pd.Series(
        0.0, index=SYMBOLS
    )

    for _, trade in trades.iterrows():
        sign = 1 if trade["side"] == "BUY" else -1
        expected_change[trade["instrument_id"]] += (
            sign * trade["quantity"]
        )

    np.testing.assert_allclose(
        (after - before).to_numpy(),
        expected_change.to_numpy(),
        rtol=1e-9,
        atol=1e-6,
    )


@pytest.mark.parametrize("trade_date", TRADE_DATES)
def test_post_trade_var_matches_current_holdings(
    market_data, trade_date
):
    prices, positions, transactions, metrics = market_data

    date = pd.Timestamp(trade_date)
    date_index = prices.index.get_loc(date)
    previous_date = prices.index[date_index - 1]

    quantities_before = (
        positions.loc[
            positions["date"] == previous_date
        ]
        .set_index("instrument_id")["quantity"]
        .reindex(SYMBOLS)
    )

    quantities_after = (
        positions.loc[
            positions["date"] == date
        ]
        .set_index("instrument_id")["quantity"]
        .reindex(SYMBOLS)
    )

    current_prices = prices.loc[date, SYMBOLS]

    historical_returns = prices[SYMBOLS].pct_change(
        fill_method=None
    ).iloc[date_index - WINDOW:date_index]

    # Counterfactual: previous holdings valued at today's prices.
    pre_trade_values = quantities_before * current_prices

    # Actual closing holdings after the synthetic trades.
    post_trade_values = quantities_after * current_prices

    pre_var, pre_es = independent_historical_var_es(
        historical_returns, pre_trade_values
    )

    post_var, post_es = independent_historical_var_es(
        historical_returns, post_trade_values
    )

    actual_var = metrics.loc[date, "var_99_usd"]
    actual_es = metrics.loc[date, "es_99_usd"]

    assert actual_var == pytest.approx(
        post_var, rel=1e-10, abs=1e-5
    )
    assert actual_es == pytest.approx(
        post_es, rel=1e-10, abs=1e-5
    )

    # Confirm trades materially changed the scenario P&L
    # distribution; VaR/ES need not always change.
    pre_scenarios = (
        historical_returns.to_numpy()
        @ pre_trade_values.to_numpy()
    )
    post_scenarios = (
        historical_returns.to_numpy()
        @ post_trade_values.to_numpy()
    )

    assert not np.allclose(
        pre_scenarios, post_scenarios,
        rtol=1e-10, atol=1e-6
    )

    print(
        f"\n{trade_date}: "
        f"Pre-trade VaR=${pre_var:,.2f}, "
        f"Post-trade VaR=${post_var:,.2f}, "
        f"Trade effect=${post_var - pre_var:,.2f}"
    )
