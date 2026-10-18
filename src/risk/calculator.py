
from pathlib import Path
import numpy as np
import pandas as pd

ASSET_GROUPS = {
    "SPY": "Equity",
    "QQQ": "Equity",
    "IEF": "Rates",
    "TLT": "Rates",
    "LQD": "Credit",
    "HYG": "Credit",
}

TRADING_DAYS = 252
VAR_WINDOW = 252
VOL_WINDOW = 63
BETA_WINDOW = 126
CONFIDENCE = 0.99


def calculate_risk_metrics(data_dir="data/sample"):
    data_dir = Path(data_dir)

    prices = pd.read_csv(data_dir / "adjusted_prices.csv")
    portfolio = pd.read_csv(data_dir / "daily_portfolio.csv")
    positions = pd.read_csv(data_dir / "daily_positions.csv")

    for df in (prices, portfolio, positions):
        df["date"] = pd.to_datetime(df["date"])

    prices = prices.sort_values("date").set_index("date")
    portfolio = portfolio.sort_values("date").set_index("date")

    if prices.index.has_duplicates or portfolio.index.has_duplicates:
        raise ValueError("Duplicate dates in prices or portfolio data")

    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("Missing or non-positive adjusted prices")

    if not prices.index.equals(portfolio.index):
        raise ValueError("Price and portfolio dates do not match")

    symbols = list(ASSET_GROUPS)

    if set(prices.columns) != set(symbols):
        raise ValueError("Price columns do not match asset universe")

    if positions.duplicated(["date", "instrument_id"]).any():
        raise ValueError("Duplicate daily positions")

    if positions["instrument_id"].isin(symbols).all() is False:
        raise ValueError("Unknown instrument in positions")

    if positions["portfolio_id"].nunique() != 1:
        raise ValueError("Expected exactly one portfolio")

    if portfolio["portfolio_id"].nunique() != 1:
        raise ValueError("Expected exactly one portfolio")

    if positions["portfolio_id"].iloc[0] != portfolio["portfolio_id"].iloc[0]:
        raise ValueError("Portfolio ID mismatch")

    quantities = (
        positions.pivot(
            index="date",
            columns="instrument_id",
            values="quantity",
        )
        .reindex(index=prices.index, columns=symbols)
    )

    if quantities.isna().any().any():
        raise ValueError("Missing daily positions")

    if (quantities < -1e-9).any().any():
        raise ValueError("Negative positions are not supported")

    prices = prices[symbols]
    market_values = quantities * prices
    nav = portfolio["nav_usd"]
    cash = portfolio["cash_usd"]

    calculated_nav = market_values.sum(axis=1) + cash

    if not np.allclose(
        calculated_nav.to_numpy(),
        nav.to_numpy(),
        rtol=1e-9,
        atol=0.01,
    ):
        raise ValueError("NAV reconciliation failed")

    if (nav <= 0).any():
        raise ValueError("Portfolio NAV must be positive")

    price_returns = prices.pct_change(fill_method=None)
    portfolio_returns = nav.pct_change(fill_method=None)

    results = pd.DataFrame(index=prices.index)
    results.index.name = "date"

    results["portfolio_id"] = portfolio["portfolio_id"]
    results["nav_usd"] = nav
    results["cash_usd"] = cash
    results["daily_pnl_usd"] = portfolio["daily_pnl_usd"]
    results["daily_return"] = portfolio_returns

    expected_pnl = nav.diff().fillna(0)

    if not np.allclose(
        expected_pnl.to_numpy(),
        results["daily_pnl_usd"].to_numpy(),
        rtol=1e-9,
        atol=0.01,
    ):
        raise ValueError("Daily P&L reconciliation failed")

    # Historical Simulation: current holdings under past daily returns.
    # Require 252 complete historical return observations.
    var_values = []
    es_values = []

    # Track historical data availability for VaR and ES.
    var_available_observations = []

    for i, date in enumerate(prices.index):
        # Exclude the valuation date's return.
        # The first price observation has no daily return.
        available = min(max(i - 1, 0), VAR_WINDOW)
        var_available_observations.append(available)

        if i <= VAR_WINDOW:
            var_values.append(np.nan)
            es_values.append(np.nan)
            continue

        # Use the previous 252 completed daily returns.
        historical_returns = price_returns.iloc[
            i - VAR_WINDOW : i
        ]

        current_values = market_values.loc[date]

        scenario_pnl = historical_returns.mul(
            current_values, axis=1
        ).sum(axis=1)

        # 99% loss VaR and ES, expressed as positive loss amounts.
        scenario_losses = -scenario_pnl.to_numpy()

        var_99 = float(np.quantile(
            scenario_losses, CONFIDENCE, method="linear"
        ))

        tail_losses = scenario_losses[
            scenario_losses >= var_99
        ]

        var_values.append(max(0.0, var_99))
        es_values.append(max(0.0, float(tail_losses.mean())))

    results["var_99_usd"] = var_values
    results["es_99_usd"] = es_values
    results["var_99_pct_nav"] = results["var_99_usd"] / nav
    results["es_99_pct_nav"] = results["es_99_usd"] / nav

    # Historical data availability for UI and AI investigation.
    results["var_available_observations"] = var_available_observations
    results["var_required_observations"] = VAR_WINDOW

    results["es_available_observations"] = var_available_observations
    results["es_required_observations"] = VAR_WINDOW

    # Rolling realized portfolio volatility.
    results["annualized_volatility"] = (
        portfolio_returns.rolling(
            VOL_WINDOW, min_periods=VOL_WINDOW
        ).std(ddof=1) * np.sqrt(TRADING_DAYS)
    )

    # Since inception drawdown.
    running_peak = nav.cummax()
    results["drawdown"] = nav / running_peak - 1.0

    # Current exposure, as a percentage of NAV.
    for group in ("Equity", "Rates", "Credit"):
        group_symbols = [
            symbol for symbol, category in ASSET_GROUPS.items()
            if category == group
        ]

        exposure = market_values[group_symbols].sum(axis=1)

        results[f"{group.lower()}_exposure_usd"] = exposure
        results[f"{group.lower()}_exposure_pct"] = exposure / nav

    weights = market_values.div(nav, axis=0)
    results["largest_position_weight"] = weights.max(axis=1)
    results["largest_position_symbol"] = weights.idxmax(axis=1)

    # Portfolio beta versus SPY using historical portfolio returns.
    benchmark_returns = price_returns["SPY"]

    rolling_cov = portfolio_returns.rolling(
        BETA_WINDOW, min_periods=BETA_WINDOW
    ).cov(benchmark_returns)

    rolling_var = benchmark_returns.rolling(
        BETA_WINDOW, min_periods=BETA_WINDOW
    ).var(ddof=1)

    results["equity_beta_spy"] = rolling_cov / rolling_var.replace(
        0, np.nan
    )

    return results.reset_index()


def save_risk_metrics(
    data_dir="data/sample",
    output_path="data/outputs/daily_risk_metrics.csv",
):
    metrics = calculate_risk_metrics(data_dir)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(output_path, index=False)
    return metrics
