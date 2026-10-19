
import numpy as np
import pandas as pd

from src.risk.calculator import ASSET_GROUPS
from src.tools.common import (
    read_csv,
    select_snapshot,
    normalize_date,
)


def analyze_positions(
    portfolio_id,
    as_of_date,
    positions_path="data/sample/daily_positions.csv",
    portfolio_path="data/sample/daily_portfolio.csv",
    prices_path="data/sample/adjusted_prices.csv",
):
    positions = read_csv(
        positions_path,
        required_columns={
            "date", "portfolio_id", "instrument_id",
            "quantity", "market_value_usd",
        },
    )

    portfolio = read_csv(
        portfolio_path,
        required_columns={
            "date", "portfolio_id", "nav_usd", "cash_usd",
        },
    )

    prices = read_csv(
        prices_path,
        required_columns={"date", *ASSET_GROUPS.keys()},
    )

    positions["date"] = pd.to_datetime(positions["date"])
    portfolio["date"] = pd.to_datetime(portfolio["date"])
    prices["date"] = pd.to_datetime(prices["date"])

    positions = positions.loc[
        positions["portfolio_id"] == portfolio_id
    ].copy()

    portfolio = portfolio.loc[
        portfolio["portfolio_id"] == portfolio_id
    ].sort_values("date").reset_index(drop=True)

    prices = prices.sort_values("date").set_index("date")

    as_of = pd.Timestamp(as_of_date)

    current_portfolio = select_snapshot(
        portfolio, portfolio_id, as_of
    )

    current_positions = positions.loc[
        positions["date"] == as_of
    ].copy()

    if current_positions.empty:
        raise ValueError("No positions for requested date")

    if current_positions["instrument_id"].duplicated().any():
        raise ValueError("Duplicate instruments in positions")

    symbols = list(ASSET_GROUPS)

    if set(current_positions["instrument_id"]) != set(symbols):
        raise ValueError("Incomplete position universe")

    if as_of not in prices.index:
        raise ValueError("Missing current market prices")

    current_positions = current_positions.set_index(
        "instrument_id"
    ).reindex(symbols)

    nav = float(current_portfolio["nav_usd"])
    cash = float(current_portfolio["cash_usd"])

    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("Invalid NAV")

    current_values = current_positions[
        "market_value_usd"
    ].astype(float)

    current_quantities = current_positions[
        "quantity"
    ].astype(float)

    current_prices = prices.loc[as_of, symbols].astype(float)

    if not np.isfinite(current_values).all():
        raise ValueError("Non-finite market values")

    if not np.isfinite(current_quantities).all():
        raise ValueError("Non-finite quantities")

    if not np.isfinite(current_prices).all() or (
        current_prices <= 0
    ).any():
        raise ValueError("Invalid market prices")

    if not np.allclose(
        current_quantities * current_prices,
        current_values,
        rtol=1e-9,
        atol=0.01,
    ):
        raise ValueError("Position market value reconciliation failed")

    if not np.isclose(
        current_values.sum() + cash,
        nav,
        rtol=1e-9,
        atol=0.01,
    ):
        raise ValueError("Portfolio NAV reconciliation failed")

    weights = current_values / nav

    holdings = [
        {
            "instrument_id": symbol,
            "asset_class": ASSET_GROUPS[symbol],
            "quantity": float(current_quantities[symbol]),
            "market_value_usd": float(current_values[symbol]),
            "weight": float(weights[symbol]),
        }
        for symbol in symbols
    ]

    holdings.sort(
        key=lambda item: item["market_value_usd"],
        reverse=True,
    )

    asset_class_exposures = []

    for group in ("Equity", "Rates", "Credit"):
        group_symbols = [
            symbol for symbol in symbols
            if ASSET_GROUPS[symbol] == group
        ]

        value = float(current_values[group_symbols].sum())

        asset_class_exposures.append({
            "asset_class": group,
            "market_value_usd": value,
            "weight": value / nav,
        })

    concentration = {
        "largest_position": holdings[0]["instrument_id"],
        "largest_weight": holdings[0]["weight"],
        "top_two_weight": sum(
            item["weight"] for item in holdings[:2]
        ),
        "hhi": float(np.square(weights).sum()),
    }

    # Historical weight changes.
    historical_changes = {}

    available_dates = portfolio.loc[
        portfolio["date"] <= as_of, "date"
    ].tolist()

    if not available_dates or available_dates[-1] != as_of:
        raise ValueError("As-of date not in portfolio history")

    current_index = len(available_dates) - 1

    for lookback in (5, 20):
        if current_index < lookback:
            historical_changes[f"{lookback}d"] = None
            continue

        prior_date = available_dates[current_index - lookback]

        prior_portfolio = select_snapshot(
            portfolio, portfolio_id, prior_date
        )

        prior_positions = positions.loc[
            positions["date"] == prior_date
        ].set_index("instrument_id").reindex(symbols)

        if prior_positions["market_value_usd"].isna().any():
            raise ValueError("Incomplete historical positions")

        prior_nav = float(prior_portfolio["nav_usd"])

        if prior_nav <= 0:
            raise ValueError("Invalid historical NAV")

        prior_weights = (
            prior_positions["market_value_usd"].astype(float)
            / prior_nav
        )

        historical_changes[f"{lookback}d"] = {
            symbol: float(
                weights[symbol] - prior_weights[symbol]
            )
            for symbol in symbols
        }

    # Previous-day market/trade decomposition.
    daily_weight_bridge = None

    if current_index >= 1:
        prior_date = available_dates[current_index - 1]

        prior_portfolio = select_snapshot(
            portfolio, portfolio_id, prior_date
        )

        prior_positions = positions.loc[
            positions["date"] == prior_date
        ].set_index("instrument_id").reindex(symbols)

        if prior_positions["quantity"].isna().any():
            raise ValueError("Incomplete previous-day positions")

        prior_quantities = prior_positions[
            "quantity"
        ].astype(float)

        prior_nav = float(prior_portfolio["nav_usd"])
        prior_cash = float(prior_portfolio["cash_usd"])

        if prior_date not in prices.index:
            raise ValueError("Missing previous-day prices")

        prior_prices = prices.loc[
            prior_date, symbols
        ].astype(float)

        prior_values = prior_quantities * prior_prices
        prior_weights = prior_values / prior_nav

        # Revalue previous closing holdings at today's prices.
        # No current-day trades have been applied yet.
        pre_trade_values = prior_quantities * current_prices
        pre_trade_nav = float(
            pre_trade_values.sum() + prior_cash
        )

        if pre_trade_nav <= 0:
            raise ValueError("Invalid pre-trade NAV")

        pre_trade_weights = pre_trade_values / pre_trade_nav

        market_effect = pre_trade_weights - prior_weights
        trade_effect = weights - pre_trade_weights

        # Exact additive weight bridge.
        total_change = weights - prior_weights

        if not np.allclose(
            market_effect + trade_effect,
            total_change,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ValueError("Weight bridge reconciliation failed")

        daily_weight_bridge = {
            symbol: {
                "previous_weight": float(prior_weights[symbol]),
                "pre_trade_weight": float(
                    pre_trade_weights[symbol]
                ),
                "current_weight": float(weights[symbol]),
                "market_effect": float(market_effect[symbol]),
                "trade_effect": float(trade_effect[symbol]),
                "total_change": float(total_change[symbol]),
            }
            for symbol in symbols
        }

    return {
        "tool": "analyze_positions",
        "portfolio_id": portfolio_id,
        "as_of_date": normalize_date(as_of),
        "status": "SUCCESS",
        "nav_usd": nav,
        "largest_position": holdings[0],
        "positions": holdings,
        "analysis": {
            "asset_class_exposures": asset_class_exposures,
            "concentration": concentration,
            "historical_weight_changes": historical_changes,
            "daily_weight_bridge": daily_weight_bridge,
        },
        "methodology": {
            "weight": "market_value / portfolio_nav",
            "hhi": "sum of squared instrument weights",
            "daily_bridge": (
                "Previous closing weights -> "
                "current-price revaluation of prior holdings -> "
                "current closing weights"
            ),
            "change_units": (
                "Decimal weight changes; multiply by 100 "
                "for percentage points"
            ),
        },
        "data_sources": [
            str(positions_path),
            str(portfolio_path),
            str(prices_path),
        ],
        "limitations": [
            "ETF categories are exposure proxies, not DV01 or CS01.",
            "The market/trade bridge uses a sequential "
            "market-first convention.",
            "Adjusted prices represent synthetic total-return "
            "units rather than executable market prices.",
        ],
    }
