
import numpy as np
import pandas as pd

from src.risk.calculator import ASSET_GROUPS
from src.tools.common import read_csv, normalize_date


def get_pnl_attribution(
    portfolio_id,
    as_of_date,
    lookback_days=1,
    start_date=None,
    positions_path="data/sample/daily_positions.csv",
    portfolio_path="data/sample/daily_portfolio.csv",
    prices_path="data/sample/adjusted_prices.csv",
):
    """
    Calculate deterministic instrument and asset-class P&L.

    lookback_days counts trading days, including as_of_date.
    start_date, when supplied, overrides lookback_days.

    Daily P&L uses previous closing quantities and adjusted
    price changes. Trades at today's close affect future P&L.
    """
    if isinstance(lookback_days, bool) or not isinstance(
        lookback_days, int
    ) or lookback_days < 1:
        raise ValueError("lookback_days must be a positive integer")

    as_of = pd.Timestamp(as_of_date)

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
            "date", "portfolio_id", "nav_usd",
            "daily_pnl_usd", "cash_usd",
        },
    )

    symbols = list(ASSET_GROUPS)

    prices = read_csv(
        prices_path,
        required_columns={"date", *symbols},
    )

    positions["date"] = pd.to_datetime(
        positions["date"], errors="raise"
    )
    portfolio["date"] = pd.to_datetime(
        portfolio["date"], errors="raise"
    )
    prices["date"] = pd.to_datetime(
        prices["date"], errors="raise"
    )

    portfolio = portfolio.loc[
        portfolio["portfolio_id"] == portfolio_id
    ].sort_values("date").reset_index(drop=True)

    positions = positions.loc[
        positions["portfolio_id"] == portfolio_id
    ].copy()

    if portfolio["date"].duplicated().any():
        raise ValueError("Duplicate portfolio dates")

    if positions.duplicated(
        ["date", "instrument_id"]
    ).any():
        raise ValueError("Duplicate position records")

    if prices["date"].duplicated().any():
        raise ValueError("Duplicate price dates")

    prices = prices.set_index("date").sort_index()

    dates = portfolio["date"].tolist()

    if as_of not in dates:
        raise ValueError("As-of date not found")

    end_index = dates.index(as_of)

    if end_index == 0:
        raise ValueError(
            "Cannot calculate P&L on first observation "
            "without opening positions"
        )

    if start_date is not None:
        start = pd.Timestamp(start_date)

        if start > as_of:
            raise ValueError("start_date cannot exceed as_of_date")

        eligible_indices = [
            i for i, date in enumerate(dates)
            if start <= date <= as_of
        ]

        if not eligible_indices:
            raise ValueError("No trading days in requested range")

        start_index = eligible_indices[0]

        if start_index == 0:
            raise ValueError(
                "Selected window includes first observation, "
                "which has no prior closing holdings"
            )
    else:
        start_index = max(1, end_index - lookback_days + 1)

    window_dates = dates[start_index:end_index + 1]

    # Pivot positions into date x instrument quantities.
    quantities = positions.pivot(
        index="date",
        columns="instrument_id",
        values="quantity",
    ).reindex(columns=symbols)

    daily_records = []
    instrument_pnl = {symbol: 0.0 for symbol in symbols}
    total_portfolio_pnl = 0.0

    for date in window_dates:
        index = dates.index(date)
        prior_date = dates[index - 1]

        if date not in prices.index or prior_date not in prices.index:
            raise ValueError(
                f"Missing prices for {normalize_date(date)}"
            )

        if prior_date not in quantities.index:
            raise ValueError(
                f"Missing prior holdings for {normalize_date(date)}"
            )

        prior_qty = quantities.loc[
            prior_date, symbols
        ].astype(float)

        current_prices = prices.loc[
            date, symbols
        ].astype(float)

        prior_prices = prices.loc[
            prior_date, symbols
        ].astype(float)

        if (
            prior_qty.isna().any()
            or current_prices.isna().any()
            or prior_prices.isna().any()
        ):
            raise ValueError(
                f"Incomplete data for {normalize_date(date)}"
            )

        if (
            not np.isfinite(prior_qty).all()
            or not np.isfinite(current_prices).all()
            or not np.isfinite(prior_prices).all()
            or (current_prices <= 0).any()
            or (prior_prices <= 0).any()
        ):
            raise ValueError("Invalid quantities or prices")

        pnl_by_symbol = prior_qty * (
            current_prices - prior_prices
        )

        calculated_pnl = float(pnl_by_symbol.sum())

        row = portfolio.iloc[index]
        reported_pnl = float(row["daily_pnl_usd"])

        if not np.isfinite(reported_pnl):
            raise ValueError("Invalid reported P&L")

        difference = calculated_pnl - reported_pnl

        if not np.isclose(
            calculated_pnl,
            reported_pnl,
            rtol=1e-9,
            atol=0.05,
        ):
            raise ValueError(
                f"P&L reconciliation failed on "
                f"{normalize_date(date)}: "
                f"calculated={calculated_pnl:.6f}, "
                f"reported={reported_pnl:.6f}"
            )

        total_portfolio_pnl += calculated_pnl

        contributions = []

        for symbol in symbols:
            value = float(pnl_by_symbol[symbol])
            instrument_pnl[symbol] += value

            contributions.append({
                "instrument_id": symbol,
                "asset_class": ASSET_GROUPS[symbol],
                "pnl_usd": value,
            })

        contributions.sort(
            key=lambda item: item["pnl_usd"]
        )

        daily_records.append({
            "date": normalize_date(date),
            "calculated_pnl_usd": calculated_pnl,
            "reported_pnl_usd": reported_pnl,
            "reconciliation_difference_usd": difference,
            "instrument_contributions": contributions,
            "reconciliation_status": "PASS",
        })

    beginning_nav = float(
        portfolio.iloc[start_index - 1]["nav_usd"]
    )

    ending_nav = float(
        portfolio.iloc[end_index]["nav_usd"]
    )

    if not np.isfinite(beginning_nav) or beginning_nav <= 0:
        raise ValueError("Invalid beginning NAV")

    if not np.isfinite(ending_nav) or ending_nav <= 0:
        raise ValueError("Invalid ending NAV")

    instrument_attribution = []

    for symbol, pnl in instrument_pnl.items():
        instrument_attribution.append({
            "instrument_id": symbol,
            "asset_class": ASSET_GROUPS[symbol],
            "pnl_usd": float(pnl),
            "contribution_bps": float(
                pnl / beginning_nav * 10000
            ),
        })

    instrument_attribution.sort(
        key=lambda item: item["pnl_usd"]
    )

    asset_class_attribution = []

    for group in ("Equity", "Rates", "Credit"):
        group_pnl = sum(
            item["pnl_usd"]
            for item in instrument_attribution
            if item["asset_class"] == group
        )

        asset_class_attribution.append({
            "asset_class": group,
            "pnl_usd": float(group_pnl),
            "contribution_bps": float(
                group_pnl / beginning_nav * 10000
            ),
        })

    reported_cumulative_pnl = float(
        portfolio.iloc[start_index:end_index + 1][
            "daily_pnl_usd"
        ].sum()
    )

    nav_change = ending_nav - beginning_nav

    if not np.isclose(
        total_portfolio_pnl,
        reported_cumulative_pnl,
        rtol=1e-9,
        atol=0.05,
    ):
        raise ValueError("Cumulative P&L reconciliation failed")

    if not np.isclose(
        total_portfolio_pnl,
        nav_change,
        rtol=1e-9,
        atol=0.05,
    ):
        raise ValueError("P&L versus NAV change reconciliation failed")

    return {
        "tool": "get_pnl_attribution",
        "portfolio_id": portfolio_id,
        "as_of_date": normalize_date(as_of),
        "status": "SUCCESS",
        "analysis_window": {
            "start_date": normalize_date(window_dates[0]),
            "end_date": normalize_date(as_of),
            "trading_days": len(window_dates),
            "beginning_nav_usd": beginning_nav,
            "ending_nav_usd": ending_nav,
        },
        "summary": {
            "portfolio_pnl_usd": total_portfolio_pnl,
            "portfolio_return": (
                total_portfolio_pnl / beginning_nav
            ),
            "portfolio_return_bps": (
                total_portfolio_pnl / beginning_nav * 10000
            ),
            "reported_cumulative_pnl_usd": (
                reported_cumulative_pnl
            ),
            "nav_change_usd": nav_change,
            "reconciliation_difference_usd": (
                total_portfolio_pnl - reported_cumulative_pnl
            ),
            "reconciliation_status": "PASS",
        },
        "instrument_attribution": instrument_attribution,
        "asset_class_attribution": asset_class_attribution,
        "largest_loss_contributors": [
            item for item in instrument_attribution
            if item["pnl_usd"] < 0
        ][:3],
        "largest_gain_contributors": sorted(
            [
                item for item in instrument_attribution
                if item["pnl_usd"] > 0
            ],
            key=lambda item: item["pnl_usd"],
            reverse=True,
        )[:3],
        "daily_attribution": daily_records,
        "methodology": {
            "daily_holding_pnl": (
                "Previous closing quantity multiplied by "
                "current minus previous adjusted closing price"
            ),
            "multi_day_pnl": (
                "Sum of daily instrument P&L using the "
                "holdings applicable to each trading day"
            ),
            "contribution_bps": (
                "Instrument P&L / beginning NAV * 10000"
            ),
            "portfolio_return": (
                "Cumulative P&L / beginning NAV; "
                "assumes no external cash flows"
            ),
        },
        "data_sources": [
            str(positions_path),
            str(portfolio_path),
            str(prices_path),
        ],
        "limitations": [
            "Adjusted prices represent synthetic total-return units.",
            "Same-day trades occur at closing prices and "
            "affect subsequent holding P&L.",
            "No intraday execution or realized trading P&L "
            "is calculated separately.",
            "The model assumes zero fees and no external cash flows.",
            "Asset-class attribution uses ETF exposure categories, "
            "not factor sensitivities such as DV01 or CS01.",
        ],
    }
