
import numpy as np
import pandas as pd

from src.tools.common import read_csv, normalize_date
from src.tools.analyze_positions import analyze_positions


def analyze_transactions(
    portfolio_id,
    as_of_date,
    lookback_days=20,
    transactions_path="data/sample/transactions.csv",
    positions_path="data/sample/daily_positions.csv",
    portfolio_path="data/sample/daily_portfolio.csv",
    prices_path="data/sample/adjusted_prices.csv",
):
    """
    Investigate historical transactions and their portfolio impact.

    lookback_days is a trading-day lookback, inclusive of
    the as-of date. A value of 1 selects the as-of date only.
    """
    if isinstance(lookback_days, bool) or not isinstance(
        lookback_days, int
    ) or lookback_days < 1:
        raise ValueError("lookback_days must be a positive integer")

    transactions = read_csv(
        transactions_path,
        required_columns={
            "transaction_id", "portfolio_id", "trade_date",
            "instrument_id", "side", "quantity",
            "execution_price", "fees_usd",
        },
    )

    positions = read_csv(
        positions_path,
        required_columns={
            "date", "portfolio_id", "instrument_id",
            "quantity",
        },
    )

    portfolio = read_csv(
        portfolio_path,
        required_columns={
            "date", "portfolio_id", "nav_usd", "cash_usd",
        },
    )

    transactions["trade_date"] = pd.to_datetime(
        transactions["trade_date"], errors="raise"
    )
    positions["date"] = pd.to_datetime(
        positions["date"], errors="raise"
    )
    portfolio["date"] = pd.to_datetime(
        portfolio["date"], errors="raise"
    )

    as_of = pd.Timestamp(as_of_date)

    portfolio = portfolio.loc[
        portfolio["portfolio_id"] == portfolio_id
    ].sort_values("date").copy()

    dates = portfolio.loc[
        portfolio["date"] <= as_of, "date"
    ].tolist()

    if not dates or dates[-1] != as_of:
        raise ValueError("As-of date not found in portfolio history")

    start_index = max(0, len(dates) - lookback_days)
    window_dates = dates[start_index:]
    start_date = window_dates[0]

    transactions = transactions.loc[
        (transactions["portfolio_id"] == portfolio_id)
        & (transactions["trade_date"] >= start_date)
        & (transactions["trade_date"] <= as_of)
    ].copy()

    if transactions["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs")

    if not transactions["side"].isin(["BUY", "SELL"]).all():
        raise ValueError("Invalid transaction side")

    for column in ("quantity", "execution_price", "fees_usd"):
        transactions[column] = pd.to_numeric(
            transactions[column], errors="raise"
        )

    if not np.isfinite(
        transactions[
            ["quantity", "execution_price", "fees_usd"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite transaction values")

    if (
        (transactions["quantity"] <= 0).any()
        or (transactions["execution_price"] <= 0).any()
        or (transactions["fees_usd"] < 0).any()
    ):
        raise ValueError("Invalid transaction quantities or prices")

    transactions["notional_usd"] = (
        transactions["quantity"]
        * transactions["execution_price"]
    )

    transactions["signed_quantity"] = np.where(
        transactions["side"] == "BUY",
        transactions["quantity"],
        -transactions["quantity"],
    )

    transactions["cash_flow_usd"] = np.where(
        transactions["side"] == "BUY",
        -transactions["notional_usd"],
        transactions["notional_usd"],
    ) - transactions["fees_usd"]

    trade_records = []

    for _, trade in transactions.sort_values(
        ["trade_date", "transaction_id"]
    ).iterrows():
        trade_records.append({
            "transaction_id": str(trade["transaction_id"]),
            "trade_date": normalize_date(trade["trade_date"]),
            "instrument_id": str(trade["instrument_id"]),
            "side": str(trade["side"]),
            "quantity": float(trade["quantity"]),
            "execution_price": float(trade["execution_price"]),
            "notional_usd": float(trade["notional_usd"]),
            "fees_usd": float(trade["fees_usd"]),
            "cash_flow_usd": float(trade["cash_flow_usd"]),
        })

    buy_notional = float(
        transactions.loc[
            transactions["side"] == "BUY", "notional_usd"
        ].sum()
    )

    sell_notional = float(
        transactions.loc[
            transactions["side"] == "SELL", "notional_usd"
        ].sum()
    )

    total_fees = float(transactions["fees_usd"].sum())
    net_cash_flow = float(transactions["cash_flow_usd"].sum())

    # Reconcile trade quantities and cash movements
    # against the recorded previous and current closing books.
    daily_reconciliations = []

    for date in sorted(transactions["trade_date"].unique()):
        date = pd.Timestamp(date)
        date_index = dates.index(date)

        if date_index == 0:
            raise ValueError(
                "Cannot reconcile trades on the first "
                "portfolio observation without opening holdings"
            )

        previous_date = dates[date_index - 1]

        previous = positions.loc[
            (positions["portfolio_id"] == portfolio_id)
            & (positions["date"] == previous_date)
        ]

        current = positions.loc[
            (positions["portfolio_id"] == portfolio_id)
            & (positions["date"] == date)
        ]

        if (
            previous["instrument_id"].duplicated().any()
            or current["instrument_id"].duplicated().any()
        ):
            raise ValueError("Duplicate position records")

        previous_qty = previous.set_index(
            "instrument_id"
        )["quantity"].astype(float)

        current_qty = current.set_index(
            "instrument_id"
        )["quantity"].astype(float)

        day_trades = transactions.loc[
            transactions["trade_date"] == date
        ]

        traded_qty = day_trades.groupby(
            "instrument_id"
        )["signed_quantity"].sum()

        symbols = sorted(
            set(previous_qty.index)
            | set(current_qty.index)
            | set(traded_qty.index)
        )

        quantity_checks = []

        for symbol in symbols:
            prior = float(previous_qty.get(symbol, 0.0))
            actual = float(current_qty.get(symbol, 0.0))
            traded = float(traded_qty.get(symbol, 0.0))
            expected = prior + traded

            difference = actual - expected

            if not np.isclose(
                actual, expected, rtol=1e-9, atol=1e-5
            ):
                raise ValueError(
                    f"Quantity reconciliation failed: "
                    f"{symbol} on {normalize_date(date)}"
                )

            quantity_checks.append({
                "instrument_id": symbol,
                "previous_quantity": prior,
                "traded_quantity": traded,
                "current_quantity": actual,
                "difference": difference,
            })

        prior_cash_rows = portfolio.loc[
            portfolio["date"] == previous_date
        ]
        current_cash_rows = portfolio.loc[
            portfolio["date"] == date
        ]

        if len(prior_cash_rows) != 1 or len(current_cash_rows) != 1:
            raise ValueError("Invalid daily portfolio records")

        prior_cash = float(prior_cash_rows.iloc[0]["cash_usd"])
        current_cash = float(current_cash_rows.iloc[0]["cash_usd"])

        day_cash_flow = float(day_trades["cash_flow_usd"].sum())
        expected_cash = prior_cash + day_cash_flow

        if not np.isclose(
            current_cash, expected_cash, rtol=1e-9, atol=0.01
        ):
            raise ValueError(
                f"Cash reconciliation failed on {normalize_date(date)}"
            )

        position_analysis = analyze_positions(
            portfolio_id,
            normalize_date(date),
            positions_path=positions_path,
            portfolio_path=portfolio_path,
            prices_path=prices_path,
        )

        bridge = position_analysis[
            "analysis"
        ]["daily_weight_bridge"]

        daily_reconciliations.append({
            "trade_date": normalize_date(date),
            "trade_count": len(day_trades),
            "quantity_reconciliation": quantity_checks,
            "previous_cash_usd": prior_cash,
            "transaction_cash_flow_usd": day_cash_flow,
            "expected_cash_usd": expected_cash,
            "actual_cash_usd": current_cash,
            "cash_difference_usd": current_cash - expected_cash,
            "position_weight_effects": {
                symbol: {
                    "market_effect": values["market_effect"],
                    "trade_effect": values["trade_effect"],
                    "total_change": values["total_change"],
                }
                for symbol, values in bridge.items()
            },
            "reconciliation_status": "PASS",
        })

    instrument_summary = []

    if not transactions.empty:
        for symbol, group in transactions.groupby("instrument_id"):
            instrument_summary.append({
                "instrument_id": str(symbol),
                "trade_count": int(len(group)),
                "buy_notional_usd": float(
                    group.loc[
                        group["side"] == "BUY", "notional_usd"
                    ].sum()
                ),
                "sell_notional_usd": float(
                    group.loc[
                        group["side"] == "SELL", "notional_usd"
                    ].sum()
                ),
                "net_quantity_change": float(
                    group["signed_quantity"].sum()
                ),
            })

    return {
        "tool": "analyze_transactions",
        "portfolio_id": portfolio_id,
        "as_of_date": normalize_date(as_of),
        "status": "SUCCESS",
        "analysis_window": {
            "start_date": normalize_date(start_date),
            "end_date": normalize_date(as_of),
            "trading_days": len(window_dates),
        },
        "summary": {
            "trade_count": len(trade_records),
            "buy_notional_usd": buy_notional,
            "sell_notional_usd": sell_notional,
            "gross_notional_usd": buy_notional + sell_notional,
            "net_cash_flow_usd": net_cash_flow,
            "total_fees_usd": total_fees,
            "trade_dates": len(daily_reconciliations),
        },
        "transactions": trade_records,
        "instrument_summary": instrument_summary,
        "daily_reconciliations": daily_reconciliations,
        "methodology": {
            "cash_flow": (
                "SELL notional - BUY notional - transaction fees"
            ),
            "quantity_reconciliation": (
                "Previous closing quantity + signed traded "
                "quantity = current closing quantity"
            ),
            "weight_effects": (
                "Market-first sequential bridge from "
                "analyze_positions"
            ),
        },
        "data_sources": [
            str(transactions_path),
            str(positions_path),
            str(portfolio_path),
            str(prices_path),
        ],
        "limitations": [
            "Transactions use synthetic adjusted-price execution.",
            "Trades are assumed to affect closing holdings.",
            "No external cash flows are modeled.",
            "Trade effects describe position-weight changes, "
            "not realized trading P&L.",
        ],
    }
