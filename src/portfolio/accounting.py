"""Deterministic daily accounting for synthetic total-return units."""
import math
import pandas as pd

TOL = 1e-5


def simulate_portfolio(prices: pd.DataFrame, portfolio: dict, transactions: pd.DataFrame):
    """Return (daily_summary, daily_positions), with close-of-day execution.

    Prices are adjusted-close levels, not real tradable prices. All dates must
    be trading dates and price rows must be complete and strictly positive.
    """
    symbols = [h["instrument_id"] for h in portfolio["holdings"]]
    prices = prices.copy().sort_index()
    prices.index = pd.to_datetime(prices.index).normalize()
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Invalid or duplicate price dates")
    if not set(symbols).issubset(prices.columns):
        raise ValueError(f"Missing price symbols: {set(symbols) - set(prices.columns)}")
    prices = prices[symbols].astype(float)
    if prices.isna().any().any() or not (prices > 0).all().all():
        raise ValueError("Missing, nonpositive or invalid prices")
    inception = pd.Timestamp(portfolio["inception_date"])
    if inception not in prices.index:
        raise ValueError(f"Inception date {inception.date()} not found in prices")
    prices = prices.loc[inception:]
    initial_nav = float(portfolio["initial_nav"])
    first = prices.iloc[0]
    quantities = {h["instrument_id"]: initial_nav * float(h["initial_weight"]) / first[h["instrument_id"]] for h in portfolio["holdings"]}
    cash = 0.0

    tx = transactions.copy()
    if len(tx):
        tx["trade_date"] = pd.to_datetime(tx["trade_date"]).dt.normalize()
        if tx["transaction_id"].duplicated().any():
            raise ValueError("Duplicate transaction IDs")
        if not set(tx["instrument_id"]).issubset(symbols):
            raise ValueError("Transaction references unknown instrument")
        if not set(tx["trade_date"]).issubset(set(prices.index)):
            raise ValueError("Transaction date missing from price history")
        if (tx["trade_date"] < inception).any():
            raise ValueError("Transaction before inception")
    else:
        tx = pd.DataFrame(columns=["trade_date", "side", "instrument_id", "quantity", "execution_price", "fees_usd"])

    summary, positions = [], []
    previous_nav = None
    previous_prices = None
    for date, row in prices.iterrows():
        # P&L accrues on yesterday's closing holdings, before today's trades.
        pnl = 0.0 if previous_prices is None else sum(quantities[s] * (row[s] - previous_prices[s]) for s in symbols)
        pretrade_nav = cash + sum(quantities[s] * row[s] for s in symbols)
        if previous_nav is not None and not math.isclose(pretrade_nav - previous_nav, pnl, abs_tol=TOL, rel_tol=1e-12):
            raise ValueError(f"P&L reconciliation failed on {date.date()}")
        todays = tx.loc[tx["trade_date"] == date]
        # Sell first, then buy, so self-financed reallocations can use proceeds.
        todays = todays.assign(_order=todays["side"].map({"SELL": 0, "BUY": 1})).sort_values("_order")
        for _, trade in todays.iterrows():
            symbol = trade["instrument_id"]
            qty, execution_price = float(trade["quantity"]), float(trade["execution_price"])
            fee = float(trade["fees_usd"])
            if qty <= 0 or fee != 0 or not math.isclose(execution_price, row[symbol], rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError(f"Invalid synthetic execution on {date.date()} for {symbol}")
            notional = qty * execution_price
            if trade["side"] == "SELL":
                if qty > quantities[symbol] + 1e-9:
                    raise ValueError(f"Insufficient holdings to sell {symbol} on {date.date()}")
                quantities[symbol] -= qty
                cash += notional
            elif trade["side"] == "BUY":
                if notional > cash + TOL:
                    raise ValueError(
                        f"Insufficient cash to buy {symbol} on {date.date()}: "
                        f"available=${cash:,.8f}, "
                        f"required=${notional:,.8f}, "
                        f"shortfall=${notional - cash:,.8f}"
                    )
                quantities[symbol] += qty
                cash -= notional
            else:
                raise ValueError("Unknown transaction side")
        if abs(cash) < TOL:
            cash = 0.0
        if cash < -TOL or any(q < -1e-9 for q in quantities.values()):
            raise ValueError(f"Negative cash/holdings on {date.date()}")
        nav = cash + sum(quantities[s] * row[s] for s in symbols)
        if not math.isclose(nav, pretrade_nav, abs_tol=TOL, rel_tol=1e-12):
            raise ValueError(f"Trade NAV reconciliation failed on {date.date()}")
        summary.append({"date": date, "portfolio_id": portfolio["portfolio_id"], "nav_usd": nav, "daily_pnl_usd": pnl, "cash_usd": cash, "trade_count": len(todays)})
        for symbol in symbols:
            mv = quantities[symbol] * row[symbol]
            positions.append({"date": date, "portfolio_id": portfolio["portfolio_id"], "instrument_id": symbol, "quantity": quantities[symbol], "adjusted_price": row[symbol], "market_value_usd": mv, "weight": mv / nav})
        previous_nav, previous_prices = nav, row
    return pd.DataFrame(summary), pd.DataFrame(positions)
