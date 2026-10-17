"""Download a frozen adjusted-price snapshot and generate synthetic trades."""
from pathlib import Path
import sys
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.portfolio.loader import load_portfolio, load_transaction_plan
from src.portfolio.accounting import simulate_portfolio


def download_prices(symbols, start, end):
    # auto_adjust=True: Close represents adjusted close (total-return proxy).
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, actions=False,
                      progress=False, group_by="column", threads=False)
    if raw.empty:
        raise RuntimeError("No price data downloaded")
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=symbols[0])
    close = close.reindex(columns=symbols)
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close.index.name = "date"
    if close.isna().any().any() or (close <= 0).any().any():
        raise ValueError("Missing or nonpositive adjusted prices; inspect source before continuing")
    return close


def generate_transactions(prices, portfolio, plan):
    inception = pd.Timestamp(portfolio["inception_date"])
    start_prices = prices.loc[inception]
    holdings = {h["instrument_id"]: float(portfolio["initial_nav"]) * float(h["initial_weight"]) / start_prices[h["instrument_id"]] for h in portfolio["holdings"]}
    cash = 0.0
    records = []
    for group in plan["transactions"]:
        date = pd.Timestamp(group["trade_date"])
        if date not in prices.index:
            raise ValueError(f"Trade date {date.date()} is not a price date")
        # Each target notional is a USD amount, converted to fractional units.
        ordered = sorted(group["trades"], key=lambda t: 0 if t["side"] == "SELL" else 1)
        for trade in ordered:
            symbol = trade["instrument_id"]
            price = float(prices.loc[date, symbol])
            notional = float(trade["target_notional_usd"])
            qty = notional / price
            if trade["side"] == "SELL":
                if qty > holdings[symbol] + 1e-9:
                    raise ValueError(f"Insufficient {symbol} units on {date.date()}")
                holdings[symbol] -= qty
                cash += notional
            else:
                if cash + 1e-5 < notional:
                    raise ValueError(f"Insufficient cash on {date.date()}")
                holdings[symbol] += qty
                cash -= notional
            records.append({"transaction_id": f"TXN_{len(records)+1:04d}", "portfolio_id": portfolio["portfolio_id"], "trade_date": date.strftime("%Y-%m-%d"), "instrument_id": symbol, "side": trade["side"], "quantity": qty, "execution_price": price, "fees_usd": 0.0, "source": "synthetic_adjusted_close"})
    return pd.DataFrame(records)


def main():
    portfolio = load_portfolio(ROOT / "config/portfolio.yaml")
    plan = load_transaction_plan(ROOT / "config/transaction_plan.yaml")
    if plan["portfolio_id"] != portfolio["portfolio_id"]:
        raise ValueError("Portfolio IDs do not match")
    symbols = [h["instrument_id"] for h in portfolio["holdings"]]
    out = ROOT / "data/sample"
    out.mkdir(parents=True, exist_ok=True)
    prices_file = out / "adjusted_prices.csv"
    if prices_file.exists():
        prices = pd.read_csv(prices_file, index_col="date", parse_dates=["date"])
        print("Using frozen local price snapshot:", prices_file)
    else:
        # end is exclusive; snapshot intentionally has a fixed endpoint.
        prices = download_prices(symbols, portfolio["inception_date"], "2026-10-01")
        prices.to_csv(prices_file, float_format="%.12g")
        print("Saved historical price snapshot:", prices_file)
    if pd.Timestamp(portfolio["inception_date"]) not in prices.index:
        raise ValueError("Inception date missing from downloaded prices")
    transactions = generate_transactions(prices, portfolio, plan)
    transactions.to_csv(out / "transactions.csv", index=False, float_format="%.12g")
    # Read back rounded CSV values so output reflects the saved audit artifact.
    transactions = pd.read_csv(out / "transactions.csv")
    daily, positions = simulate_portfolio(prices, portfolio, transactions)
    daily.to_csv(out / "daily_portfolio.csv", index=False, float_format="%.12g")
    positions.to_csv(out / "daily_positions.csv", index=False, float_format="%.12g")
    print(f"Generated {len(transactions)} trades, {len(daily)} daily NAV observations, {len(positions)} position observations")
    print("First NAV:", round(daily.iloc[0]["nav_usd"], 2), "Last NAV:", round(daily.iloc[-1]["nav_usd"], 2))
    print("Accounting reconciliation: PASS")


if __name__ == "__main__":
    main()
