import pandas as pd
import pytest
from src.portfolio.accounting import simulate_portfolio


def test_close_trade_preserves_nav_and_next_day_pnl():
    prices = pd.DataFrame({"AAA": [100., 110., 120.], "BBB": [100., 100., 90.]}, index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
    portfolio = {"portfolio_id": "TEST", "inception_date": "2020-01-02", "initial_nav": 1000, "holdings": [{"instrument_id": "AAA", "initial_weight": .5}, {"instrument_id": "BBB", "initial_weight": .5}]}
    tx = pd.DataFrame([{"transaction_id": "T1", "trade_date": "2020-01-03", "instrument_id": "AAA", "side": "SELL", "quantity": 1., "execution_price": 110., "fees_usd": 0}, {"transaction_id": "T2", "trade_date": "2020-01-03", "instrument_id": "BBB", "side": "BUY", "quantity": 1.1, "execution_price": 100., "fees_usd": 0}])
    daily, positions = simulate_portfolio(prices, portfolio, tx)
    assert daily["nav_usd"].tolist() == pytest.approx([1000, 1050, 1029])
    assert daily["daily_pnl_usd"].tolist() == pytest.approx([0, 50, -21])
    assert daily["cash_usd"].tolist() == pytest.approx([0, 0, 0])
    assert positions.loc[(positions["date"] == pd.Timestamp("2020-01-03")) & (positions["instrument_id"] == "AAA"), "quantity"].iloc[0] == pytest.approx(4)
