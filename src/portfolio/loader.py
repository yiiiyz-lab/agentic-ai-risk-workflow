"""Read and validate portfolio and synthetic transaction configurations."""
from pathlib import Path
import yaml


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def load_portfolio(path="config/portfolio.yaml"):
    cfg = read_yaml(path)["portfolio"]
    holdings = cfg["holdings"]
    symbols = [h["instrument_id"] for h in holdings]
    if len(symbols) != len(set(symbols)):
        raise ValueError("Duplicate portfolio instruments")
    if abs(sum(float(h["initial_weight"]) for h in holdings) - 1) > 1e-9:
        raise ValueError("Initial weights must sum to 1")
    if any(float(h["initial_weight"]) < 0 for h in holdings):
        raise ValueError("Negative initial weights not supported")
    if float(cfg["initial_nav"]) <= 0:
        raise ValueError("Initial NAV must be positive")
    if cfg.get("position_management", {}).get("strategy") != "transaction_driven":
        raise ValueError("Expected transaction_driven position management")
    return cfg


def load_transaction_plan(path="config/transaction_plan.yaml"):
    plan = read_yaml(path)
    model = plan["execution_model"]
    if model["price_basis"] != "adjusted_close" or model["quantity_type"] != "synthetic_total_return_units":
        raise ValueError("This implementation requires synthetic units and adjusted close")
    if model.get("execution_timing") != "market_close":
        raise ValueError("Only market_close execution is supported")
    if float(model.get("fees_usd", 0)) != 0:
        raise ValueError("This MVP assumes zero transaction fees")
    dates = [str(g["trade_date"]) for g in plan["transactions"]]
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise ValueError("Transaction dates must be unique and sorted")
    for group in plan["transactions"]:
        trades = group["trades"]
        sell = sum(float(t["target_notional_usd"]) for t in trades if t["side"] == "SELL")
        buy = sum(float(t["target_notional_usd"]) for t in trades if t["side"] == "BUY")
        if abs(sell - buy) > 0.01:
            raise ValueError(f"Trade notionals do not balance on {group['trade_date']}")
        if any(t["side"] not in ("BUY", "SELL") or float(t["target_notional_usd"]) <= 0 for t in trades):
            raise ValueError("Invalid trade side or notional")
    return plan
