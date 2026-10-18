
import pandas as pd
import pytest

from src.risk.exception_detector import (
    load_risk_policy,
    evaluate_rule,
    detect_exceptions,
)


@pytest.fixture
def policy():
    return load_risk_policy()


def test_max_limit_boundary():
    rule = {
        "direction": "max",
        "warning_threshold": 0.02,
        "breach_threshold": 0.025,
    }

    assert evaluate_rule(0.01, rule) == "PASS"
    assert evaluate_rule(0.02, rule) == "WARNING"
    assert evaluate_rule(0.025, rule) == "WARNING"
    assert evaluate_rule(0.026, rule) == "BREACH"


def test_min_limit_boundary():
    rule = {
        "direction": "min",
        "warning_threshold": -0.10,
        "breach_threshold": -0.15,
    }

    assert evaluate_rule(-0.05, rule) == "PASS"
    assert evaluate_rule(-0.10, rule) == "WARNING"
    assert evaluate_rule(-0.15, rule) == "WARNING"
    assert evaluate_rule(-0.16, rule) == "BREACH"


def test_missing_metric_not_evaluated():
    rule = {
        "direction": "max",
        "warning_threshold": 0.02,
        "breach_threshold": 0.025,
    }

    assert evaluate_rule(float("nan"), rule) == "NOT_EVALUATED"


def test_no_exceptions(policy):
    metrics = pd.DataFrame([{
        "date": "2026-09-30",
        "portfolio_id": "PORT_001",
        "var_99_pct_nav": 0.01,
        "es_99_pct_nav": 0.02,
        "equity_exposure_pct": 0.40,
        "largest_position_weight": 0.20,
        "drawdown": -0.03,
    }])

    result = detect_exceptions(metrics, policy)

    assert (result["status"] == "PASS").all()
    assert (result["status"] == "BREACH").sum() == 0


def test_multiple_breaches(policy):
    metrics = pd.DataFrame([{
        "date": "2026-09-30",
        "portfolio_id": "PORT_001",
        "var_99_pct_nav": 0.01,
        "es_99_pct_nav": 0.02,
        "equity_exposure_pct": 0.70,
        "largest_position_weight": 0.43,
        "drawdown": -0.03,
    }])

    result = detect_exceptions(metrics, policy)

    breaches = result.loc[result["status"] == "BREACH"]

    assert len(breaches) == 2
    assert set(breaches["rule_id"]) == {
        "EQUITY_EXPOSURE_LIMIT",
        "SINGLE_INSTRUMENT_LIMIT",
    }


def test_missing_required_metric_raises(policy):
    metrics = pd.DataFrame([{
        "date": "2026-09-30",
        "portfolio_id": "PORT_001",
    }])

    with pytest.raises(ValueError, match="Missing risk metrics"):
        detect_exceptions(metrics, policy)


def test_real_market_data(policy):
    metrics = pd.read_csv(
        "data/outputs/daily_risk_metrics.csv"
    )

    result = detect_exceptions(metrics, policy)

    assert len(result) == len(metrics) * len(policy["rules"])

    latest = result.loc[
        result["date"] == "2026-09-30"
    ]

    assert len(latest) == 5
    assert (
        latest["status"] == "BREACH"
    ).sum() == 2
