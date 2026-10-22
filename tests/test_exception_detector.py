
import numpy as np
import pandas as pd
import pytest

from src.risk.calculator import calculate_risk_metrics
from src.risk.exception_detector import (
    load_risk_policy,
    evaluate_rule,
    classify_unavailability,
    detect_exceptions,
)


@pytest.fixture
def policy():
    return load_risk_policy()


@pytest.fixture
def sample_metrics():
    """
    Sample observation containing all six monitored metrics.

    All metrics are initially within their monitoring limits.
    """
    return pd.DataFrame([
        {
            "date": "2026-09-30",
            "portfolio_id": "PORT_001",
            "var_99_pct_nav": 0.01,
            "es_99_pct_nav": 0.02,
            "equity_exposure_pct": 0.40,
            "largest_position_weight": 0.20,
            "drawdown": -0.03,
            "pnl_loss_to_prior_var": 0.50,
            "var_available_observations": 252,
            "var_required_observations": 252,
            "es_available_observations": 252,
            "es_required_observations": 252,
            "prior_var_available_observations": 252,
        }
    ])


def test_policy_loads(policy):
    assert policy["portfolio_id"] == "PORT_001"
    assert policy["policy_version"] == "1.1"
    assert len(policy["rules"]) == 6


def test_policy_rule_ids_are_unique(policy):
    rule_ids = [
        rule["rule_id"]
        for rule in policy["rules"]
    ]

    assert len(rule_ids) == len(set(rule_ids))


def test_pnl_rule_exists(policy):
    rules = {
        rule["rule_id"]: rule
        for rule in policy["rules"]
    }

    assert "PNL_VAR_EXCEEDANCE" in rules

    pnl_rule = rules["PNL_VAR_EXCEEDANCE"]

    assert pnl_rule["metric"] == "pnl_loss_to_prior_var"
    assert pnl_rule["direction"] == "max"
    assert pnl_rule["warning_threshold"] == 0.80
    assert pnl_rule["breach_threshold"] == 1.00
    assert pnl_rule["severity"] == "HIGH"


def test_max_rule_thresholds():
    rule = {
        "direction": "max",
        "warning_threshold": 0.60,
        "breach_threshold": 0.65,
    }

    assert evaluate_rule(0.50, rule) == "PASS"
    assert evaluate_rule(0.60, rule) == "WARNING"
    assert evaluate_rule(0.65, rule) == "WARNING"
    assert evaluate_rule(0.66, rule) == "BREACH"


def test_min_rule_thresholds():
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
        "warning_threshold": 0.80,
        "breach_threshold": 1.00,
    }

    assert evaluate_rule(np.nan, rule) == "NOT_EVALUATED"
    assert evaluate_rule(np.inf, rule) == "NOT_EVALUATED"
    assert evaluate_rule(-np.inf, rule) == "NOT_EVALUATED"


def test_no_exceptions(policy, sample_metrics):
    evaluations = detect_exceptions(
        sample_metrics,
        policy,
    )

    assert len(evaluations) == len(policy["rules"])
    assert set(evaluations["status"]) == {"PASS"}


def test_multiple_breaches(policy, sample_metrics):
    metrics = sample_metrics.copy()

    metrics.loc[0, "equity_exposure_pct"] = 0.70
    metrics.loc[0, "largest_position_weight"] = 0.43

    evaluations = detect_exceptions(metrics, policy)

    breaches = evaluations.loc[
        evaluations["status"] == "BREACH"
    ]

    assert set(breaches["rule_id"]) == {
        "EQUITY_EXPOSURE_LIMIT",
        "SINGLE_INSTRUMENT_LIMIT",
    }

    assert len(breaches) == 2


def test_pnl_threshold_boundaries(policy):
    pnl_rule = next(
        rule
        for rule in policy["rules"]
        if rule["rule_id"] == "PNL_VAR_EXCEEDANCE"
    )

    assert evaluate_rule(0.79, pnl_rule) == "PASS"
    assert evaluate_rule(0.80, pnl_rule) == "WARNING"
    assert evaluate_rule(1.00, pnl_rule) == "WARNING"
    assert evaluate_rule(1.01, pnl_rule) == "BREACH"


def test_pnl_breach_detected(policy, sample_metrics):
    metrics = sample_metrics.copy()

    metrics.loc[0, "pnl_loss_to_prior_var"] = 1.25

    evaluations = detect_exceptions(metrics, policy)

    pnl = evaluations.loc[
        evaluations["rule_id"] == "PNL_VAR_EXCEEDANCE"
    ].iloc[0]

    assert pnl["status"] == "BREACH"
    assert pnl["metric_value"] == pytest.approx(1.25)
    assert pnl["severity"] == "HIGH"

    assert pnl["exception_id"] == (
        "2026-09-30_PNL_VAR_EXCEEDANCE"
    )

    assert pnl["available_observations"] == 252
    assert pnl["required_observations"] == 252


def test_pnl_warning_detected(policy, sample_metrics):
    metrics = sample_metrics.copy()

    metrics.loc[0, "pnl_loss_to_prior_var"] = 0.90

    evaluations = detect_exceptions(metrics, policy)

    pnl = evaluations.loc[
        evaluations["rule_id"] == "PNL_VAR_EXCEEDANCE"
    ].iloc[0]

    assert pnl["status"] == "WARNING"
    assert pnl["metric_value"] == pytest.approx(0.90)


def test_pnl_missing_prior_var_history(
    policy,
    sample_metrics,
):
    metrics = sample_metrics.copy()

    metrics.loc[0, "pnl_loss_to_prior_var"] = np.nan
    metrics.loc[
        0, "prior_var_available_observations"
    ] = 251

    evaluations = detect_exceptions(metrics, policy)

    pnl = evaluations.loc[
        evaluations["rule_id"] == "PNL_VAR_EXCEEDANCE"
    ].iloc[0]

    assert pnl["status"] == "NOT_EVALUATED"
    assert pnl["reason_code"] == "INSUFFICIENT_HISTORY"
    assert pnl["available_observations"] == 251
    assert pnl["required_observations"] == 252


def test_pnl_missing_with_sufficient_history(
    policy,
    sample_metrics,
):
    metrics = sample_metrics.copy()

    metrics.loc[0, "pnl_loss_to_prior_var"] = np.nan

    evaluations = detect_exceptions(metrics, policy)

    pnl = evaluations.loc[
        evaluations["rule_id"] == "PNL_VAR_EXCEEDANCE"
    ].iloc[0]

    assert pnl["status"] == "NOT_EVALUATED"
    assert pnl["reason_code"] == "UNKNOWN"


def test_var_warmup_classification():
    observation = {
        "var_available_observations": 100,
        "var_required_observations": 252,
    }

    assert classify_unavailability(
        observation,
        "var_99_pct_nav",
    ) == "INSUFFICIENT_HISTORY"


def test_pnl_warmup_classification():
    observation = {
        "prior_var_available_observations": 251,
        "var_required_observations": 252,
    }

    assert classify_unavailability(
        observation,
        "pnl_loss_to_prior_var",
    ) == "INSUFFICIENT_HISTORY"


def test_duplicate_portfolio_date_rejected(
    policy,
    sample_metrics,
):
    metrics = pd.concat(
        [sample_metrics, sample_metrics],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate portfolio/date",
    ):
        detect_exceptions(metrics, policy)


def test_missing_required_metric_rejected(
    policy,
    sample_metrics,
):
    metrics = sample_metrics.drop(
        columns=["pnl_loss_to_prior_var"]
    )

    with pytest.raises(
        ValueError,
        match="Missing risk metrics",
    ):
        detect_exceptions(metrics, policy)


def test_portfolio_mismatch_rejected(
    policy,
    sample_metrics,
):
    metrics = sample_metrics.copy()
    metrics["portfolio_id"] = "OTHER_PORTFOLIO"

    with pytest.raises(
        ValueError,
        match="Portfolio ID does not match",
    ):
        detect_exceptions(metrics, policy)


def test_real_market_data(policy):
    """
    Validate the full monitoring framework against calculated
    portfolio risk metrics, including P&L monitoring.

    Calculate metrics directly to avoid depending on an
    outdated CSV file from a previous pipeline run.
    """
    metrics = calculate_risk_metrics()

    evaluations = detect_exceptions(metrics, policy)

    assert len(evaluations) == (
        len(metrics) * len(policy["rules"])
    )

    assert set(evaluations["rule_id"]) == {
        rule["rule_id"]
        for rule in policy["rules"]
    }

    assert set(evaluations["status"]).issubset({
        "PASS",
        "WARNING",
        "BREACH",
        "NOT_EVALUATED",
    })

    pnl = evaluations.loc[
        evaluations["rule_id"] == "PNL_VAR_EXCEEDANCE"
    ]

    assert len(pnl) == len(metrics)

    # P&L monitoring must be unavailable before prior-day
    # VaR has sufficient historical observations.
    assert (
        pnl.iloc[0]["status"] == "NOT_EVALUATED"
    )

    # The final observation has sufficient historical data.
    assert (
        pnl.iloc[-1]["status"] != "NOT_EVALUATED"
    )


def test_all_rules_evaluated_each_day(policy):
    metrics = calculate_risk_metrics()

    evaluations = detect_exceptions(
        metrics,
        policy,
    )

    daily_counts = evaluations.groupby(
        ["date", "portfolio_id"]
    )["rule_id"].nunique()

    assert (
        daily_counts == len(policy["rules"])
    ).all()
