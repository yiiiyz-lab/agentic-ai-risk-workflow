
"""Tests for P&L exception monitoring integration."""

import pandas as pd

from src.risk.exception_detector import (
    classify_unavailability,
    detect_exceptions,
    evaluate_rule,
)


PNL_RULE = {
    "rule_id": "PNL_VAR_EXCEEDANCE",
    "metric": "pnl_loss_to_prior_var",
    "direction": "max",
    "warning_threshold": 0.80,
    "breach_threshold": 1.00,
    "severity": "HIGH",
}


def test_pnl_threshold_boundaries():
    assert evaluate_rule(0.79, PNL_RULE) == "PASS"
    assert evaluate_rule(0.80, PNL_RULE) == "WARNING"
    assert evaluate_rule(1.00, PNL_RULE) == "WARNING"
    assert evaluate_rule(1.01, PNL_RULE) == "BREACH"


def test_missing_prior_var_history():
    observation = {
        "prior_var_available_observations": 251,
        "var_required_observations": 252,
    }

    assert classify_unavailability(
        observation,
        "pnl_loss_to_prior_var",
    ) == "INSUFFICIENT_HISTORY"


def test_missing_pnl_with_sufficient_history():
    observation = {
        "prior_var_available_observations": 252,
        "var_required_observations": 252,
    }

    assert classify_unavailability(
        observation,
        "pnl_loss_to_prior_var",
    ) == "UNKNOWN"


def test_pnl_exception_enters_existing_detector():
    metrics = pd.DataFrame([
        {
            "date": "2026-09-29",
            "portfolio_id": "PORT_001",
            "pnl_loss_to_prior_var": 0.50,
            "prior_var_available_observations": 252,
            "var_required_observations": 252,
        },
        {
            "date": "2026-09-30",
            "portfolio_id": "PORT_001",
            "pnl_loss_to_prior_var": 1.25,
            "prior_var_available_observations": 252,
            "var_required_observations": 252,
        },
    ])

    policy = {
        "portfolio_id": "PORT_001",
        "policy_version": "test",
        "rules": [PNL_RULE],
    }

    evaluations = detect_exceptions(metrics, policy)

    assert evaluations["status"].tolist() == [
        "PASS",
        "BREACH",
    ]

    breach = evaluations.iloc[1]

    assert breach["exception_id"] == (
        "2026-09-30_PNL_VAR_EXCEEDANCE"
    )
    assert breach["metric"] == "pnl_loss_to_prior_var"
    assert breach["metric_value"] == 1.25
    assert breach["available_observations"] == 252
    assert breach["required_observations"] == 252
