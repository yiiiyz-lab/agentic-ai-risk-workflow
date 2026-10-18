
import numpy as np
import pandas as pd

from src.risk.calculator import calculate_risk_metrics
from src.risk.exception_detector import (
    classify_unavailability,
    detect_exceptions,
    load_risk_policy,
    save_exception_results,
)


def test_var_es_warmup_observations():
    """VaR/ES require 252 prior completed returns."""
    metrics = calculate_risk_metrics()

    assert metrics.iloc[0]["var_available_observations"] == 0
    assert metrics.iloc[120]["var_available_observations"] == 119
    assert metrics.iloc[252]["var_available_observations"] == 251
    assert metrics.iloc[253]["var_available_observations"] == 252

    assert metrics.iloc[252]["es_available_observations"] == 251
    assert metrics.iloc[253]["es_required_observations"] == 252

    assert pd.isna(metrics.iloc[252]["var_99_usd"])
    assert pd.isna(metrics.iloc[252]["es_99_usd"])

    assert pd.notna(metrics.iloc[253]["var_99_usd"])
    assert pd.notna(metrics.iloc[253]["es_99_usd"])


def test_insufficient_history_classification():
    """Expected warm-up must have a known reason code."""
    observation = pd.Series({
        "var_available_observations": 120,
        "var_required_observations": 252,
        "es_available_observations": 120,
        "es_required_observations": 252,
    })

    assert classify_unavailability(
        observation, "var_99_pct_nav"
    ) == "INSUFFICIENT_HISTORY"

    assert classify_unavailability(
        observation, "es_99_pct_nav"
    ) == "INSUFFICIENT_HISTORY"


def test_unknown_unavailability():
    """Do not incorrectly classify unexplained missing data."""
    observation = pd.Series({
        "var_available_observations": 252,
        "var_required_observations": 252,
    })

    assert classify_unavailability(
        observation, "var_99_pct_nav"
    ) == "UNKNOWN"

    assert classify_unavailability(
        observation, "drawdown"
    ) == "UNKNOWN"


def test_warmup_is_not_evaluated():
    """Historical warm-up remains visible in evaluations."""
    metrics = calculate_risk_metrics()
    policy = load_risk_policy()

    evaluations = detect_exceptions(
        metrics.iloc[:100], policy
    )

    warmup = evaluations.loc[
        evaluations["status"] == "NOT_EVALUATED"
    ]

    assert len(warmup) > 0
    assert set(warmup["rule_id"]) == {
        "VAR_99_LIMIT",
        "ES_99_LIMIT",
    }
    assert (
        warmup["reason_code"] == "INSUFFICIENT_HISTORY"
    ).all()


def test_unexpected_missing_metric_is_unknown():
    """Missing VaR with enough history is not normal warm-up."""
    metrics = calculate_risk_metrics()
    policy = load_risk_policy()

    latest = metrics.tail(1).copy()
    latest.loc[:, "var_99_pct_nav"] = np.nan

    evaluations = detect_exceptions(latest, policy)

    var_row = evaluations.loc[
        evaluations["rule_id"] == "VAR_99_LIMIT"
    ].iloc[0]

    assert var_row["status"] == "NOT_EVALUATED"
    assert var_row["reason_code"] == "UNKNOWN"
    assert var_row["available_observations"] == 252
    assert var_row["required_observations"] == 252


def test_latest_breaches_remain_detected():
    """Existing breach detection must remain unchanged."""
    metrics = calculate_risk_metrics()
    policy = load_risk_policy()

    evaluations = detect_exceptions(
        metrics.tail(1), policy
    )

    breaches = evaluations.loc[
        evaluations["status"] == "BREACH"
    ]

    assert len(breaches) == 2
    assert set(breaches["rule_id"]) == {
        "EQUITY_EXPOSURE_LIMIT",
        "SINGLE_INSTRUMENT_LIMIT",
    }


def test_unified_queue_contains_all_non_pass(tmp_path):
    """Queue must contain every non-PASS evaluation."""
    metrics = calculate_risk_metrics()
    policy = load_risk_policy()

    # Combine an early warm-up date and the latest date.
    sample = pd.concat([
        metrics.iloc[[120]],
        metrics.tail(1),
    ], ignore_index=True)

    metrics_path = tmp_path / "metrics.csv"
    policy_path = "config/risk_limits.yaml"
    output_dir = tmp_path / "outputs"

    sample.to_csv(metrics_path, index=False)

    evaluations, exceptions, summary = save_exception_results(
        metrics_path=metrics_path,
        policy_path=policy_path,
        output_dir=output_dir,
    )

    queue = pd.read_csv(
        output_dir / "investigation_queue.csv"
    )

    assert len(evaluations) == 10

    expected = evaluations.loc[
        evaluations["status"] != "PASS"
    ]

    assert len(queue) == len(expected)
    assert set(queue["exception_id"]) == set(
        expected["exception_id"]
    )

    assert "NOT_EVALUATED" in set(queue["status"])
    assert "BREACH" in set(queue["status"])
    assert "PASS" not in set(queue["status"])

    assert set(exceptions["status"]).issubset({
        "WARNING", "BREACH"
    })

    assert summary["date"] == pd.Timestamp(
        metrics.iloc[-1]["date"]
    ).strftime("%Y-%m-%d")

def test_output_files_and_audit_consistency(tmp_path):
    """All outputs must remain consistent."""
    metrics = calculate_risk_metrics()

    metrics_path = tmp_path / "metrics.csv"
    output_dir = tmp_path / "outputs"

    metrics.to_csv(metrics_path, index=False)

    evaluations, exceptions, summary = save_exception_results(
        metrics_path=metrics_path,
        output_dir=output_dir,
    )

    expected_files = [
        "risk_limit_evaluations.csv",
        "risk_exceptions.csv",
        "investigation_queue.csv",
        "latest_exception_summary.json",
    ]

    for filename in expected_files:
        assert (output_dir / filename).exists()

    queue = pd.read_csv(
        output_dir / "investigation_queue.csv"
    )

    assert len(evaluations) == len(metrics) * 5
    assert len(queue) == (
        evaluations["status"] != "PASS"
    ).sum()

    assert len(exceptions) == (
        evaluations["status"].isin(["WARNING", "BREACH"])
    ).sum()

    assert summary["breach_count"] == 2
    assert summary["not_evaluated_count"] == 0
