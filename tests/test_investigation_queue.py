
import json

import numpy as np
import pandas as pd
import pytest

from src.risk.calculator import calculate_risk_metrics
from src.risk.exception_detector import (
    load_risk_policy,
    evaluate_rule,
    classify_unavailability,
    detect_exceptions,
    save_exception_results,
    INVESTIGATION_STATUSES,
)


@pytest.fixture(scope="module")
def policy():
    return load_risk_policy()


@pytest.fixture(scope="module")
def metrics():
    return calculate_risk_metrics()


def test_all_rules_evaluated_every_day(metrics, policy):
    """Every configured rule must be evaluated for each date."""
    evaluations = detect_exceptions(metrics, policy)

    assert len(evaluations) == (
        len(metrics) * len(policy["rules"])
    )

    counts = evaluations.groupby(
        ["date", "portfolio_id"]
    )["rule_id"].nunique()

    assert (
        counts == len(policy["rules"])
    ).all()


def test_evaluation_statuses_are_valid(metrics, policy):
    """Every evaluation must have a supported status."""
    evaluations = detect_exceptions(metrics, policy)

    valid_statuses = {
        "PASS",
        "WARNING",
        "BREACH",
        "NOT_EVALUATED",
    }

    assert set(evaluations["status"]).issubset(
        valid_statuses
    )


def test_warmup_is_not_evaluated(metrics, policy):
    """Historical warm-up must remain visible in evaluations."""
    evaluations = detect_exceptions(
        metrics.iloc[:100],
        policy,
    )

    warmup = evaluations.loc[
        evaluations["status"] == "NOT_EVALUATED"
    ]

    assert not warmup.empty

    assert set(warmup["rule_id"]) == {
        "VAR_99_LIMIT",
        "ES_99_LIMIT",
        "PNL_VAR_EXCEEDANCE",
    }

    assert set(warmup["reason_code"]) == {
        "INSUFFICIENT_HISTORY",
        "UNKNOWN",
    }


def test_var_and_es_warmup_reason_codes(metrics, policy):
    """VaR and ES warm-up must be classified correctly."""
    evaluations = detect_exceptions(
        metrics.iloc[1:100],
        policy,
    )

    warmup = evaluations.loc[
        evaluations["rule_id"].isin([
            "VAR_99_LIMIT",
            "ES_99_LIMIT",
        ])
    ]

    assert (
        warmup["status"] == "NOT_EVALUATED"
    ).all()

    assert (
        warmup["reason_code"] == "INSUFFICIENT_HISTORY"
    ).all()


def test_pnl_warmup_reason_codes(metrics, policy):
    """P&L monitoring must use prior-day VaR availability."""
    evaluations = detect_exceptions(
        metrics.iloc[:100],
        policy,
    )

    pnl = evaluations.loc[
        evaluations["rule_id"] == "PNL_VAR_EXCEEDANCE"
    ]

    assert (
        pnl["status"] == "NOT_EVALUATED"
    ).all()

    # The first date has no prior trading-day observation.
    first = pnl.iloc[0]

    assert first["reason_code"] == "UNKNOWN"
    assert pd.isna(first["available_observations"])

    # Subsequent warm-up dates have prior-day history counts.
    subsequent = pnl.iloc[1:]

    assert (
        subsequent["reason_code"]
        == "INSUFFICIENT_HISTORY"
    ).all()


def test_pnl_first_valid_observation(metrics, policy):
    """P&L becomes evaluable only after prior-day VaR exists."""
    evaluations = detect_exceptions(metrics, policy)

    pnl = evaluations.loc[
        evaluations["rule_id"] == "PNL_VAR_EXCEEDANCE"
    ].reset_index(drop=True)

    valid_prior_var = (
        metrics["prior_var_99_usd"].notna()
        & np.isfinite(metrics["prior_var_99_usd"])
        & (metrics["prior_var_99_usd"] > 0)
    )

    first_valid_index = np.flatnonzero(
        valid_prior_var.to_numpy()
    )[0]

    assert (
        pnl.loc[
            :first_valid_index - 1,
            "status",
        ] == "NOT_EVALUATED"
    ).all()

    assert pnl.loc[
        first_valid_index,
        "status",
    ] in {"PASS", "WARNING", "BREACH"}


def test_pnl_monitoring_metadata(metrics, policy):
    """P&L evaluations retain prior-day VaR history metadata."""
    evaluations = detect_exceptions(metrics, policy)

    pnl = evaluations.loc[
        evaluations["rule_id"] == "PNL_VAR_EXCEEDANCE"
    ].reset_index(drop=True)

    for i in [1, 100, 253, len(metrics) - 1]:
        assert pnl.loc[
            i, "available_observations"
        ] == int(
            metrics.loc[
                i, "prior_var_available_observations"
            ]
        )

        assert pnl.loc[
            i, "required_observations"
        ] == 252


def test_pnl_breach_enters_investigation_queue(policy):
    """P&L breaches must be eligible for AI investigation."""
    observation = {
        "date": "2026-09-30",
        "portfolio_id": "PORT_001",
        "var_99_pct_nav": 0.01,
        "es_99_pct_nav": 0.02,
        "equity_exposure_pct": 0.40,
        "largest_position_weight": 0.20,
        "drawdown": -0.03,
        "pnl_loss_to_prior_var": 1.25,
        "var_available_observations": 252,
        "var_required_observations": 252,
        "es_available_observations": 252,
        "es_required_observations": 252,
        "prior_var_available_observations": 252,
    }

    evaluations = detect_exceptions(
        pd.DataFrame([observation]),
        policy,
    )

    queue = evaluations.loc[
        evaluations["status"].isin(
            INVESTIGATION_STATUSES
        )
    ]

    assert len(queue) == 1

    exception = queue.iloc[0]

    assert exception["rule_id"] == "PNL_VAR_EXCEEDANCE"
    assert exception["status"] == "BREACH"
    assert exception["metric_value"] == pytest.approx(1.25)

    assert exception["exception_id"] == (
        "2026-09-30_PNL_VAR_EXCEEDANCE"
    )


def test_unified_queue_contains_all_non_pass(
    tmp_path,
    metrics,
    policy,
):
    """Queue must contain every non-PASS evaluation."""
    sample = pd.concat(
        [
            metrics.iloc[[120]],
            metrics.tail(1),
        ],
        ignore_index=True,
    )

    metrics_path = tmp_path / "metrics.csv"
    output_dir = tmp_path / "outputs"

    sample.to_csv(
        metrics_path,
        index=False,
    )

    evaluations, exceptions, summary = (
        save_exception_results(
            metrics_path=metrics_path,
            policy_path="config/risk_limits.yaml",
            output_dir=output_dir,
        )
    )

    queue = pd.read_csv(
        output_dir / "investigation_queue.csv"
    )

    assert len(evaluations) == (
        len(sample) * len(policy["rules"])
    )

    expected_queue = evaluations.loc[
        evaluations["status"].isin(
            INVESTIGATION_STATUSES
        )
    ]

    assert len(queue) == len(expected_queue)

    assert set(queue["exception_id"]) == set(
        expected_queue["exception_id"]
    )

    assert set(queue["status"]).issubset(
        INVESTIGATION_STATUSES
    )

    assert "NOT_EVALUATED" in set(queue["status"])

    assert len(exceptions) == (
        evaluations["status"].isin(
            ["WARNING", "BREACH"]
        ).sum()
    )

    assert summary["date"] == pd.Timestamp(
        sample["date"].max()
    ).strftime("%Y-%m-%d")


def test_not_evaluated_enters_queue(tmp_path, metrics):
    """Unavailable metrics must not be silently excluded."""
    sample = metrics.iloc[:5].copy()

    metrics_path = tmp_path / "metrics.csv"
    output_dir = tmp_path / "outputs"

    sample.to_csv(
        metrics_path,
        index=False,
    )

    save_exception_results(
        metrics_path=metrics_path,
        output_dir=output_dir,
    )

    queue = pd.read_csv(
        output_dir / "investigation_queue.csv"
    )

    assert "NOT_EVALUATED" in set(queue["status"])

    assert {
        "VAR_99_LIMIT",
        "ES_99_LIMIT",
        "PNL_VAR_EXCEEDANCE",
    }.issubset(
        set(queue["rule_id"])
    )


def test_pass_does_not_enter_queue(tmp_path, policy):
    """PASS evaluations must not enter the investigation queue."""
    observation = {
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

    metrics_path = tmp_path / "metrics.csv"
    output_dir = tmp_path / "outputs"

    pd.DataFrame([observation]).to_csv(
        metrics_path,
        index=False,
    )

    evaluations, exceptions, summary = (
        save_exception_results(
            metrics_path=metrics_path,
            output_dir=output_dir,
        )
    )

    queue = pd.read_csv(
        output_dir / "investigation_queue.csv"
    )

    assert len(evaluations) == len(policy["rules"])
    assert set(evaluations["status"]) == {"PASS"}
    assert exceptions.empty
    assert queue.empty

    assert summary["pass_count"] == len(policy["rules"])
    assert summary["warning_count"] == 0
    assert summary["breach_count"] == 0
    assert summary["not_evaluated_count"] == 0


def test_output_files_and_audit_consistency(
    tmp_path,
    metrics,
    policy,
):
    """Saved outputs must agree with in-memory evaluations."""
    metrics_path = tmp_path / "metrics.csv"
    output_dir = tmp_path / "outputs"

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    evaluations, exceptions, summary = (
        save_exception_results(
            metrics_path=metrics_path,
            output_dir=output_dir,
        )
    )

    expected_files = [
        "risk_limit_evaluations.csv",
        "risk_exceptions.csv",
        "investigation_queue.csv",
        "latest_exception_summary.json",
    ]

    for filename in expected_files:
        assert (
            output_dir / filename
        ).exists()

    saved_evaluations = pd.read_csv(
        output_dir / "risk_limit_evaluations.csv"
    )

    saved_exceptions = pd.read_csv(
        output_dir / "risk_exceptions.csv"
    )

    queue = pd.read_csv(
        output_dir / "investigation_queue.csv"
    )

    with open(
        output_dir / "latest_exception_summary.json",
        "r",
        encoding="utf-8",
    ) as file:
        saved_summary = json.load(file)

    assert len(evaluations) == (
        len(metrics) * len(policy["rules"])
    )

    assert len(saved_evaluations) == len(evaluations)
    assert len(saved_exceptions) == len(exceptions)

    expected_queue = evaluations.loc[
        evaluations["status"].isin(
            INVESTIGATION_STATUSES
        )
    ]

    assert len(queue) == len(expected_queue)

    assert set(queue["exception_id"]) == set(
        expected_queue["exception_id"]
    )

    assert saved_summary == summary

    latest_date = evaluations["date"].max()

    latest = evaluations.loc[
        evaluations["date"] == latest_date
    ]

    assert (
        summary["pass_count"]
        + summary["warning_count"]
        + summary["breach_count"]
        + summary["not_evaluated_count"]
        == len(policy["rules"])
    )

    assert summary["breach_count"] == int(
        (latest["status"] == "BREACH").sum()
    )

    assert summary["not_evaluated_count"] == int(
        (latest["status"] == "NOT_EVALUATED").sum()
    )


def test_pnl_exception_id_is_unique_per_date(metrics, policy):
    """Each date has exactly one P&L rule evaluation."""
    evaluations = detect_exceptions(metrics, policy)

    pnl = evaluations.loc[
        evaluations["rule_id"] == "PNL_VAR_EXCEEDANCE"
    ]

    assert len(pnl) == len(metrics)
    assert pnl["exception_id"].is_unique


def test_all_exception_ids_are_unique(metrics, policy):
    """Every date/rule combination has a unique identifier."""
    evaluations = detect_exceptions(metrics, policy)

    assert evaluations["exception_id"].is_unique
