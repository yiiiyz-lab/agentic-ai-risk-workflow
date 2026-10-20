
import copy
import json

import pytest
import yaml

from src.risk.exception_detector import (
    evaluate_rule,
    load_risk_policy,
)
from src.tools.retrieve_risk_policy import (
    retrieve_risk_policy,
)


@pytest.fixture
def policy_file(tmp_path):
    policy = {
        "portfolio_id": "TEST_PORT",
        "policy_version": "test-1.0",
        "policy_type": "illustrative_internal_limits",
        "rules": [
            {
                "rule_id": "EQUITY_LIMIT",
                "metric": "equity_exposure_pct",
                "label": "Equity Exposure",
                "direction": "max",
                "warning_threshold": 0.60,
                "breach_threshold": 0.65,
                "unit": "pct_nav",
                "severity": "MEDIUM",
            },
            {
                "rule_id": "DRAWDOWN_LIMIT",
                "metric": "drawdown",
                "label": "Portfolio Drawdown",
                "direction": "min",
                "warning_threshold": -0.10,
                "breach_threshold": -0.15,
                "unit": "return",
                "severity": "HIGH",
            },
        ],
    }

    path = tmp_path / "risk_limits.yaml"
    path.write_text(
        yaml.safe_dump(policy, sort_keys=False),
        encoding="utf-8",
    )
    return path


def retrieve(path, **kwargs):
    return retrieve_risk_policy(
        portfolio_id="TEST_PORT",
        policy_path=path,
        **kwargs,
    )


def test_retrieve_all_rules(policy_file):
    result = retrieve(policy_file)
    assert result["status"] == "SUCCESS"
    assert result["summary"]["matched_rule_count"] == 2


def test_retrieve_by_rule_id(policy_file):
    result = retrieve(
        policy_file,
        rule_id="EQUITY_LIMIT",
    )
    assert len(result["rules"]) == 1
    assert result["rules"][0]["metric"] == (
        "equity_exposure_pct"
    )


def test_retrieve_by_metric(policy_file):
    result = retrieve(
        policy_file,
        metric="drawdown",
    )
    assert result["rules"][0]["rule_id"] == (
        "DRAWDOWN_LIMIT"
    )


def test_retrieve_by_both_filters(policy_file):
    result = retrieve(
        policy_file,
        rule_id="EQUITY_LIMIT",
        metric="equity_exposure_pct",
    )
    assert len(result["rules"]) == 1


def test_conflicting_filters_rejected(policy_file):
    with pytest.raises(ValueError):
        retrieve(
            policy_file,
            rule_id="EQUITY_LIMIT",
            metric="drawdown",
        )


def test_unknown_rule_rejected(policy_file):
    with pytest.raises(ValueError):
        retrieve(
            policy_file,
            rule_id="UNKNOWN",
        )


def test_portfolio_mismatch_rejected(policy_file):
    with pytest.raises(ValueError):
        retrieve_risk_policy(
            portfolio_id="OTHER_PORT",
            policy_path=policy_file,
        )


def test_max_thresholds(policy_file):
    result = retrieve(
        policy_file,
        rule_id="EQUITY_LIMIT",
    )
    boundaries = result["rules"][0][
        "status_boundaries"
    ]

    assert boundaries["pass"] == {
        "operator": "<",
        "threshold": 0.60,
    }
    assert boundaries["breach"] == {
        "operator": ">",
        "threshold": 0.65,
    }


def test_min_thresholds(policy_file):
    result = retrieve(
        policy_file,
        rule_id="DRAWDOWN_LIMIT",
    )
    boundaries = result["rules"][0][
        "status_boundaries"
    ]

    assert boundaries["pass"] == {
        "operator": ">",
        "threshold": -0.10,
    }
    assert boundaries["breach"] == {
        "operator": "<",
        "threshold": -0.15,
    }


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.5999, "PASS"),
        (0.60, "WARNING"),
        (0.625, "WARNING"),
        (0.65, "WARNING"),
        (0.6501, "BREACH"),
    ],
)
def test_max_boundaries_match_detector(
    policy_file,
    value,
    expected,
):
    policy = load_risk_policy(policy_file)
    rule = policy["rules"][0]
    assert evaluate_rule(value, rule) == expected

    retrieved = retrieve(
        policy_file,
        rule_id="EQUITY_LIMIT",
    )["rules"][0]

    assert retrieved["warning_threshold"] == 0.60
    assert retrieved["breach_threshold"] == 0.65


@pytest.mark.parametrize(
    "value, expected",
    [
        (-0.0999, "PASS"),
        (-0.10, "WARNING"),
        (-0.125, "WARNING"),
        (-0.15, "WARNING"),
        (-0.1501, "BREACH"),
    ],
)
def test_min_boundaries_match_detector(
    policy_file,
    value,
    expected,
):
    policy = load_risk_policy(policy_file)
    rule = policy["rules"][1]
    assert evaluate_rule(value, rule) == expected

    retrieved = retrieve(
        policy_file,
        rule_id="DRAWDOWN_LIMIT",
    )["rules"][0]

    assert retrieved["warning_threshold"] == -0.10
    assert retrieved["breach_threshold"] == -0.15


def test_policy_version_and_type(policy_file):
    result = retrieve(policy_file)

    assert result["policy_version"] == "test-1.0"
    assert result["policy_type"] == (
        "illustrative_internal_limits"
    )


def test_source_fingerprint_stable(policy_file):
    first = retrieve(policy_file)
    second = retrieve(policy_file)

    assert first["policy_source"]["sha256"] == (
        second["policy_source"]["sha256"]
    )
    assert len(first["policy_source"]["sha256"]) == 64


def test_source_fingerprint_changes(policy_file):
    first = retrieve(policy_file)

    policy = yaml.safe_load(
        policy_file.read_text(encoding="utf-8")
    )
    policy["rules"][0]["warning_threshold"] = 0.59

    policy_file.write_text(
        yaml.safe_dump(policy),
        encoding="utf-8",
    )

    second = retrieve(policy_file)

    assert first["policy_source"]["sha256"] != (
        second["policy_source"]["sha256"]
    )


def test_duplicate_rule_ids_rejected(policy_file):
    policy = yaml.safe_load(
        policy_file.read_text(encoding="utf-8")
    )
    policy["rules"][1]["rule_id"] = "EQUITY_LIMIT"

    policy_file.write_text(
        yaml.safe_dump(policy),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        retrieve(policy_file)


def test_invalid_threshold_order_rejected(policy_file):
    policy = yaml.safe_load(
        policy_file.read_text(encoding="utf-8")
    )
    policy["rules"][0]["warning_threshold"] = 0.70

    policy_file.write_text(
        yaml.safe_dump(policy),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        retrieve(policy_file)


def test_missing_policy_file_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        retrieve(
            tmp_path / "missing.yaml"
        )


def test_empty_rule_filter_rejected(policy_file):
    with pytest.raises(ValueError):
        retrieve(policy_file, rule_id="")


def test_json_serializable(policy_file):
    result = retrieve(policy_file)
    json.dumps(result, allow_nan=False)
