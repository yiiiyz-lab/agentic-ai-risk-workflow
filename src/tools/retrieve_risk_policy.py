
from pathlib import Path
import hashlib
import json

from src.risk.exception_detector import load_risk_policy


def _comparison_rules(rule):
    """
    Describe exact boundary behavior used by evaluate_rule().
    Do not independently evaluate metric observations.
    """
    direction = rule["direction"]
    warning = float(rule["warning_threshold"])
    breach = float(rule["breach_threshold"])

    if direction == "max":
        return {
            "pass": {
                "operator": "<",
                "threshold": warning,
            },
            "warning": {
                "lower_operator": ">=",
                "lower_threshold": warning,
                "upper_operator": "<=",
                "upper_threshold": breach,
            },
            "breach": {
                "operator": ">",
                "threshold": breach,
            },
        }

    return {
        "pass": {
            "operator": ">",
            "threshold": warning,
        },
        "warning": {
            "lower_operator": ">=",
            "lower_threshold": breach,
            "upper_operator": "<=",
            "upper_threshold": warning,
        },
        "breach": {
            "operator": "<",
            "threshold": breach,
        },
    }


def retrieve_risk_policy(
    portfolio_id,
    rule_id=None,
    metric=None,
    policy_path="config/risk_limits.yaml",
):
    """
    Retrieve validated risk policy rules.

    Supports optional filtering by rule_id and/or metric.

    This is a read-only evidence retrieval tool.
    It does not evaluate risk observations, detect
    exceptions, or determine escalation actions.
    """
    path = Path(policy_path)

    # Reuse the authoritative policy parser and validation.
    policy = load_risk_policy(path)

    if portfolio_id != policy["portfolio_id"]:
        raise ValueError(
            "Requested portfolio_id does not match "
            "the configured risk policy"
        )

    for name, value in [
        ("rule_id", rule_id),
        ("metric", metric),
    ]:
        if value is not None and (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                f"{name} must be a non-empty string"
            )

    selected = [
        rule
        for rule in policy["rules"]
        if (
            rule_id is None
            or rule["rule_id"] == rule_id
        )
        and (
            metric is None
            or rule["metric"] == metric
        )
    ]

    if not selected:
        raise ValueError(
            "No matching risk policy rules found"
        )

    # Fingerprint the exact policy file used for retrieval.
    policy_hash = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    rules = []

    for rule in selected:
        warning = float(rule["warning_threshold"])
        breach = float(rule["breach_threshold"])

        rules.append({
            "rule_id": rule["rule_id"],
            "metric": rule["metric"],
            "label": rule.get(
                "label",
                rule["metric"],
            ),
            "direction": rule["direction"],
            "warning_threshold": warning,
            "breach_threshold": breach,
            "unit": rule.get("unit"),
            "severity": rule["severity"],
            "status_boundaries": _comparison_rules(
                rule
            ),
            "source_reference": {
                "policy_path": str(path),
                "policy_version": policy.get(
                    "policy_version"
                ),
                "rule_id": rule["rule_id"],
                "sha256": policy_hash,
            },
        })

    return {
        "tool": "retrieve_risk_policy",
        "status": "SUCCESS",
        "portfolio_id": portfolio_id,
        "policy_version": policy.get(
            "policy_version"
        ),
        "policy_type": policy.get(
            "policy_type"
        ),
        "query": {
            "rule_id": rule_id,
            "metric": metric,
        },
        "summary": {
            "total_policy_rules": len(policy["rules"]),
            "matched_rule_count": len(rules),
        },
        "rules": rules,
        "policy_source": {
            "path": str(path),
            "sha256": policy_hash,
        },
        "methodology": {
            "validation": (
                "Uses the existing load_risk_policy "
                "function from exception_detector.py"
            ),
            "boundary_semantics": (
                "Breach uses strict inequality; "
                "warning includes both threshold boundaries"
            ),
            "evaluation": (
                "No metric evaluation or exception "
                "detection is performed"
            ),
        },
        "limitations": [
            "Configured policy is illustrative, "
            "not evidence of institutional approval.",
            "Policy version is metadata, not proof "
            "of historical effective dates.",
            "No escalation workflow or approval "
            "requirements are inferred.",
            "The tool does not determine whether "
            "a risk metric is currently in breach.",
        ],
    }
