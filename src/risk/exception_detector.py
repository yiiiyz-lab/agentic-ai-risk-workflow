
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml


VALID_DIRECTIONS = {"max", "min"}
VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
INVESTIGATION_STATUSES = {"WARNING", "BREACH", "NOT_EVALUATED"}


def load_risk_policy(path="config/risk_limits.yaml"):
    with open(path, "r", encoding="utf-8") as file:
        policy = yaml.safe_load(file)

    if not isinstance(policy, dict):
        raise ValueError("Risk policy must be a YAML mapping")

    if not policy.get("portfolio_id"):
        raise ValueError("Missing portfolio_id")

    rules = policy.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("Risk policy must contain rules")

    rule_ids = set()

    for rule in rules:
        required = {
            "rule_id", "metric", "direction",
            "warning_threshold", "breach_threshold",
            "severity",
        }

        if not isinstance(rule, dict) or not required.issubset(rule):
            raise ValueError("Incomplete risk limit rule")

        if rule["rule_id"] in rule_ids:
            raise ValueError("Duplicate risk rule ID")

        rule_ids.add(rule["rule_id"])

        if rule["direction"] not in VALID_DIRECTIONS:
            raise ValueError("Invalid rule direction")

        if rule["severity"] not in VALID_SEVERITIES:
            raise ValueError("Invalid rule severity")

        warning = float(rule["warning_threshold"])
        breach = float(rule["breach_threshold"])

        if not np.isfinite([warning, breach]).all():
            raise ValueError("Risk thresholds must be finite")

        if rule["direction"] == "max" and warning >= breach:
            raise ValueError("Max rule requires warning < breach")

        if rule["direction"] == "min" and warning <= breach:
            raise ValueError("Min rule requires warning > breach")

    return policy


def evaluate_rule(value, rule):
    if pd.isna(value):
        return "NOT_EVALUATED"

    value = float(value)

    if not np.isfinite(value):
        return "NOT_EVALUATED"

    warning = float(rule["warning_threshold"])
    breach = float(rule["breach_threshold"])

    if rule["direction"] == "max":
        if value > breach:
            return "BREACH"
        if value >= warning:
            return "WARNING"

    else:
        if value < breach:
            return "BREACH"
        if value <= warning:
            return "WARNING"

    return "PASS"


def classify_unavailability(observation, metric):
    """
    Classify unavailable risk metrics using deterministic evidence.

    Expected VaR/ES warm-up:
        INSUFFICIENT_HISTORY

    Missing or inconsistent evidence:
        UNKNOWN

    This function does not determine whether an observation
    should enter the investigation queue.
    """
    if metric.startswith("var_99_"):
        prefix = "var"
    elif metric.startswith("es_99_"):
        prefix = "es"
    else:
        return "UNKNOWN"

    available = observation.get(
        f"{prefix}_available_observations"
    )
    required = observation.get(
        f"{prefix}_required_observations"
    )

    if pd.isna(available) or pd.isna(required):
        return "UNKNOWN"

    try:
        available = float(available)
        required = float(required)
    except (TypeError, ValueError):
        return "UNKNOWN"

    if not np.isfinite(available) or not np.isfinite(required):
        return "UNKNOWN"

    if required > 0 and 0 <= available < required:
        return "INSUFFICIENT_HISTORY"

    return "UNKNOWN"


def detect_exceptions(metrics, policy):
    if metrics.empty:
        raise ValueError("Risk metrics cannot be empty")

    if metrics.duplicated(["date", "portfolio_id"]).any():
        raise ValueError("Duplicate portfolio/date risk observations")

    required = {"date", "portfolio_id"}
    required.update(rule["metric"] for rule in policy["rules"])

    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing risk metrics: {sorted(missing)}")

    portfolio_ids = set(metrics["portfolio_id"].dropna())
    if portfolio_ids != {policy["portfolio_id"]}:
        raise ValueError("Portfolio ID does not match risk policy")

    records = []

    for _, observation in metrics.iterrows():
        date = pd.Timestamp(
            observation["date"]
        ).strftime("%Y-%m-%d")

        for rule in policy["rules"]:
            metric = rule["metric"]
            raw_value = observation[metric]
            status = evaluate_rule(raw_value, rule)

            value = (
                float(raw_value)
                if pd.notna(raw_value)
                and np.isfinite(float(raw_value))
                else None
            )

            # Classify unavailable metrics using available evidence.
            reason_code = (
                classify_unavailability(observation, metric)
                if status == "NOT_EVALUATED"
                else None
            )

            # Extract historical observation counts for VaR/ES.
            if metric.startswith("var_99_"):
                prefix = "var"
            elif metric.startswith("es_99_"):
                prefix = "es"
            else:
                prefix = None

            available_observations = (
                observation.get(
                    f"{prefix}_available_observations"
                )
                if prefix else None
            )

            required_observations = (
                observation.get(
                    f"{prefix}_required_observations"
                )
                if prefix else None
            )

            # Convert missing/non-finite metadata to None.
            if available_observations is not None:
                if pd.isna(available_observations):
                    available_observations = None
                else:
                    available_observations = int(
                        available_observations
                    )

            if required_observations is not None:
                if pd.isna(required_observations):
                    required_observations = None
                else:
                    required_observations = int(
                        required_observations
                    )

            records.append({
                "exception_id": f"{date}_{rule['rule_id']}",
                "date": date,
                "portfolio_id": policy["portfolio_id"],
                "policy_version": policy.get("policy_version", ""),
                "rule_id": rule["rule_id"],
                "metric": metric,
                "metric_value": value,
                "warning_threshold": float(
                    rule["warning_threshold"]
                ),
                "breach_threshold": float(
                    rule["breach_threshold"]
                ),
                "direction": rule["direction"],
                "severity": rule["severity"],
                "status": status,
                "reason_code": reason_code,
                "available_observations": available_observations,
                "required_observations": required_observations,
            })

    return pd.DataFrame(records)


def save_exception_results(
    metrics_path="data/outputs/daily_risk_metrics.csv",
    policy_path="config/risk_limits.yaml",
    output_dir="data/outputs",
):
    metrics = pd.read_csv(metrics_path)
    policy = load_risk_policy(policy_path)

    evaluations = detect_exceptions(metrics, policy)

    # Market risk exceptions: preserve the existing definition.
    exceptions = evaluations.loc[
        evaluations["status"].isin(["WARNING", "BREACH"])
    ].copy()

    # Unified investigation queue: all non-PASS evaluations.
    investigation_queue = evaluations.loc[
        evaluations["status"].isin(INVESTIGATION_STATUSES)
    ].copy()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluations.to_csv(
        output_dir / "risk_limit_evaluations.csv",
        index=False,
    )

    exceptions.to_csv(
        output_dir / "risk_exceptions.csv",
        index=False,
    )

    investigation_queue.to_csv(
        output_dir / "investigation_queue.csv",
        index=False,
    )

    latest_date = evaluations["date"].max()
    latest = evaluations.loc[
        evaluations["date"] == latest_date
    ]

    summary = {
        "date": latest_date,
        "portfolio_id": policy["portfolio_id"],
        "policy_version": policy.get("policy_version", ""),
        "pass_count": int(
            (latest["status"] == "PASS").sum()
        ),
        "warning_count": int(
            (latest["status"] == "WARNING").sum()
        ),
        "breach_count": int(
            (latest["status"] == "BREACH").sum()
        ),
        "not_evaluated_count": int(
            (latest["status"] == "NOT_EVALUATED").sum()
        ),
        "breaches": latest.loc[
            latest["status"] == "BREACH",
            [
                "exception_id",
                "rule_id",
                "metric_value",
                "severity",
            ],
        ].to_dict(orient="records"),
    }

    with open(
        output_dir / "latest_exception_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            allow_nan=False,
        )

    # Preserve compatibility with existing scripts/tests.
    return evaluations, exceptions, summary
