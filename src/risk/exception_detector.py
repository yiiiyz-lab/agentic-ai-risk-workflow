
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml


VALID_DIRECTIONS = {"max", "min"}
VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
INVESTIGATION_STATUSES = {"WARNING", "BREACH", "NOT_EVALUATED"}

PNL_VAR_METRIC = "pnl_loss_to_prior_var"


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
            "rule_id",
            "metric",
            "direction",
            "warning_threshold",
            "breach_threshold",
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
    """
    Evaluate one metric against its configured thresholds.

    Max rule:
        value > breach   -> BREACH
        value >= warning -> WARNING
        otherwise        -> PASS

    Min rule:
        value < breach   -> BREACH
        value <= warning -> WARNING
        otherwise        -> PASS

    Missing or non-finite values:
        NOT_EVALUATED
    """
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


def _observation_metadata(observation, metric):
    """
    Return (available, required) historical observation counts.

    VaR and ES use their existing metadata.

    P&L monitoring requires prior-day VaR, so its available
    observation count must come from the previous trading day.
    """
    if metric.startswith("var_99_"):
        available_key = "var_available_observations"
        required_key = "var_required_observations"

    elif metric.startswith("es_99_"):
        available_key = "es_available_observations"
        required_key = "es_required_observations"

    elif metric == PNL_VAR_METRIC:
        available_key = "prior_var_available_observations"
        required_key = "var_required_observations"

    else:
        return None, None

    return (
        observation.get(available_key),
        observation.get(required_key),
    )


def _normalize_observation_count(value):
    """
    Convert a finite, nonnegative, integer-valued count to int.

    Missing or invalid counts are returned as None.
    """
    if value is None or pd.isna(value):
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if (
        not np.isfinite(numeric)
        or numeric < 0
        or not numeric.is_integer()
    ):
        return None

    return int(numeric)


def classify_unavailability(observation, metric):
    """
    Classify unavailable risk metrics using deterministic evidence.

    INSUFFICIENT_HISTORY:
        Required historical observations are not yet available.

    UNKNOWN:
        Historical observations are sufficient, metadata is
        missing/invalid, or another input issue exists.

    For P&L monitoring, the historical observation count refers
    to the prior trading day's VaR availability.
    """
    available, required = _observation_metadata(
        observation,
        metric,
    )

    available = _normalize_observation_count(available)
    required = _normalize_observation_count(required)

    if available is None or required is None:
        return "UNKNOWN"

    if required > 0 and available < required:
        return "INSUFFICIENT_HISTORY"

    return "UNKNOWN"


def detect_exceptions(metrics, policy):
    """
    Evaluate every configured risk rule on every portfolio date.

    Output includes all evaluations:
        PASS
        WARNING
        BREACH
        NOT_EVALUATED

    P&L monitoring uses the same evaluation and exception
    structure as all existing risk rules.
    """
    if metrics.empty:
        raise ValueError("Risk metrics cannot be empty")

    if metrics.duplicated(["date", "portfolio_id"]).any():
        raise ValueError("Duplicate portfolio/date risk observations")

    required = {"date", "portfolio_id"}

    required.update(
        rule["metric"]
        for rule in policy["rules"]
    )

    missing = required - set(metrics.columns)

    if missing:
        raise ValueError(
            f"Missing risk metrics: {sorted(missing)}"
        )

    portfolio_ids = set(metrics["portfolio_id"].dropna())

    if portfolio_ids != {policy["portfolio_id"]}:
        raise ValueError(
            "Portfolio ID does not match risk policy"
        )

    records = []

    for _, observation in metrics.iterrows():
        date = pd.Timestamp(
            observation["date"]
        ).strftime("%Y-%m-%d")

        for rule in policy["rules"]:
            metric = rule["metric"]
            raw_value = observation[metric]

            status = evaluate_rule(
                raw_value,
                rule,
            )

            value = (
                float(raw_value)
                if status != "NOT_EVALUATED"
                else None
            )

            reason_code = (
                classify_unavailability(
                    observation,
                    metric,
                )
                if status == "NOT_EVALUATED"
                else None
            )

            available_raw, required_raw = (
                _observation_metadata(
                    observation,
                    metric,
                )
            )

            available_observations = (
                _normalize_observation_count(
                    available_raw
                )
            )

            required_observations = (
                _normalize_observation_count(
                    required_raw
                )
            )

            records.append({
                "exception_id": (
                    f"{date}_{rule['rule_id']}"
                ),
                "date": date,
                "portfolio_id": policy["portfolio_id"],
                "policy_version": policy.get(
                    "policy_version", ""
                ),
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
                "available_observations": (
                    available_observations
                ),
                "required_observations": (
                    required_observations
                ),
            })

    return pd.DataFrame(records)


def save_exception_results(
    metrics_path="data/outputs/daily_risk_metrics.csv",
    policy_path="config/risk_limits.yaml",
    output_dir="data/outputs",
):
    """
    Save risk evaluations and investigation queue.

    Outputs:
        risk_limit_evaluations.csv
        risk_exceptions.csv
        investigation_queue.csv
        latest_exception_summary.json

    P&L exceptions are included automatically when the
    PNL_VAR_EXCEEDANCE rule is configured in risk_limits.yaml.
    """
    metrics = pd.read_csv(metrics_path)
    policy = load_risk_policy(policy_path)

    evaluations = detect_exceptions(
        metrics,
        policy,
    )

    # Preserve existing definition of risk exceptions:
    # WARNING and BREACH only.
    exceptions = evaluations.loc[
        evaluations["status"].isin(
            ["WARNING", "BREACH"]
        )
    ].copy()

    # Unified investigation queue:
    # All non-PASS observations, including NOT_EVALUATED.
    investigation_queue = evaluations.loc[
        evaluations["status"].isin(
            INVESTIGATION_STATUSES
        )
    ].copy()

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        "policy_version": policy.get(
            "policy_version", ""
        ),
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
        ].to_dict(
            orient="records"
        ),
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

    # Preserve compatibility with existing scripts and tests.
    return evaluations, exceptions, summary
