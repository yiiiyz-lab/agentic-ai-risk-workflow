
"""Run one live, tool-driven risk exception investigation."""

import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.agent.budget import AgentBudget
from src.agent.openai_client import build_live_client
from src.agent.orchestrator import InvestigationOrchestrator


EXCEPTIONS_PATH = Path(
    "data/outputs/risk_exceptions.csv"
)

OUTPUT_PATH = Path(
    "data/outputs/live_investigation_equity_20260930.json"
)

TARGET_PORTFOLIO = "PORT_001"
TARGET_DATE = "2026-09-30"
TARGET_RULE = "EQUITY_EXPOSURE_LIMIT"


def load_target_exception():
    """Load one genuine exception from the risk engine."""

    if not EXCEPTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Exception file not found: {EXCEPTIONS_PATH}"
        )

    df = pd.read_csv(EXCEPTIONS_PATH)

    required_columns = {
        "portfolio_id",
        "date",
        "exception_id",
        "rule_id",
        "status",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing exception columns: {sorted(missing)}"
        )

    matches = df.loc[
        (df["portfolio_id"] == TARGET_PORTFOLIO)
        & (df["date"] == TARGET_DATE)
        & (df["rule_id"] == TARGET_RULE)
        & (df["status"] == "BREACH")
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one matching BREACH; "
            f"found {len(matches)}"
        )

    # Convert pandas values to ordinary Python/JSON types.
    record = matches.iloc[0].to_dict()

    clean_record = {}

    for key, value in record.items():
        if pd.isna(value):
            clean_record[key] = None
        elif hasattr(value, "item"):
            clean_record[key] = value.item()
        else:
            clean_record[key] = value

    return clean_record


def main():
    load_dotenv()

    exception = load_target_exception()

    budget = AgentBudget(
        max_llm_calls=10,
        max_tool_calls=12,
        max_output_tokens_per_call=2000,
        max_estimated_cost_usd=Decimal("0.05"),
    )

    client = build_live_client(budget)

    orchestrator = InvestigationOrchestrator(
        adapter=client,
        budget=budget,
        max_rounds=10,
    )

    print("=" * 60)
    print("FIRST LIVE AI RISK INVESTIGATION")
    print("=" * 60)

    print("\nTarget exception:")
    print("Portfolio:", exception["portfolio_id"])
    print("Date:", exception["date"])
    print("Rule:", exception["rule_id"])
    print("Status:", exception["status"])
    print("Observed metric:", exception.get("metric_value"))

    print("\nStarting live investigation...")
    print("Estimated application budget: $0.05")

    print(
        "Maximum model calls:",
        budget.max_llm_calls,
    )
    print(
        "Maximum tool calls:",
        budget.max_tool_calls,
    )

    result = orchestrator.investigate(exception)

    print("\n" + "=" * 60)
    print("INVESTIGATION RESULT")
    print("=" * 60)

    print("\nStatus:", result.state.status)

    print("\nTools used:")
    for evidence_index, evidence in enumerate(
        result.state.evidence
    ):
        print(
            f"  [{evidence_index}] "
            f"{evidence.get('tool_name', 'unknown')}"
        )

    print("\nDraft report:")
    print(
        result.draft_report
        or "(No draft report generated)"
    )

    print("\nLimitations:")
    for limitation in result.state.limitations:
        print("-", limitation)

    print("\nBudget:")
    print(
        json.dumps(
            result.budget_snapshot,
            indent=2,
        )
    )

    print("\nAPI audit:")
    for record in client.audit_log:
        print(
            json.dumps(
                record.to_dict(),
                indent=2,
            )
        )

    # Save the report, evidence, and investigation trace.
    # Do not save raw provider reasoning items.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = result.to_dict()
    output["api_audit"] = [
        record.to_dict()
        for record in client.audit_log
    ]

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved investigation to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
