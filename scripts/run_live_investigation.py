"""Run one live, tool-driven risk exception investigation."""

import argparse
import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.agent.budget import AgentBudget
from src.agent.openai_client import build_live_client
from src.agent.orchestrator import InvestigationOrchestrator


EXCEPTIONS_PATH = Path("data/outputs/risk_exceptions.csv")
OUTPUT_DIRECTORY = Path("data/outputs")

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a live investigation for a selected risk exception."
    )
    parser.add_argument("--portfolio", help="Portfolio ID (e.g. PORT_001)")
    parser.add_argument("--date", help="Exception date (YYYY-MM-DD)")
    parser.add_argument("--rule", help="Risk rule ID")
    parser.add_argument("--status", help="Exception status (BREACH or WARNING)")
    parser.add_argument("--exception-id", help="Optional exact exception ID")
    parser.add_argument(
        "--list", action="store_true",
        help="List available exceptions without making API calls",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip confirmation before API calls or replacing an output file",
    )
    return parser.parse_args(argv)


def read_exceptions():
    if not EXCEPTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Exception file not found: {EXCEPTIONS_PATH}"
        )
    df = pd.read_csv(
        EXCEPTIONS_PATH,
        dtype={
            "portfolio_id": "string",
            "date": "string",
            "exception_id": "string",
            "rule_id": "string",
            "status": "string",
        },
    )
    required = {"portfolio_id", "date", "exception_id", "rule_id", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing exception columns: {sorted(missing)}")
    return df


def list_exceptions(df):
    cols = ["date", "portfolio_id", "rule_id", "status", "exception_id"]
    rows = df.sort_values(["date", "portfolio_id", "rule_id"], ascending=[
        False, True, True
    ])
    print(f"\\nAvailable exceptions ({len(rows)}):")
    print(rows[cols].to_string(index=False, max_rows=60))
    if len(rows) > 60:
        print("Showing up to 60 rows. Use --date/--rule to select a case.")


def choose_exception(df, args):
    """Use CLI filters or interactive selection; never guess an ambiguous case."""
    any_filters = any((
        args.portfolio, args.date, args.rule, args.status, args.exception_id
    ))
    if not any_filters:
        print("\\nSelect an exception from the available records.")
        # The most recent exceptions are displayed for convenient selection.
        recent = df.sort_values("date", ascending=False).head(30)
        for index, (_, row) in enumerate(recent.iterrows(), start=1):
            print(
                f"{index:>2}. {row['date']} | {row['portfolio_id']} | "
                f"{row['rule_id']} | {row['status']} | {row['exception_id']}"
            )
        choice = input("\\nEnter a number (or q to cancel): ").strip()
        if choice.lower() == "q":
            raise SystemExit("Cancelled; no API calls made.")
        if not choice.isdigit() or not 1 <= int(choice) <= len(recent):
            raise ValueError("Invalid selection; no API calls made.")
        selected = recent.iloc[int(choice) - 1]
        args.exception_id = str(selected["exception_id"])
        args.portfolio = str(selected["portfolio_id"])
        args.date = str(selected["date"])
        args.rule = str(selected["rule_id"])
        args.status = str(selected["status"])

    matches = df
    filters = {
        "portfolio_id": args.portfolio,
        "date": args.date,
        "rule_id": args.rule,
        "status": args.status,
        "exception_id": args.exception_id,
    }
    for column, value in filters.items():
        if value is not None:
            matches = matches.loc[matches[column].astype(str) == str(value)]

    if len(matches) != 1:
        print(f"Found {len(matches)} matching exceptions.")
        if not matches.empty:
            list_exceptions(matches)
        raise ValueError(
            "Selection must match exactly one exception. "
            "Specify --portfolio, --date, --rule and optionally "
            "--status or --exception-id."
        )

    record = matches.iloc[0].to_dict()
    clean = {}
    for key, value in record.items():
        if pd.isna(value):
            clean[key] = None
        elif hasattr(value, "item"):
            clean[key] = value.item()
        else:
            clean[key] = value
    return clean


def get_investigation_output_path(exception):
    """Build a stable filename from the validated source exception."""
    rule_id = str(exception["rule_id"])
    as_of_date = date.fromisoformat(str(exception["date"])[:10])
    rule_names = {
        "EQUITY_EXPOSURE_LIMIT": "equity",
        "PNL_VAR_EXCEEDANCE": "pnl",
    }
    rule_name = rule_names.get(
        rule_id,
        re.sub(r"[^a-z0-9]+", "_", rule_id.lower()).strip("_"),
    )
    if not rule_name:
        raise ValueError("Cannot derive an output name from the rule ID")
    portfolio_id = str(exception["portfolio_id"])
    # Preserve existing PORT_001 filenames used by the dashboard.
    # Include other portfolio IDs to avoid cross-portfolio overwrites.
    portfolio_suffix = (
        "" if portfolio_id == "PORT_001"
        else "_" + re.sub(r"[^a-z0-9]+", "_", portfolio_id.lower()).strip("_")
    )
    return OUTPUT_DIRECTORY / (
        f"live_investigation_{rule_name}_{as_of_date:%Y%m%d}"
        f"{portfolio_suffix}.json"
    )


def validate_result_identifiers(result, exception):
    """Prevent a result being written under the wrong case filename."""
    state = result.state
    expected = {
        "portfolio_id": str(exception["portfolio_id"]),
        "as_of_date": str(exception["date"]),
        "rule_id": str(exception["rule_id"]),
        "exception_id": str(exception["exception_id"]),
    }
    actual = {
        "portfolio_id": str(state.portfolio_id),
        "as_of_date": str(state.as_of_date),
        "rule_id": str(state.rule_id),
        "exception_id": str(state.exception_id),
    }
    mismatches = {
        key: (expected[key], actual[key])
        for key in expected
        if expected[key] != actual[key]
    }
    if mismatches:
        raise ValueError(
            f"Investigation result identifiers do not match: {mismatches}"
        )


def main(argv=None):
    args = parse_args(argv)
    df = read_exceptions()
    if args.list:
        list_exceptions(df)
        return

    exception = choose_exception(df, args)
    output_path = get_investigation_output_path(exception)

    print("\\nSelected exception:")
    print(f"  Portfolio: {exception['portfolio_id']}")
    print(f"  Date:      {exception['date']}")
    print(f"  Rule:      {exception['rule_id']}")
    print(f"  Status:    {exception['status']}")
    print(f"  Output:    {output_path}")
    if output_path.exists():
        print("WARNING: An existing saved investigation will be replaced.")
    if not args.yes:
        confirmation = input(
            "Proceed with paid API calls and save this result? [y/N]: "
        ).strip().lower()
        if confirmation not in ("y", "yes"):
            print("Cancelled; no API calls made.")
            return

    load_dotenv()

    budget = AgentBudget(
        max_llm_calls=10,
        max_tool_calls=12,
        max_output_tokens_per_call=6000,
        max_estimated_cost_usd=Decimal("0.05"),
    )
    client = build_live_client(budget)
    orchestrator = InvestigationOrchestrator(
        adapter=client,
        budget=budget,
        max_rounds=10,
    )

    print("=" * 60)
    print("LIVE AI RISK INVESTIGATION")
    print("=" * 60)
    print("\nTarget exception:")
    print("Portfolio:", exception["portfolio_id"])
    print("Date:", exception["date"])
    print("Rule:", exception["rule_id"])
    print("Status:", exception["status"])
    print("Observed metric:", exception.get("metric_value"))
    print("\nStarting live investigation...")
    print("Estimated application budget: $0.05")
    print("Maximum model calls:", budget.max_llm_calls)
    print("Maximum tool calls:", budget.max_tool_calls)

    result = orchestrator.investigate(exception)

    print("\n" + "=" * 60)
    print("INVESTIGATION RESULT")
    print("=" * 60)
    print("\nStatus:", result.state.status)
    print("\nTools used:")
    for evidence_index, evidence in enumerate(result.state.evidence):
        print(
            f"  [{evidence_index}] "
            f"{evidence.get('tool_name', 'unknown')}"
        )

    print("\nDraft report:")
    print(result.draft_report or "(No draft report generated)")
    print("\nLimitations:")
    for limitation in result.state.limitations:
        print("-", limitation)
    print("\nBudget:")
    print(json.dumps(result.budget_snapshot, indent=2))
    print("\nAPI audit:")
    for record in client.audit_log:
        print(json.dumps(record.to_dict(), indent=2))

    # Preserve all existing output fields, including evidence and trace.
    # Never serialize raw provider reasoning items.
    validate_result_identifiers(result, exception)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = result.to_dict()
    output["api_audit"] = [
        record.to_dict() for record in client.audit_log
    ]
    output_path.write_text(
        json.dumps(output, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    print("\nSaved investigation to:")
    print(output_path)


if __name__ == "__main__":
    main()
