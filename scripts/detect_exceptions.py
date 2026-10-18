
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.risk.exception_detector import save_exception_results


def main():
    evaluations, exceptions, summary = save_exception_results()

    print(f"Generated {len(evaluations)} rule evaluations")
    print(f"Historical warnings and breaches: {len(exceptions)}")

    print(f"\nLatest date: {summary['date']}")
    print(f"PASS: {summary['pass_count']}")
    print(f"WARNING: {summary['warning_count']}")
    print(f"BREACH: {summary['breach_count']}")
    print(f"NOT_EVALUATED: {summary['not_evaluated_count']}")

    latest = evaluations.loc[
        evaluations["date"] == summary["date"],
        ["rule_id", "metric_value", "status"],
    ]

    print("\nLatest risk limit evaluation:")
    print(latest.to_string(index=False))

    print("\nException detection: PASS")


if __name__ == "__main__":
    main()
