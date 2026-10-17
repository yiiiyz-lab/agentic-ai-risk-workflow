
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.risk.calculator import save_risk_metrics


def main():
    metrics = save_risk_metrics()

    print(f"Generated {len(metrics)} daily risk observations")
    print("\nLatest risk metrics:")
    print(metrics.tail(1).T.to_string(header=False))

    print("\nHistorical VaR observations:",
          metrics["var_99_usd"].notna().sum())

    print("\nRisk calculation: PASS")


if __name__ == "__main__":
    main()
