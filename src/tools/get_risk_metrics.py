
from src.tools.common import (
    read_csv,
    select_snapshot,
    json_safe,
    normalize_date,
)


def get_risk_metrics(
    portfolio_id,
    as_of_date,
    metrics_path="data/outputs/daily_risk_metrics.csv",
):
    metrics = read_csv(
        metrics_path,
        required_columns={"date", "portfolio_id"},
    )

    snapshot = select_snapshot(
        metrics,
        portfolio_id,
        as_of_date,
    )

    return {
        "tool": "get_risk_metrics",
        "portfolio_id": portfolio_id,
        "as_of_date": normalize_date(as_of_date),
        "status": "SUCCESS",
        "data": {
            key: json_safe(value)
            for key, value in snapshot.to_dict().items()
        },
        "source": str(metrics_path),
    }
