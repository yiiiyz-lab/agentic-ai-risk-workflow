
from pathlib import Path

import numpy as np
import pandas as pd

from src.risk.calculator import ASSET_GROUPS
from src.tools.common import normalize_date, read_csv


def _number(value, name):
    """Validate a finite numeric input."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be numeric"
        ) from exc

    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")

    return result


def _date(value, name):
    """Parse a date and reject timestamps with a time."""
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}") from exc

    if pd.isna(result) or result != result.normalize():
        raise ValueError(f"Invalid {name}")

    if result.tzinfo is not None:
        raise ValueError(
            f"{name} must be timezone-naive"
        )

    return result


def _load_portfolio(
    portfolio_id,
    as_of,
    positions_path,
    portfolio_path,
):
    """Load and reconcile current portfolio exposures."""
    positions = read_csv(
        positions_path,
        required_columns={
            "date",
            "portfolio_id",
            "instrument_id",
            "market_value_usd",
        },
    ).copy()

    portfolio = read_csv(
        portfolio_path,
        required_columns={
            "date",
            "portfolio_id",
            "nav_usd",
            "cash_usd",
        },
    ).copy()

    for frame in (positions, portfolio):
        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="raise",
        )

    holdings = positions.loc[
        (positions["portfolio_id"] == portfolio_id)
        & (positions["date"] == as_of)
    ].copy()

    snapshots = portfolio.loc[
        (portfolio["portfolio_id"] == portfolio_id)
        & (portfolio["date"] == as_of)
    ]

    if holdings.empty or len(snapshots) != 1:
        raise ValueError(
            "Unique portfolio snapshot not found"
        )

    if holdings["instrument_id"].isna().any():
        raise ValueError("Missing instrument identifier")

    if holdings["instrument_id"].duplicated().any():
        raise ValueError("Duplicate portfolio positions")

    holdings["market_value_usd"] = [
        _number(value, "market_value_usd")
        for value in holdings["market_value_usd"]
    ]

    snapshot = snapshots.iloc[0]

    nav = _number(snapshot["nav_usd"], "nav_usd")
    cash = _number(snapshot["cash_usd"], "cash_usd")

    if nav <= 0:
        raise ValueError("Portfolio NAV must be positive")

    market_value = float(
        holdings["market_value_usd"].sum()
    )

    difference = market_value + cash - nav

    tolerance = max(0.01, abs(nav) * 1e-8)

    if abs(difference) > tolerance:
        raise ValueError(
            "Portfolio NAV does not reconcile with "
            "positions and cash"
        )

    holdings = holdings.sort_values(
        "instrument_id"
    )

    return holdings, nav, cash


def _load_prices(prices_path, instruments, as_of):
    """Load adjusted prices available as of the request."""
    prices = read_csv(
        prices_path,
        required_columns={
            "date",
            *instruments,
        },
    ).copy()

    prices["date"] = pd.to_datetime(
        prices["date"],
        errors="raise",
    )

    # Future observations must not enter calculations.
    prices = prices.loc[
        prices["date"] <= as_of
    ].copy()

    if prices["date"].duplicated().any():
        raise ValueError("Duplicate historical price dates")

    prices = prices.sort_values(
        "date"
    ).set_index("date")

    if as_of not in prices.index:
        raise ValueError(
            "As-of date missing from adjusted prices"
        )

    for instrument in instruments:
        prices[instrument] = pd.to_numeric(
            prices[instrument],
            errors="raise",
        )

    values = prices[instruments].to_numpy()

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite adjusted prices"
        )

    if (values <= 0).any():
        raise ValueError(
            "Adjusted prices must be positive"
        )

    return prices


def _scenario_shocks(
    scenario,
    instruments,
    as_of,
    prices,
):
    """Resolve historical or hypothetical ETF shocks."""
    scenario_type = scenario.get("type")

    if scenario_type == "historical":
        start = _date(
            scenario.get("start_date"),
            "start_date",
        )
        end = _date(
            scenario.get("end_date"),
            "end_date",
        )

        if start >= end:
            raise ValueError(
                "Historical start_date must precede end_date"
            )

        if end >= as_of:
            raise ValueError(
                "Historical scenario must end before "
                "the investigation as-of date"
            )

        if start not in prices.index:
            raise ValueError(
                "Historical start_date missing from prices"
            )

        if end not in prices.index:
            raise ValueError(
                "Historical end_date missing from prices"
            )

        shocks = {
            symbol: float(
                prices.loc[end, symbol]
                / prices.loc[start, symbol]
                - 1.0
            )
            for symbol in instruments
        }

        return shocks, {
            "start_date": normalize_date(start),
            "end_date": normalize_date(end),
            "shock_source": "historical_adjusted_prices",
        }

    if scenario_type == "hypothetical":
        supplied = scenario.get("shocks")

        if not isinstance(supplied, dict):
            raise ValueError(
                "Hypothetical shocks must be a dictionary"
            )

        if set(supplied) != set(instruments):
            raise ValueError(
                "Hypothetical shocks must cover "
                "every current instrument exactly"
            )

        shocks = {
            symbol: _number(
                supplied[symbol],
                f"shock for {symbol}",
            )
            for symbol in instruments
        }

        if any(
            shock < -1.0
            for shock in shocks.values()
        ):
            raise ValueError(
                "An ETF price-return shock cannot "
                "be less than -100%"
            )

        return shocks, {
            "shock_source": "user_defined_hypothetical",
        }

    raise ValueError(
        "Scenario type must be 'historical' "
        "or 'hypothetical'"
    )


def _evaluate_scenario(
    scenario,
    holdings,
    nav,
    cash,
    as_of,
    prices,
):
    """Value one scenario and reconcile its attribution."""
    if not isinstance(scenario, dict):
        raise ValueError(
            "Each scenario must be a dictionary"
        )

    scenario_id = scenario.get("id")
    scenario_name = scenario.get("name")

    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ValueError("Scenario id is required")

    if not isinstance(scenario_name, str) or not scenario_name.strip():
        raise ValueError("Scenario name is required")

    instruments = holdings[
        "instrument_id"
    ].tolist()

    shocks, source = _scenario_shocks(
        scenario=scenario,
        instruments=instruments,
        as_of=as_of,
        prices=prices,
    )

    instrument_results = []

    for _, position in holdings.iterrows():
        symbol = position["instrument_id"]
        market_value = float(
            position["market_value_usd"]
        )

        shock = shocks[symbol]
        pnl = market_value * shock

        instrument_results.append({
            "instrument_id": symbol,
            "asset_class": ASSET_GROUPS.get(
                symbol,
                "Unclassified",
            ),
            "market_value_usd": market_value,
            "portfolio_weight": market_value / nav,
            "scenario_return": shock,
            "stress_pnl_usd": float(pnl),
            "stress_pnl_pct_nav": float(pnl / nav),
            "stressed_market_value_usd": float(
                market_value + pnl
            ),
        })

    portfolio_pnl = float(
        sum(
            item["stress_pnl_usd"]
            for item in instrument_results
        )
    )

    asset_classes = sorted({
        item["asset_class"]
        for item in instrument_results
    })

    asset_class_results = []

    for asset_class in asset_classes:
        matching = [
            item
            for item in instrument_results
            if item["asset_class"] == asset_class
        ]

        class_pnl = float(
            sum(
                item["stress_pnl_usd"]
                for item in matching
            )
        )

        asset_class_results.append({
            "asset_class": asset_class,
            "market_value_usd": float(
                sum(
                    item["market_value_usd"]
                    for item in matching
                )
            ),
            "stress_pnl_usd": class_pnl,
            "stress_pnl_pct_nav": class_pnl / nav,
        })

    class_total = float(
        sum(
            item["stress_pnl_usd"]
            for item in asset_class_results
        )
    )

    stressed_nav = nav + portfolio_pnl

    stressed_assets_plus_cash = (
        sum(
            item["stressed_market_value_usd"]
            for item in instrument_results
        )
        + cash
    )

    reconciliation_difference = float(
        class_total - portfolio_pnl
    )

    nav_reconciliation_difference = float(
        stressed_assets_plus_cash - stressed_nav
    )

    tolerance = max(0.01, abs(nav) * 1e-8)

    reconciliation_pass = (
        abs(reconciliation_difference) <= tolerance
        and abs(nav_reconciliation_difference) <= tolerance
    )

    if not reconciliation_pass:
        raise ValueError(
            "Stress attribution reconciliation failed"
        )

    return {
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "scenario_type": scenario["type"],
        "status": "SUCCESS",
        "scenario_source": source,
        "portfolio_stress_pnl_usd": portfolio_pnl,
        "portfolio_stress_pnl_pct_nav": (
            portfolio_pnl / nav
        ),
        "portfolio_stress_loss_usd": -portfolio_pnl,
        "portfolio_stress_loss_pct_nav": (
            -portfolio_pnl / nav
        ),
        "stressed_nav_usd": float(stressed_nav),
        "instrument_attribution": instrument_results,
        "asset_class_attribution": asset_class_results,
        "reconciliation": {
            "status": "PASS",
            "asset_class_pnl_difference_usd": (
                reconciliation_difference
            ),
            "stressed_nav_difference_usd": (
                nav_reconciliation_difference
            ),
        },
    }


def run_stress_test(
    portfolio_id,
    as_of_date,
    scenarios,
    positions_path="data/sample/daily_positions.csv",
    portfolio_path="data/sample/daily_portfolio.csv",
    prices_path="data/sample/adjusted_prices.csv",
):
    """
    Apply historical or hypothetical ETF price shocks
    to a static portfolio snapshot.

    Historical scenarios must end before as_of_date.

    Hypothetical scenarios require decimal returns
    for every current instrument, e.g. -0.10 = -10%.

    Cash is held constant. No trades or rebalancing
    occur during the scenario.
    """
    as_of = _date(as_of_date, "as_of_date")

    if not isinstance(scenarios, (list, tuple)) or not scenarios:
        raise ValueError(
            "scenarios must be a non-empty list"
        )

    holdings, nav, cash = _load_portfolio(
        portfolio_id=portfolio_id,
        as_of=as_of,
        positions_path=positions_path,
        portfolio_path=portfolio_path,
    )

    instruments = holdings[
        "instrument_id"
    ].tolist()

    prices = _load_prices(
        prices_path=prices_path,
        instruments=instruments,
        as_of=as_of,
    )

    results = []
    scenario_ids = set()

    for scenario in scenarios:
        result = _evaluate_scenario(
            scenario=scenario,
            holdings=holdings,
            nav=nav,
            cash=cash,
            as_of=as_of,
            prices=prices,
        )

        if result["scenario_id"] in scenario_ids:
            raise ValueError(
                "Duplicate scenario id"
            )

        scenario_ids.add(result["scenario_id"])
        results.append(result)

    worst = min(
        results,
        key=lambda item: item[
            "portfolio_stress_pnl_usd"
        ],
    )

    warnings = []

    if any(
        item["scenario_type"] == "hypothetical"
        for item in results
    ):
        warnings.append({
            "code": "ILLUSTRATIVE_HYPOTHETICAL_SHOCKS",
            "severity": "INFO",
            "message": (
                "Hypothetical shocks are supplied "
                "assumptions, not model forecasts."
            ),
        })

    if any(
        item["stressed_nav_usd"] < 0
        for item in results
    ):
        warnings.append({
            "code": "NEGATIVE_STRESSED_NAV",
            "severity": "WARNING",
            "message": (
                "At least one scenario produces "
                "negative stressed portfolio NAV."
            ),
        })

    return {
        "tool": "run_stress_test",
        "portfolio_id": portfolio_id,
        "as_of_date": normalize_date(as_of),
        "status": "SUCCESS",
        "portfolio_snapshot": {
            "nav_usd": nav,
            "cash_usd": cash,
            "instrument_count": len(instruments),
        },
        "summary": {
            "scenario_count": len(results),
            "worst_scenario_id": worst["scenario_id"],
            "worst_scenario_name": worst["scenario_name"],
            "worst_stress_pnl_usd": worst[
                "portfolio_stress_pnl_usd"
            ],
            "worst_stress_loss_pct_nav": worst[
                "portfolio_stress_loss_pct_nav"
            ],
        },
        "scenarios": results,
        "governance_warnings": warnings,
        "methodology": {
            "valuation": (
                "Current market value multiplied by "
                "scenario simple return"
            ),
            "historical_shocks": (
                "Adjusted price at scenario end divided "
                "by adjusted price at scenario start, "
                "minus one"
            ),
            "hypothetical_shocks": (
                "Explicit instrument-level decimal "
                "price-return assumptions"
            ),
            "portfolio_assumption": (
                "Static positions, unchanged cash, "
                "no rebalancing"
            ),
            "historical_information_cutoff": (
                "Historical scenario end date must "
                "precede investigation as-of date"
            ),
        },
        "data_sources": [
            str(Path(positions_path)),
            str(Path(portfolio_path)),
            str(Path(prices_path)),
        ],
        "limitations": [
            "Static-position, linear price-shock valuation.",
            "ETF price shocks are not direct DV01 or CS01 shocks.",
            "No dynamic hedging, rebalancing, transaction costs, "
            "funding effects, or market impact.",
            "Historical scenarios do not predict future losses.",
            "Hypothetical scenarios depend on supplied assumptions.",
            "Stress P&L is not a VaR or ES estimate.",
        ],
    }
