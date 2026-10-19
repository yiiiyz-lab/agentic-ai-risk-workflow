
import numpy as np
import pandas as pd

from src.risk.calculator import ASSET_GROUPS
from src.tools.common import read_csv, normalize_date


VAR_WINDOW = 252
CONFIDENCE_LEVEL = 0.99


def _validate_snapshot(
    portfolio_id,
    as_of,
    positions,
    metrics,
    prices,
    symbols,
):
    """Validate the requested portfolio, date, and position universe."""

    metric_rows = metrics.loc[
        (metrics["portfolio_id"] == portfolio_id)
        & (metrics["date"] == as_of)
    ]

    if len(metric_rows) != 1:
        raise ValueError(
            "Expected exactly one risk metric observation "
            "for the requested portfolio and date"
        )

    if as_of not in prices.index:
        raise ValueError("As-of date missing from prices")

    current_positions = positions.loc[
        (positions["portfolio_id"] == portfolio_id)
        & (positions["date"] == as_of)
    ].copy()

    if current_positions["instrument_id"].duplicated().any():
        raise ValueError("Duplicate current positions")

    if set(current_positions["instrument_id"]) != set(symbols):
        raise ValueError(
            "Current positions do not match the expected universe"
        )

    if len(current_positions) != len(symbols):
        raise ValueError("Incomplete current position snapshot")

    return metric_rows.iloc[0], current_positions


def _build_tail_scenario_diagnostics(
    tail_losses,
    historical_returns,
    instrument_losses,
    symbols,
):
    """Describe every selected portfolio ES-tail scenario."""

    scenarios = []

    for date, loss in tail_losses.sort_values(
        ascending=False
    ).items():

        instrument_contributions = {
            symbol: float(instrument_losses.loc[date, symbol])
            for symbol in symbols
        }

        scenario_returns = {
            symbol: float(historical_returns.loc[date, symbol])
            for symbol in symbols
        }

        scenarios.append({
            "historical_date": normalize_date(date),
            "portfolio_loss_usd": float(loss),
            "instrument_losses_usd": instrument_contributions,
            "instrument_returns": scenario_returns,
            "reconciliation_difference_usd": float(
                sum(instrument_contributions.values()) - loss
            ),
        })

    return scenarios


def _build_concentration_diagnostics(
    instrument_attribution,
    nav,
):
    """Compare portfolio weights with ES contribution shares."""

    diagnostics = []

    for item in instrument_attribution:
        weight = item["market_value_usd"] / nav
        es_share = item["share_of_portfolio_es"]

        diagnostics.append({
            "instrument_id": item["instrument_id"],
            "asset_class": item["asset_class"],
            "portfolio_weight": float(weight),
            "es_contribution_share": es_share,
            "es_share_minus_weight": (
                float(es_share - weight)
                if es_share is not None else None
            ),
            "es_to_weight_ratio": (
                float(es_share / weight)
                if es_share is not None and weight > 0
                else None
            ),
        })

    diagnostics.sort(
        key=lambda item: (
            item["es_share_minus_weight"]
            if item["es_share_minus_weight"] is not None
            else float("-inf")
        ),
        reverse=True,
    )

    return diagnostics


def _build_leave_one_out_sensitivity(
    tail_instrument_losses,
    symbols,
):
    """
    Remove one selected ES-tail scenario at a time.

    This is a conditional sensitivity diagnostic, not a
    re-estimated 99% Historical Simulation VaR or ES.
    """

    count = len(tail_instrument_losses)

    if count < 2:
        return {
            "status": "NOT_EVALUATED",
            "reason": (
                "At least two ES-tail scenarios are required "
                "for leave-one-out analysis"
            ),
            "results": [],
        }

    results = []

    for excluded_date in tail_instrument_losses.index:
        remaining = tail_instrument_losses.drop(
            index=excluded_date
        )

        contributions = remaining.mean(axis=0)
        conditional_mean_loss = float(contributions.sum())

        results.append({
            "excluded_scenario_date": normalize_date(
                excluded_date
            ),
            "remaining_scenario_count": len(remaining),
            "conditional_mean_loss_usd": conditional_mean_loss,
            "instrument_contributions_usd": {
                symbol: float(contributions[symbol])
                for symbol in symbols
            },
        })

    conditional_losses = [
        item["conditional_mean_loss_usd"]
        for item in results
    ]

    return {
        "status": "SUCCESS",
        "method": "Leave-one-selected-tail-scenario-out",
        "results": results,
        "minimum_conditional_mean_loss_usd": float(
            min(conditional_losses)
        ),
        "maximum_conditional_mean_loss_usd": float(
            max(conditional_losses)
        ),
        "range_usd": float(
            max(conditional_losses) - min(conditional_losses)
        ),
        "interpretation": (
            "Sensitivity of the selected ES-tail sample "
            "to individual scenarios. This is not a "
            "re-estimated 99% VaR or ES."
        ),
    }


def get_var_attribution(
    portfolio_id,
    as_of_date,
    positions_path="data/sample/daily_positions.csv",
    prices_path="data/sample/adjusted_prices.csv",
    metrics_path="data/outputs/daily_risk_metrics.csv",
):
    """
    Reconstruct 99% Historical Simulation VaR/ES and attribute
    ES to instruments using common portfolio-tail scenarios.

    Enhancements:
    - Tail-scenario return and loss diagnostics
    - ES concentration relative to portfolio weights
    - Leave-one-tail-scenario-out sensitivity
    - Governance warnings
    - Point-in-time and data-quality validation

    Preserves the existing risk engine's 252-observation
    lookback and linear-quantile conventions.
    """

    symbols = list(ASSET_GROUPS)
    as_of = pd.Timestamp(as_of_date)

    # ---------------------------------------------------------
    # 1. Load source data
    # ---------------------------------------------------------

    positions = read_csv(
        positions_path,
        required_columns={
            "date",
            "portfolio_id",
            "instrument_id",
            "market_value_usd",
        },
    )

    prices = read_csv(
        prices_path,
        required_columns={"date", *symbols},
    )

    metrics = read_csv(
        metrics_path,
        required_columns={
            "date",
            "portfolio_id",
            "nav_usd",
            "var_99_usd",
            "es_99_usd",
        },
    )

    positions["date"] = pd.to_datetime(
        positions["date"], errors="raise"
    )

    prices["date"] = pd.to_datetime(
        prices["date"], errors="raise"
    )

    metrics["date"] = pd.to_datetime(
        metrics["date"], errors="raise"
    )

    if prices["date"].duplicated().any():
        raise ValueError("Duplicate price dates")

    prices = prices.sort_values("date").set_index("date")

    # ---------------------------------------------------------
    # 2. Validate requested snapshot before warm-up check
    # ---------------------------------------------------------

    metric_row, validated_positions = _validate_snapshot(
        portfolio_id=portfolio_id,
        as_of=as_of,
        positions=positions,
        metrics=metrics,
        prices=prices,
        symbols=symbols,
    )

    date_index = prices.index.get_loc(as_of)

    # Preserve the existing calculator's warm-up convention.
    # The first valid VaR/ES observation is index 253.

    if date_index <= VAR_WINDOW:
        return {
            "tool": "get_var_attribution",
            "portfolio_id": portfolio_id,
            "as_of_date": normalize_date(as_of),
            "status": "NOT_EVALUATED",
            "reason_code": "INSUFFICIENT_HISTORY",
            "available_observations": min(
                max(date_index - 1, 0),
                VAR_WINDOW,
            ),
            "required_observations": VAR_WINDOW,
            "analysis": None,
            "limitations": [
                "Historical VaR/ES attribution requires "
                "252 prior daily return observations "
                "under the existing calculator convention."
            ],
        }

    current_positions = validated_positions.set_index(
        "instrument_id"
    ).reindex(symbols)

    market_values = current_positions[
        "market_value_usd"
    ].astype(float)

    if market_values.isna().any():
        raise ValueError("Missing current market values")

    if not np.isfinite(market_values).all():
        raise ValueError("Non-finite market values")

    # ---------------------------------------------------------
    # 3. Reconstruct historical scenario returns
    # ---------------------------------------------------------

    price_data = prices[symbols].astype(float)

    if not np.isfinite(price_data.to_numpy()).all():
        raise ValueError("Non-finite adjusted prices")

    if (price_data <= 0).any().any():
        raise ValueError("Adjusted prices must be positive")

    returns = price_data.pct_change(fill_method=None)

    # Match existing calculator:
    # returns.iloc[i - 252:i]
    # Current-day return is excluded.

    historical_returns = returns.iloc[
        date_index - VAR_WINDOW:date_index
    ].copy()

    if len(historical_returns) != VAR_WINDOW:
        raise ValueError("Incorrect historical window length")

    if historical_returns.isna().any().any():
        raise ValueError("Missing historical returns")

    if not np.isfinite(
        historical_returns.to_numpy()
    ).all():
        raise ValueError("Non-finite historical returns")

    # ---------------------------------------------------------
    # 4. Calculate scenario losses, VaR, and ES
    # ---------------------------------------------------------

    instrument_losses = historical_returns.mul(
        -market_values,
        axis=1,
    )

    portfolio_losses = instrument_losses.sum(axis=1)

    raw_var = float(
        np.quantile(
            portfolio_losses.to_numpy(),
            CONFIDENCE_LEVEL,
            method="linear",
        )
    )

    calculated_var = max(0.0, raw_var)

    tail_mask = portfolio_losses >= calculated_var

    tail_losses = portfolio_losses.loc[tail_mask]

    tail_instrument_losses = instrument_losses.loc[
        tail_mask
    ]

    if tail_losses.empty:
        calculated_es = 0.0

        es_contributions = pd.Series(
            0.0,
            index=symbols,
        )

        attribution_available = False

    else:
        calculated_es = max(
            0.0,
            float(tail_losses.mean()),
        )

        if calculated_es > 0:
            es_contributions = (
                tail_instrument_losses.mean(axis=0)
            )
            attribution_available = True

        else:
            es_contributions = pd.Series(
                0.0,
                index=symbols,
            )
            attribution_available = False

    # ---------------------------------------------------------
    # 5. Reconcile with existing risk engine
    # ---------------------------------------------------------

    reported_var = float(metric_row["var_99_usd"])
    reported_es = float(metric_row["es_99_usd"])
    nav = float(metric_row["nav_usd"])

    if not np.isfinite(
        [reported_var, reported_es, nav]
    ).all():
        raise ValueError("Invalid reported risk metrics")

    if nav <= 0:
        raise ValueError("Invalid portfolio NAV")

    if not np.isclose(
        calculated_var,
        reported_var,
        rtol=1e-9,
        atol=0.05,
    ):
        raise ValueError(
            "Reconstructed VaR does not match risk engine"
        )

    if not np.isclose(
        calculated_es,
        reported_es,
        rtol=1e-9,
        atol=0.05,
    ):
        raise ValueError(
            "Reconstructed ES does not match risk engine"
        )

    total_contribution = float(
        es_contributions.sum()
    )

    if not np.isclose(
        total_contribution,
        calculated_es,
        rtol=1e-9,
        atol=0.05,
    ):
        raise ValueError(
            "ES contribution reconciliation failed"
        )

    # ---------------------------------------------------------
    # 6. Instrument-level ES attribution
    # ---------------------------------------------------------

    instrument_attribution = []

    for symbol in symbols:
        contribution = float(
            es_contributions[symbol]
        )

        instrument_attribution.append({
            "instrument_id": symbol,
            "asset_class": ASSET_GROUPS[symbol],
            "market_value_usd": float(
                market_values[symbol]
            ),
            "es_contribution_usd": contribution,
            "es_contribution_pct_nav": (
                contribution / nav
            ),
            "share_of_portfolio_es": (
                contribution / calculated_es
                if calculated_es > 0
                else None
            ),
        })

    instrument_attribution.sort(
        key=lambda item: item["es_contribution_usd"],
        reverse=True,
    )

    # ---------------------------------------------------------
    # 7. Asset-class ES attribution
    # ---------------------------------------------------------

    asset_class_attribution = []

    for group in ("Equity", "Rates", "Credit"):
        contribution = sum(
            item["es_contribution_usd"]
            for item in instrument_attribution
            if item["asset_class"] == group
        )

        asset_class_attribution.append({
            "asset_class": group,
            "es_contribution_usd": float(contribution),
            "share_of_portfolio_es": (
                contribution / calculated_es
                if calculated_es > 0
                else None
            ),
        })

    # ---------------------------------------------------------
    # 8. Worst historical scenarios (existing output)
    # ---------------------------------------------------------

    worst_scenarios = []

    for date, loss in portfolio_losses.nlargest(10).items():
        contributions = {
            symbol: float(
                instrument_losses.loc[date, symbol]
            )
            for symbol in symbols
        }

        worst_scenarios.append({
            "historical_date": normalize_date(date),
            "portfolio_loss_usd": float(loss),
            "instrument_losses_usd": contributions,
            "is_es_tail": bool(tail_mask.loc[date]),
        })

    # ---------------------------------------------------------
    # 9. Enhanced tail-scenario diagnostics
    # ---------------------------------------------------------

    tail_scenario_diagnostics = (
        _build_tail_scenario_diagnostics(
            tail_losses=tail_losses,
            historical_returns=historical_returns,
            instrument_losses=instrument_losses,
            symbols=symbols,
        )
    )

    # Validate individual scenario loss reconciliation.

    for scenario in tail_scenario_diagnostics:
        if not np.isclose(
            scenario["reconciliation_difference_usd"],
            0.0,
            rtol=0.0,
            atol=0.01,
        ):
            raise ValueError(
                "Tail scenario loss reconciliation failed"
            )

    # ---------------------------------------------------------
    # 10. ES concentration diagnostics
    # ---------------------------------------------------------

    concentration_diagnostics = (
        _build_concentration_diagnostics(
            instrument_attribution=instrument_attribution,
            nav=nav,
        )
    )

    # ---------------------------------------------------------
    # 11. Leave-one-tail-scenario-out sensitivity
    # ---------------------------------------------------------

    if attribution_available:
        leave_one_out_sensitivity = (
            _build_leave_one_out_sensitivity(
                tail_instrument_losses=tail_instrument_losses,
                symbols=symbols,
            )
        )
    else:
        leave_one_out_sensitivity = {
            "status": "NOT_EVALUATED",
            "reason": (
                "Positive ES attribution is unavailable"
            ),
            "results": [],
        }

    # ---------------------------------------------------------
    # 12. Governance warnings
    # ---------------------------------------------------------

    governance_warnings = []

    if len(tail_losses) < 5:
        governance_warnings.append({
            "code": "SMALL_TAIL_SAMPLE",
            "severity": "WARNING",
            "message": (
                f"ES attribution is based on only "
                f"{len(tail_losses)} historical tail scenarios. "
                "Results may be sensitive to individual observations."
            ),
        })

    if not attribution_available:
        governance_warnings.append({
            "code": "ES_ATTRIBUTION_UNAVAILABLE",
            "severity": "WARNING",
            "message": (
                "Positive ES attribution is unavailable. "
                "Risk concentration ratios and leave-one-out "
                "sensitivity cannot be evaluated."
            ),
        })

    if leave_one_out_sensitivity["status"] == "SUCCESS":
        sensitivity_range = (
            leave_one_out_sensitivity["range_usd"]
        )

        if calculated_es > 0:
            relative_range = (
                sensitivity_range / calculated_es
            )

            if relative_range > 0.20:
                governance_warnings.append({
                    "code": "HIGH_TAIL_SCENARIO_SENSITIVITY",
                    "severity": "WARNING",
                    "message": (
                        "The range of leave-one-out conditional "
                        "mean losses exceeds 20% of reported ES."
                    ),
                    "relative_range": float(relative_range),
                })

    # ---------------------------------------------------------
    # 13. Return structured investigation evidence
    # ---------------------------------------------------------

    return {
        "tool": "get_var_attribution",
        "portfolio_id": portfolio_id,
        "as_of_date": normalize_date(as_of),
        "status": "SUCCESS",

        "summary": {
            "var_99_usd": calculated_var,
            "es_99_usd": calculated_es,
            "var_99_pct_nav": calculated_var / nav,
            "es_99_pct_nav": calculated_es / nav,
            "historical_observations": len(
                historical_returns
            ),
            "tail_scenario_count": len(tail_losses),
            "attribution_available": attribution_available,
        },

        "instrument_attribution": instrument_attribution,

        "asset_class_attribution": asset_class_attribution,

        "worst_historical_scenarios": worst_scenarios,

        # Enhanced investigation outputs
        "tail_scenario_diagnostics": (
            tail_scenario_diagnostics
        ),

        "concentration_diagnostics": (
            concentration_diagnostics
        ),

        "leave_one_out_sensitivity": (
            leave_one_out_sensitivity
        ),

        "governance_warnings": governance_warnings,

        "reconciliation": {
            "reported_var_usd": reported_var,
            "reconstructed_var_usd": calculated_var,
            "var_difference_usd": (
                calculated_var - reported_var
            ),
            "reported_es_usd": reported_es,
            "reconstructed_es_usd": calculated_es,
            "es_difference_usd": (
                calculated_es - reported_es
            ),
            "sum_es_contributions_usd": (
                total_contribution
            ),
            "status": "PASS",
        },

        "methodology": {
            "model": "Historical Simulation",
            "holding_period": "1 trading day",
            "confidence_level": CONFIDENCE_LEVEL,
            "window": VAR_WINDOW,

            "scenario_positions": (
                "Current closing market values"
            ),

            "scenario_returns": (
                "252 historical ETF daily returns "
                "ending before the as-of date"
            ),

            "var_quantile": (
                "99th percentile of portfolio scenario "
                "losses, linear interpolation, floored at zero"
            ),

            "es_definition": (
                "Mean of portfolio losses greater than "
                "or equal to the calculated VaR threshold"
            ),

            "es_attribution": (
                "Average instrument loss over identical "
                "portfolio ES-tail scenarios"
            ),

            "concentration_diagnostics": (
                "Compare each instrument's NAV weight "
                "with its share of portfolio ES"
            ),

            "leave_one_out_sensitivity": (
                "Remove one selected portfolio ES-tail "
                "scenario at a time and recalculate the "
                "conditional mean loss and contributions. "
                "This is not a re-estimated VaR or ES."
            ),
        },

        "data_sources": [
            str(positions_path),
            str(prices_path),
            str(metrics_path),
        ],

        "limitations": [
            "ES contributions are not VaR contributions.",

            "Historical scenarios assume unchanged "
            "current holdings.",

            "Historical ETF returns are proxies "
            "for market risk factors.",

            "The 252-day historical window may not "
            "include earlier extreme market events.",

            "Tail contributions can be negative when "
            "an instrument offsets portfolio losses.",

            "Linear quantile interpolation and "
            "threshold-based ES may differ from "
            "fractional-tail ES estimators.",

            "Leave-one-out sensitivity is conditional "
            "on the selected ES-tail scenarios and "
            "does not re-estimate the historical VaR/ES model.",

            "The 20% tail-sensitivity warning threshold "
            "is illustrative, not a regulatory limit.",
        ],
    }
