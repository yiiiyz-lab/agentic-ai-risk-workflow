
import numpy as np
import pandas as pd

from src.risk.calculator import ASSET_GROUPS
from src.tools.common import read_csv, normalize_date


TRADING_DAYS = 252
DEFAULT_WINDOWS = (20, 63, 252)
MIN_CORRELATION_OBSERVATIONS = 2


def _finite_float(value):
    """Return a JSON-safe float or None."""
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _validate_windows(windows):
    """Validate and deduplicate requested lookback windows."""
    if not windows:
        raise ValueError(
            "At least one lookback window is required"
        )

    cleaned = []

    for window in windows:
        if (
            isinstance(window, (bool, np.bool_))
            or not isinstance(window, (int, np.integer))
            or window < MIN_CORRELATION_OBSERVATIONS
        ):
            raise ValueError(
                "Lookback windows must be integers >= 2"
            )

        window = int(window)

        if window not in cleaned:
            cleaned.append(window)

    return cleaned


def _prepare_prices(prices_path, symbols):
    """Load and validate historical adjusted ETF prices."""
    prices = read_csv(
        prices_path,
        required_columns={"date", *symbols},
    )

    prices["date"] = pd.to_datetime(
        prices["date"],
        errors="raise",
    )

    if prices["date"].duplicated().any():
        raise ValueError("Duplicate price dates")

    prices = prices.sort_values("date").set_index("date")

    values = prices[symbols].apply(
        pd.to_numeric,
        errors="raise",
    )

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Non-finite adjusted prices")

    if (values <= 0).any().any():
        raise ValueError(
            "Adjusted prices must be positive"
        )

    return values


def _window_statistics(returns, symbols, window):
    """
    Calculate ETF return statistics using exactly
    `window` daily return observations.
    """
    selected = returns.tail(window)

    if len(selected) < window:
        return {
            "window": window,
            "status": "NOT_EVALUATED",
            "available_observations": len(selected),
            "required_observations": window,
            "reason_code": "INSUFFICIENT_HISTORY",
        }

    instruments = []

    for symbol in symbols:
        series = selected[symbol]

        cumulative_return = float(
            (1.0 + series).prod() - 1.0
        )

        annualized_volatility = float(
            series.std(ddof=1)
            * np.sqrt(TRADING_DAYS)
        )

        instruments.append({
            "instrument_id": symbol,
            "asset_class": ASSET_GROUPS[symbol],
            "cumulative_return": cumulative_return,
            "mean_daily_return": float(
                series.mean()
            ),
            "annualized_volatility": (
                annualized_volatility
            ),
            "minimum_daily_return": float(
                series.min()
            ),
            "maximum_daily_return": float(
                series.max()
            ),
        })

    correlations = selected.corr()

    correlation_matrix = {}

    for left in symbols:
        correlation_matrix[left] = {
            right: _finite_float(
                correlations.loc[left, right]
            )
            for right in symbols
        }

    return {
        "window": window,
        "status": "SUCCESS",
        "available_observations": window,
        "instruments": instruments,
        "correlation_matrix": correlation_matrix,
    }


def _volatility_change(returns, symbols):
    """
    Compare the latest 20 return observations
    against the preceding 20 observations.
    """
    if len(returns) < 40:
        return {
            "status": "NOT_EVALUATED",
            "reason_code": "INSUFFICIENT_HISTORY",
            "available_observations": len(returns),
            "required_observations": 40,
        }

    previous = returns.iloc[-40:-20]
    recent = returns.iloc[-20:]

    diagnostics = []

    for symbol in symbols:
        previous_vol = float(
            previous[symbol].std(ddof=1)
            * np.sqrt(TRADING_DAYS)
        )

        recent_vol = float(
            recent[symbol].std(ddof=1)
            * np.sqrt(TRADING_DAYS)
        )

        diagnostics.append({
            "instrument_id": symbol,
            "previous_20d_annualized_volatility": (
                previous_vol
            ),
            "recent_20d_annualized_volatility": (
                recent_vol
            ),
            "absolute_change": (
                recent_vol - previous_vol
            ),
            "relative_change": (
                (recent_vol / previous_vol) - 1.0
                if previous_vol > 0
                else None
            ),
        })

    return {
        "status": "SUCCESS",
        "instruments": diagnostics,
    }


def _equity_rates_correlations(window_results):
    """
    Extract equity-versus-Treasury ETF correlations
    from successfully evaluated windows.
    """
    equity_symbols = [
        symbol
        for symbol, group in ASSET_GROUPS.items()
        if group == "Equity"
    ]

    rates_symbols = [
        symbol
        for symbol, group in ASSET_GROUPS.items()
        if group == "Rates"
    ]

    output = []

    for result in window_results:
        if result["status"] != "SUCCESS":
            continue

        matrix = result["correlation_matrix"]
        pairs = []

        for equity in equity_symbols:
            for rates in rates_symbols:
                pairs.append({
                    "equity_instrument": equity,
                    "rates_instrument": rates,
                    "correlation": (
                        matrix[equity][rates]
                    ),
                })

        output.append({
            "window": result["window"],
            "pairs": pairs,
        })

    return output


def _scenario_diagnostics(
    returns,
    as_of,
    scenario_dates,
    symbols,
):
    """
    Inspect historical ETF returns on selected dates.

    Scenario dates must be strictly earlier than
    the investigation as-of date.
    """
    if scenario_dates is None:
        return []

    results = []
    seen = set()

    for date_value in scenario_dates:
        date = pd.Timestamp(date_value)

        if date >= as_of:
            raise ValueError(
                "Scenario dates must precede the as-of date"
            )

        if date in seen:
            continue

        seen.add(date)

        if date not in returns.index:
            raise ValueError(
                "Scenario date not found: "
                f"{normalize_date(date)}"
            )

        row = returns.loc[date]

        if row.isna().any():
            raise ValueError(
                "Returns unavailable for scenario date: "
                f"{normalize_date(date)}"
            )

        scenario_returns = {
            symbol: float(row[symbol])
            for symbol in symbols
        }

        ranked = sorted(
            scenario_returns.items(),
            key=lambda item: item[1],
        )

        results.append({
            "historical_date": normalize_date(date),
            "instrument_returns": scenario_returns,
            "largest_declines": [
                {
                    "instrument_id": symbol,
                    "daily_return": value,
                }
                for symbol, value in ranked[:3]
            ],
            "largest_gains": [
                {
                    "instrument_id": symbol,
                    "daily_return": value,
                }
                for symbol, value in reversed(
                    ranked[-3:]
                )
            ],
        })

    return results


def _detect_risk_signals(
    window_results,
    volatility_change,
    scenario_results,
):
    """
    Detect descriptive market-risk signals.

    These are illustrative analytical thresholds,
    not formal risk limits or policy breaches.
    """
    signals = []

    successful = {
        item["window"]: item
        for item in window_results
        if item["status"] == "SUCCESS"
    }

    # --------------------------------------------------
    # 1. Equity / Treasury ETF correlation increase
    # --------------------------------------------------
    # Compare 20-day vs 252-day correlations.
    #
    # Illustrative trigger:
    # - Recent correlation >= 0.50
    # - Increase >= 0.20 correlation points
    #
    # These windows overlap. The signal is descriptive,
    # not evidence of a structural correlation break.

    if 20 in successful and 252 in successful:
        short_matrix = successful[20][
            "correlation_matrix"
        ]

        long_matrix = successful[252][
            "correlation_matrix"
        ]

        equity_symbols = [
            symbol
            for symbol, group in ASSET_GROUPS.items()
            if group == "Equity"
        ]

        rates_symbols = [
            symbol
            for symbol, group in ASSET_GROUPS.items()
            if group == "Rates"
        ]

        for equity in equity_symbols:
            for rates in rates_symbols:
                short_corr = short_matrix[equity][rates]
                long_corr = long_matrix[equity][rates]

                if (
                    short_corr is None
                    or long_corr is None
                ):
                    continue

                change = short_corr - long_corr

                if (
                    short_corr >= 0.50
                    and change >= 0.20
                ):
                    signals.append({
                        "code": (
                            "EQUITY_RATES_CORRELATION_INCREASE"
                        ),
                        "category": "CORRELATION",
                        "severity": "WATCH",
                        "instruments": [
                            equity,
                            rates,
                        ],
                        "recent_20d_correlation": (
                            short_corr
                        ),
                        "historical_252d_correlation": (
                            long_corr
                        ),
                        "correlation_change": float(
                            change
                        ),
                        "interpretation": (
                            "Recent equity/Treasury ETF "
                            "return correlation is materially "
                            "higher than its longer-window "
                            "estimate, potentially reducing "
                            "diversification."
                        ),
                    })

    # --------------------------------------------------
    # 2. Volatility increase
    # --------------------------------------------------
    # Illustrative trigger:
    # Latest 20-day annualized volatility is at least
    # 25% higher than the preceding 20-day estimate.

    if volatility_change["status"] == "SUCCESS":
        for item in volatility_change["instruments"]:
            relative_change = item["relative_change"]

            if (
                relative_change is not None
                and relative_change >= 0.25
            ):
                signals.append({
                    "code": "VOLATILITY_INCREASE",
                    "category": "VOLATILITY",
                    "severity": "WATCH",
                    "instruments": [
                        item["instrument_id"]
                    ],
                    "previous_20d_volatility": (
                        item[
                            "previous_20d_annualized_volatility"
                        ]
                    ),
                    "recent_20d_volatility": (
                        item[
                            "recent_20d_annualized_volatility"
                        ]
                    ),
                    "relative_change": (
                        relative_change
                    ),
                    "interpretation": (
                        "Recent annualized ETF return "
                        "volatility is at least 25% higher "
                        "than in the preceding 20 "
                        "trading days."
                    ),
                })

    # --------------------------------------------------
    # 3. Cross-asset joint losses
    # --------------------------------------------------
    # Detect selected historical dates when both equity
    # ETFs and both Treasury ETFs declined.
    #
    # This is a historical scenario diagnostic, not a
    # prediction of future market behavior.

    for scenario in scenario_results:
        returns = scenario["instrument_returns"]

        equity_declines = all(
            returns[symbol] < 0
            for symbol in ("SPY", "QQQ")
        )

        rates_declines = all(
            returns[symbol] < 0
            for symbol in ("IEF", "TLT")
        )

        if equity_declines and rates_declines:
            signals.append({
                "code": "CROSS_ASSET_JOINT_LOSS",
                "category": "SCENARIO",
                "severity": "WATCH",
                "historical_date": (
                    scenario["historical_date"]
                ),
                "instruments": [
                    "SPY",
                    "QQQ",
                    "IEF",
                    "TLT",
                ],
                "instrument_returns": {
                    symbol: returns[symbol]
                    for symbol in (
                        "SPY",
                        "QQQ",
                        "IEF",
                        "TLT",
                    )
                },
                "interpretation": (
                    "Both equity ETFs and both Treasury "
                    "ETFs declined on this historical "
                    "date. Treasury ETF exposures did "
                    "not provide a positive return "
                    "offset in this scenario."
                ),
            })

    return signals


def analyze_risk_factors(
    portfolio_id,
    as_of_date,
    windows=DEFAULT_WINDOWS,
    scenario_dates=None,
    prices_path="data/sample/adjusted_prices.csv",
    positions_path="data/sample/daily_positions.csv",
):
    """
    Analyze historical ETF market-risk proxies.

    Provides:
    - Historical returns
    - Annualized volatility
    - Rolling volatility changes
    - Correlation matrices
    - Equity/rates relationships
    - Historical scenario diagnostics
    - Market-risk signals
    - Governance warnings

    All observations are restricted to information
    available on or before the as-of date.
    """

    symbols = list(ASSET_GROUPS)
    as_of = pd.Timestamp(as_of_date)
    windows = _validate_windows(windows)

    # --------------------------------------------------
    # 1. Validate portfolio and as-of date
    # --------------------------------------------------

    positions = read_csv(
        positions_path,
        required_columns={
            "date",
            "portfolio_id",
            "instrument_id",
        },
    )

    positions["date"] = pd.to_datetime(
        positions["date"],
        errors="raise",
    )

    snapshot = positions.loc[
        (positions["portfolio_id"] == portfolio_id)
        & (positions["date"] == as_of)
    ]

    if snapshot.empty:
        raise ValueError(
            "Portfolio snapshot not found "
            "for requested date"
        )

    if snapshot["instrument_id"].duplicated().any():
        raise ValueError(
            "Duplicate current positions"
        )

    if set(snapshot["instrument_id"]) != set(symbols):
        raise ValueError(
            "Portfolio positions do not match "
            "ETF universe"
        )

    prices = _prepare_prices(
        prices_path,
        symbols,
    )

    if as_of not in prices.index:
        raise ValueError(
            "As-of date missing from prices"
        )

    # --------------------------------------------------
    # 2. Restrict prices to the as-of date
    # --------------------------------------------------

    historical_prices = prices.loc[
        prices.index <= as_of
    ]

    returns = historical_prices.pct_change(
        fill_method=None
    ).iloc[1:]

    if returns.isna().any().any():
        raise ValueError(
            "Missing historical returns"
        )

    if not np.isfinite(
        returns.to_numpy()
    ).all():
        raise ValueError(
            "Non-finite historical returns"
        )

    # --------------------------------------------------
    # 3. Calculate window statistics
    # --------------------------------------------------

    window_results = [
        _window_statistics(
            returns=returns,
            symbols=symbols,
            window=window,
        )
        for window in windows
    ]

    # --------------------------------------------------
    # 4. Calculate volatility changes
    # --------------------------------------------------

    volatility_change = _volatility_change(
        returns=returns,
        symbols=symbols,
    )

    # --------------------------------------------------
    # 5. Equity/rates relationships
    # --------------------------------------------------

    equity_rates_correlations = (
        _equity_rates_correlations(
            window_results
        )
    )

    # --------------------------------------------------
    # 6. Historical scenario diagnostics
    # --------------------------------------------------

    scenario_results = _scenario_diagnostics(
        returns=returns,
        as_of=as_of,
        scenario_dates=scenario_dates,
        symbols=symbols,
    )

    # --------------------------------------------------
    # 7. Market-risk signals
    # --------------------------------------------------

    risk_signals = _detect_risk_signals(
        window_results=window_results,
        volatility_change=volatility_change,
        scenario_results=scenario_results,
    )

    risk_signal_summary = {
        "total_signals": len(risk_signals),
        "correlation_signals": sum(
            signal["category"] == "CORRELATION"
            for signal in risk_signals
        ),
        "volatility_signals": sum(
            signal["category"] == "VOLATILITY"
            for signal in risk_signals
        ),
        "scenario_signals": sum(
            signal["category"] == "SCENARIO"
            for signal in risk_signals
        ),
    }

    # --------------------------------------------------
    # 8. Governance warnings
    # --------------------------------------------------

    governance_warnings = []

    unavailable_windows = [
        item["window"]
        for item in window_results
        if item["status"] == "NOT_EVALUATED"
    ]

    if unavailable_windows:
        governance_warnings.append({
            "code": "INSUFFICIENT_HISTORY",
            "severity": "WARNING",
            "message": (
                "Insufficient return history "
                f"for windows: {unavailable_windows}"
            ),
        })

    for item in window_results:
        if item["status"] != "SUCCESS":
            continue

        missing_correlations = [
            (left, right)
            for left, row in item[
                "correlation_matrix"
            ].items()
            for right, value in row.items()
            if value is None
        ]

        if missing_correlations:
            governance_warnings.append({
                "code": "UNDEFINED_CORRELATION",
                "severity": "WARNING",
                "window": item["window"],
                "message": (
                    "Some correlations are undefined, "
                    "potentially due to zero "
                    "return variance."
                ),
            })

    # --------------------------------------------------
    # 9. Return structured investigation evidence
    # --------------------------------------------------

    return {
        "tool": "analyze_risk_factors",
        "portfolio_id": portfolio_id,
        "as_of_date": normalize_date(as_of),
        "status": "SUCCESS",

        "summary": {
            "available_return_observations": (
                len(returns)
            ),
            "requested_windows": windows,
            "evaluated_windows": [
                item["window"]
                for item in window_results
                if item["status"] == "SUCCESS"
            ],
            "scenario_count": (
                len(scenario_results)
            ),
            "instrument_count": len(symbols),
        },

        "window_statistics": window_results,

        "volatility_change": volatility_change,

        "equity_rates_correlations": (
            equity_rates_correlations
        ),

        "historical_scenarios": (
            scenario_results
        ),

        "risk_signals": risk_signals,

        "risk_signal_summary": (
            risk_signal_summary
        ),

        "governance_warnings": (
            governance_warnings
        ),

        "methodology": {
            "return_type": (
                "Daily simple returns from "
                "adjusted ETF prices"
            ),

            "annualization_factor": TRADING_DAYS,

            "volatility": (
                "Sample standard deviation of "
                "daily returns multiplied by sqrt(252)"
            ),

            "correlation": (
                "Pearson correlation of "
                "daily ETF returns"
            ),

            "cumulative_return": (
                "Product of (1 + daily return) minus 1"
            ),

            "volatility_change": (
                "Most recent 20 daily returns compared "
                "with preceding 20 daily returns"
            ),

            "scenario_definition": (
                "Observed historical ETF daily returns "
                "on user-selected dates"
            ),

            "risk_signals": (
                "Illustrative analytical triggers for "
                "correlation increases, volatility "
                "increases, and historical joint losses. "
                "Not formal risk limits."
            ),
        },

        "data_sources": [
            str(prices_path),
            str(positions_path),
        ],

        "limitations": [
            "ETF returns are market-risk proxies, "
            "not direct interest-rate or "
            "credit-spread factors.",

            "IEF and TLT represent Treasury ETF "
            "exposure; their returns are not "
            "DV01 measurements.",

            "LQD and HYG returns reflect combined "
            "market effects and are not direct "
            "CS01 measurements.",

            "Historical correlations may change "
            "in future market conditions.",

            "Scenario returns are historical "
            "observations, not forecasts or "
            "causal explanations.",

            "Volatility and correlation estimates "
            "depend on the selected historical "
            "lookback window.",

            "Market-risk signals use illustrative "
            "analytical thresholds and are not "
            "formal policy breaches.",

            "The 20-day and 252-day correlation "
            "windows overlap; a difference does "
            "not establish a structural change.",

            "Joint-loss signals describe selected "
            "historical scenarios, not the "
            "frequency of future joint losses.",
        ],
    }
