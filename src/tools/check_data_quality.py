
from pathlib import Path

import numpy as np
import pandas as pd

from src.tools.common import normalize_date, read_csv


def _issue(code, severity, message, **evidence):
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "evidence": evidence,
    }


def _numeric(series):
    return pd.to_numeric(series, errors="coerce")


def check_data_quality(
    portfolio_id,
    as_of_date,
    lookback_days=252,
    prices_path="data/sample/adjusted_prices.csv",
    positions_path="data/sample/daily_positions.csv",
    portfolio_path="data/sample/daily_portfolio.csv",
):
    """
    Perform point-in-time, read-only data quality checks.

    Historical coverage is measured in daily return
    observations, not calendar days.

    Issues are returned as structured evidence.
    Source data is never modified.
    """
    as_of = pd.Timestamp(as_of_date)

    if pd.isna(as_of) or as_of != as_of.normalize():
        raise ValueError("Invalid as_of_date")

    if as_of.tzinfo is not None:
        raise ValueError("as_of_date must be timezone-naive")

    if (
        isinstance(lookback_days, bool)
        or not isinstance(lookback_days, int)
        or lookback_days < 1
    ):
        raise ValueError(
            "lookback_days must be a positive integer"
        )

    issues = []
    checks = []

    def record(name, new_issues, details=None):
        issues.extend(new_issues)

        severities = {
            item["severity"]
            for item in new_issues
        }

        status = (
            "FAIL" if "ERROR" in severities
            else "WARNING" if severities
            else "PASS"
        )

        checks.append({
            "check": name,
            "status": status,
            "issue_count": len(new_issues),
            "details": details or {},
        })

    sources = {
        "prices": str(Path(prices_path)),
        "positions": str(Path(positions_path)),
        "portfolio": str(Path(portfolio_path)),
    }

    required = {
        "prices": {"date"},
        "positions": {
            "date", "portfolio_id", "instrument_id",
            "quantity", "adjusted_price",
            "market_value_usd",
        },
        "portfolio": {
            "date", "portfolio_id", "nav_usd",
            "cash_usd", "daily_pnl_usd",
        },
    }

    frames = {}

    for name, path in sources.items():
        try:
            frame = read_csv(path)
            missing = required[name] - set(frame.columns)

            if missing:
                raise ValueError(
                    f"Missing required columns: {sorted(missing)}"
                )

            frame = frame.copy()
            frame["date"] = pd.to_datetime(
                frame["date"],
                errors="raise",
            )

            if frame["date"].dt.tz is not None:
                raise ValueError(
                    "Timezone-aware dates are unsupported"
                )

            if not (
                frame["date"]
                == frame["date"].dt.normalize()
            ).all():
                raise ValueError(
                    "Dates must not contain time components"
                )

            frames[name] = frame.loc[
                frame["date"] <= as_of
            ].copy()

            record(
                f"{name}_source",
                [],
                {"path": path},
            )

        except (OSError, ValueError, KeyError) as exc:
            record(
                f"{name}_source",
                [_issue(
                    "SOURCE_UNAVAILABLE",
                    "ERROR",
                    f"{name}: {exc}",
                    path=path,
                )],
            )

    if len(frames) != 3:
        return _finalize(
            portfolio_id,
            as_of,
            checks,
            issues,
            sources,
            coverage={},
            reconciliation={},
        )

    prices = frames["prices"]
    positions = frames["positions"]
    portfolio = frames["portfolio"]

    # --------------------------------------------------
    # 1. Portfolio snapshot
    # --------------------------------------------------
    snapshot_issues = []

    portfolio_rows = portfolio.loc[
        portfolio["portfolio_id"] == portfolio_id
    ].sort_values("date")

    snapshot = portfolio_rows.loc[
        portfolio_rows["date"] == as_of
    ]

    if len(snapshot) != 1:
        snapshot_issues.append(_issue(
            "INVALID_PORTFOLIO_SNAPSHOT",
            "ERROR",
            "Exactly one portfolio snapshot is required",
            count=len(snapshot),
        ))

    record(
        "portfolio_snapshot",
        snapshot_issues,
        {"snapshot_count": len(snapshot)},
    )

    # --------------------------------------------------
    # 2. Current positions
    # --------------------------------------------------
    position_issues = []

    holdings = positions.loc[
        (positions["portfolio_id"] == portfolio_id)
        & (positions["date"] == as_of)
    ].copy()

    if holdings.empty:
        position_issues.append(_issue(
            "MISSING_POSITIONS",
            "ERROR",
            "No current positions found",
        ))

    if holdings["instrument_id"].isna().any():
        position_issues.append(_issue(
            "MISSING_INSTRUMENT_ID",
            "ERROR",
            "One or more positions have no instrument ID",
        ))

    duplicates = holdings[
        "instrument_id"
    ].duplicated().sum()

    if duplicates:
        position_issues.append(_issue(
            "DUPLICATE_POSITIONS",
            "ERROR",
            "Duplicate instrument positions detected",
            duplicate_count=int(duplicates),
        ))

    for column in [
        "quantity",
        "adjusted_price",
        "market_value_usd",
    ]:
        holdings[column] = _numeric(
            holdings[column]
        )

        invalid = ~np.isfinite(
            holdings[column].to_numpy(dtype=float)
        )

        if invalid.any():
            position_issues.append(_issue(
                "INVALID_POSITION_VALUE",
                "ERROR",
                f"Invalid {column} in positions",
                column=column,
                invalid_count=int(invalid.sum()),
            ))

    if (
        holdings["adjusted_price"].notna()
        & (holdings["adjusted_price"] <= 0)
    ).any():
        position_issues.append(_issue(
            "NON_POSITIVE_POSITION_PRICE",
            "ERROR",
            "Position adjusted prices must be positive",
        ))

    valid_rows = holdings[
        [
            "quantity",
            "adjusted_price",
            "market_value_usd",
        ]
    ].notna().all(axis=1)

    if valid_rows.any():
        valid = holdings.loc[valid_rows]

        expected = (
            valid["quantity"]
            * valid["adjusted_price"]
        )

        actual = valid["market_value_usd"]

        tolerance = np.maximum(
            0.01,
            np.abs(actual) * 1e-8,
        )

        mismatch = (
            np.abs(expected - actual) > tolerance
        )

        if mismatch.any():
            position_issues.append(_issue(
                "POSITION_VALUATION_MISMATCH",
                "ERROR",
                "Quantity times price does not "
                "reconcile to market value",
                mismatch_count=int(mismatch.sum()),
            ))

    record(
        "position_integrity",
        position_issues,
        {"position_count": len(holdings)},
    )

    # --------------------------------------------------
    # 3. Price history and coverage
    # --------------------------------------------------
    price_issues = []
    coverage = {}

    duplicate_dates = int(
        prices["date"].duplicated().sum()
    )

    if duplicate_dates:
        price_issues.append(_issue(
            "DUPLICATE_PRICE_DATES",
            "ERROR",
            "Duplicate price dates detected",
            duplicate_count=duplicate_dates,
        ))

    price_columns = set(prices.columns) - {"date"}

    instruments = sorted(
        holdings["instrument_id"].dropna().unique()
    )

    missing_symbols = sorted(
        set(instruments) - price_columns
    )

    if missing_symbols:
        price_issues.append(_issue(
            "MISSING_PRICE_COLUMNS",
            "ERROR",
            "Price columns missing for current holdings",
            instruments=missing_symbols,
        ))

    if as_of not in set(prices["date"]):
        price_issues.append(_issue(
            "MISSING_AS_OF_PRICES",
            "ERROR",
            "No adjusted-price row for as-of date",
        ))

    available_symbols = [
        symbol
        for symbol in instruments
        if symbol in prices.columns
    ]

    history = prices.sort_values(
        "date"
    ).set_index("date")

    # The VaR engine uses the previous 252 returns,
    # excluding the as-of-date return.
    prior_history = history.loc[
        history.index < as_of
    ]

    for symbol in available_symbols:
        series = _numeric(history[symbol])

        invalid = (
            ~np.isfinite(series.to_numpy(dtype=float))
            | (series.to_numpy(dtype=float) <= 0)
        )

        if invalid.any():
            price_issues.append(_issue(
                "INVALID_ADJUSTED_PRICE",
                "ERROR",
                f"Invalid adjusted price for {symbol}",
                instrument_id=symbol,
                invalid_count=int(invalid.sum()),
            ))

        # Return observations require two valid,
        # consecutive available trading-date prices.
        prior = _numeric(
            prior_history[symbol]
        )

        valid_price = (
            prior.notna()
            & np.isfinite(prior)
            & (prior > 0)
        )

        valid_return = (
            valid_price
            & valid_price.shift(1, fill_value=False)
        )

        # Restrict to the latest requested number
        # of return slots, without backfilling gaps.
        recent = valid_return.iloc[
            -lookback_days:
        ]

        available = int(recent.sum())

        coverage[symbol] = {
            "available_return_observations": available,
            "required_return_observations": lookback_days,
            "status": (
                "PASS"
                if available == lookback_days
                else "NOT_EVALUATED"
            ),
        }

        if available < lookback_days:
            price_issues.append(_issue(
                "INSUFFICIENT_RETURN_HISTORY",
                "WARNING",
                f"Insufficient return history for {symbol}",
                instrument_id=symbol,
                available=available,
                required=lookback_days,
            ))

    record(
        "price_integrity_and_coverage",
        price_issues,
        {
            "instrument_count": len(instruments),
            "required_return_observations": lookback_days,
        },
    )

    # --------------------------------------------------
    # 4. NAV reconciliation
    # --------------------------------------------------
    nav_issues = []
    reconciliation = {}

    if len(snapshot) == 1 and not holdings.empty:
        row = snapshot.iloc[0]

        nav = pd.to_numeric(
            pd.Series([row["nav_usd"]]),
            errors="coerce",
        ).iloc[0]

        cash = pd.to_numeric(
            pd.Series([row["cash_usd"]]),
            errors="coerce",
        ).iloc[0]

        values = holdings["market_value_usd"]

        if (
            np.isfinite(nav)
            and np.isfinite(cash)
            and np.isfinite(values).all()
            and nav > 0
        ):
            computed_nav = float(
                values.sum() + cash
            )

            difference = float(
                computed_nav - nav
            )

            tolerance = max(
                0.01,
                abs(nav) * 1e-8,
            )

            passed = abs(difference) <= tolerance

            reconciliation["nav"] = {
                "reported_nav_usd": float(nav),
                "computed_nav_usd": computed_nav,
                "difference_usd": difference,
                "status": "PASS" if passed else "FAIL",
            }

            if not passed:
                nav_issues.append(_issue(
                    "NAV_RECONCILIATION_FAILURE",
                    "ERROR",
                    "Positions plus cash do not equal NAV",
                    difference_usd=difference,
                ))
        else:
            nav_issues.append(_issue(
                "INVALID_NAV_INPUTS",
                "ERROR",
                "NAV reconciliation inputs are invalid",
            ))

    record("nav_reconciliation", nav_issues)

    # --------------------------------------------------
    # 5. Daily P&L reconciliation
    # --------------------------------------------------
    pnl_issues = []

    if len(snapshot) == 1:
        current = snapshot.iloc[0]

        previous = portfolio_rows.loc[
            portfolio_rows["date"] < as_of
        ].tail(1)

        if len(previous) == 1:
            prior = previous.iloc[0]

            current_nav = pd.to_numeric(
                pd.Series([current["nav_usd"]]),
                errors="coerce",
            ).iloc[0]

            previous_nav = pd.to_numeric(
                pd.Series([prior["nav_usd"]]),
                errors="coerce",
            ).iloc[0]

            reported_pnl = pd.to_numeric(
                pd.Series([current["daily_pnl_usd"]]),
                errors="coerce",
            ).iloc[0]

            if all(np.isfinite([
                current_nav,
                previous_nav,
                reported_pnl,
            ])):
                expected_pnl = (
                    current_nav - previous_nav
                )

                difference = float(
                    expected_pnl - reported_pnl
                )

                tolerance = max(
                    0.01,
                    abs(current_nav) * 1e-8,
                )

                passed = (
                    abs(difference) <= tolerance
                )

                reconciliation["daily_pnl"] = {
                    "expected_pnl_usd": float(
                        expected_pnl
                    ),
                    "reported_pnl_usd": float(
                        reported_pnl
                    ),
                    "difference_usd": difference,
                    "status": (
                        "PASS" if passed else "FAIL"
                    ),
                }

                if not passed:
                    pnl_issues.append(_issue(
                        "DAILY_PNL_RECONCILIATION_FAILURE",
                        "ERROR",
                        "Daily P&L does not reconcile "
                        "to the NAV change",
                        difference_usd=difference,
                    ))
            else:
                pnl_issues.append(_issue(
                    "INVALID_PNL_INPUTS",
                    "ERROR",
                    "Daily P&L inputs are invalid",
                ))
        else:
            pnl_issues.append(_issue(
                "MISSING_PREVIOUS_NAV",
                "WARNING",
                "No prior NAV snapshot for daily "
                "P&L reconciliation",
            ))

    record("daily_pnl_reconciliation", pnl_issues)

    return _finalize(
        portfolio_id,
        as_of,
        checks,
        issues,
        sources,
        coverage,
        reconciliation,
    )


def _finalize(
    portfolio_id,
    as_of,
    checks,
    issues,
    sources,
    coverage,
    reconciliation,
):
    severities = {
        issue["severity"]
        for issue in issues
    }

    overall_status = (
        "FAIL" if "ERROR" in severities
        else "WARNING" if severities
        else "PASS"
    )

    return {
        "tool": "check_data_quality",
        "portfolio_id": portfolio_id,
        "as_of_date": normalize_date(as_of),
        "overall_status": overall_status,
        "summary": {
            "check_count": len(checks),
            "issue_count": len(issues),
            "error_count": sum(
                issue["severity"] == "ERROR"
                for issue in issues
            ),
            "warning_count": sum(
                issue["severity"] == "WARNING"
                for issue in issues
            ),
        },
        "checks": checks,
        "issues": issues,
        "coverage": coverage,
        "reconciliation": reconciliation,
        "data_sources": sources,
        "methodology": {
            "point_in_time": (
                "Only source rows dated on or before "
                "the investigation date are inspected"
            ),
            "history_requirement": (
                "Previous N daily return observations "
                "excluding the as-of-date return"
            ),
            "nav_reconciliation": (
                "Sum of position market values plus "
                "cash equals reported NAV"
            ),
            "daily_pnl_reconciliation": (
                "Current NAV minus previous NAV "
                "equals reported daily P&L; assumes "
                "no external capital flows"
            ),
        },
        "limitations": [
            "Does not independently verify market prices "
            "against an external vendor.",
            "Does not classify extreme market returns "
            "as erroneous observations.",
            "Does not validate transaction-level "
            "cash and quantity movements.",
            "PASS means no issues found within "
            "the checks performed.",
        ],
    }
