
"""Tests for P&L monitoring metrics."""

import numpy as np
import pandas as pd

from src.risk.calculator import calculate_risk_metrics


def test_pnl_monitoring_uses_previous_day_var():
    metrics = calculate_risk_metrics()

    assert "prior_var_99_usd" in metrics.columns
    assert "pnl_loss_to_prior_var" in metrics.columns

    expected_prior_var = metrics["var_99_usd"].shift(1)

    pd.testing.assert_series_equal(
        metrics["prior_var_99_usd"],
        expected_prior_var,
        check_names=False,
    )


def test_pnl_ratio_matches_realized_loss():
    metrics = calculate_risk_metrics()

    valid = (
        metrics["prior_var_99_usd"].notna()
        & np.isfinite(metrics["prior_var_99_usd"])
        & (metrics["prior_var_99_usd"] > 0)
    )

    expected_loss = (
        -metrics.loc[valid, "daily_pnl_usd"]
    ).clip(lower=0)

    expected_ratio = (
        expected_loss
        / metrics.loc[valid, "prior_var_99_usd"]
    )

    np.testing.assert_allclose(
        metrics.loc[
            valid, "pnl_loss_to_prior_var"
        ].to_numpy(),
        expected_ratio.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_missing_prior_var_is_not_evaluated():
    metrics = calculate_risk_metrics()

    missing = metrics["prior_var_99_usd"].isna()

    assert missing.any()

    assert metrics.loc[
        missing, "pnl_loss_to_prior_var"
    ].isna().all()


def test_pnl_ratio_nonnegative():
    metrics = calculate_risk_metrics()

    values = metrics[
        "pnl_loss_to_prior_var"
    ].dropna()

    assert (values >= 0).all()


def test_profitable_days_have_zero_loss_ratio():
    metrics = calculate_risk_metrics()

    profitable = (
        (metrics["daily_pnl_usd"] > 0)
        & metrics["pnl_loss_to_prior_var"].notna()
    )

    assert profitable.any()

    assert (
        metrics.loc[
            profitable, "pnl_loss_to_prior_var"
        ] == 0
    ).all()
