
import numpy as np
import pandas as pd
import pytest

from src.risk.calculator import calculate_risk_metrics


@pytest.fixture(scope="module")
def metrics():
    return calculate_risk_metrics()


def test_observation_count(metrics):
    assert len(metrics) == 1695


def test_var_warmup_period(metrics):
    assert metrics["var_99_usd"].iloc[:253].isna().all()
    assert metrics["var_99_usd"].iloc[253:].notna().all()


def test_es_not_less_than_var(metrics):
    valid = metrics.dropna(subset=["var_99_usd", "es_99_usd"])
    assert (
        valid["es_99_usd"] >= valid["var_99_usd"] - 1e-8
    ).all()


def test_var_and_es_nonnegative(metrics):
    assert (metrics["var_99_usd"].dropna() >= 0).all()
    assert (metrics["es_99_usd"].dropna() >= 0).all()


def test_asset_exposure_reconciliation(metrics):
    exposure = (
        metrics["equity_exposure_usd"]
        + metrics["rates_exposure_usd"]
        + metrics["credit_exposure_usd"]
        + metrics["cash_usd"]
    )

    assert np.allclose(
        exposure,
        metrics["nav_usd"],
        rtol=1e-9,
        atol=0.01,
    )


def test_largest_position_weight(metrics):
    weights = metrics["largest_position_weight"]
    assert weights.between(0, 1).all()


def test_drawdown_nonpositive(metrics):
    assert (metrics["drawdown"] <= 1e-10).all()


def test_daily_pnl_reconciliation(metrics):
    expected = metrics["nav_usd"].diff().fillna(0)
    assert np.allclose(
        expected,
        metrics["daily_pnl_usd"],
        rtol=1e-9,
        atol=0.01,
    )
