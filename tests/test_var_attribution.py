
import json

import pytest

from src.tools.get_var_attribution import get_var_attribution


PORTFOLIO_ID = "PORT_001"
AS_OF_DATE = "2026-09-30"


@pytest.fixture(scope="module")
def result():
    """Calculate the latest VaR/ES attribution once per test module."""
    return get_var_attribution(
        PORTFOLIO_ID,
        AS_OF_DATE,
    )


# ============================================================
# 1. VaR / ES numerical reconciliation
# ============================================================

def test_var_reconciles(result):
    assert result["summary"]["var_99_usd"] == pytest.approx(
        3323239.904071,
        abs=0.10,
    )


def test_es_reconciles(result):
    assert result["summary"]["es_99_usd"] == pytest.approx(
        4240299.715092,
        abs=0.10,
    )


def test_es_instrument_contributions_add_up(result):
    total = sum(
        item["es_contribution_usd"]
        for item in result["instrument_attribution"]
    )

    assert total == pytest.approx(
        result["summary"]["es_99_usd"],
        abs=0.01,
    )


def test_es_asset_class_contributions_add_up(result):
    total = sum(
        item["es_contribution_usd"]
        for item in result["asset_class_attribution"]
    )

    assert total == pytest.approx(
        result["summary"]["es_99_usd"],
        abs=0.01,
    )


def test_historical_window(result):
    assert (
        result["summary"]["historical_observations"]
        == 252
    )


def test_tail_scenarios(result):
    assert result["summary"]["tail_scenario_count"] >= 1

    assert all(
        item["is_es_tail"]
        for item in result["worst_historical_scenarios"][
            :result["summary"]["tail_scenario_count"]
        ]
    )


def test_no_future_scenarios(result):
    assert all(
        item["historical_date"] < AS_OF_DATE
        for item in result["worst_historical_scenarios"]
    )


def test_reconciliation_status(result):
    assert result["reconciliation"]["status"] == "PASS"


def test_json_serializable(result):
    json.dumps(result, allow_nan=False)


def test_early_history_not_evaluated():
    early_result = get_var_attribution(
        PORTFOLIO_ID,
        "2020-06-01",
    )

    assert early_result["status"] == "NOT_EVALUATED"
    assert (
        early_result["reason_code"]
        == "INSUFFICIENT_HISTORY"
    )


def test_wrong_portfolio():
    with pytest.raises(ValueError):
        get_var_attribution(
            "UNKNOWN_PORTFOLIO",
            AS_OF_DATE,
        )


# ============================================================
# 2. Enhanced tail-scenario diagnostics
# ============================================================

def test_tail_diagnostics_count(result):
    assert len(result["tail_scenario_diagnostics"]) == (
        result["summary"]["tail_scenario_count"]
    )


def test_tail_scenarios_reconcile(result):
    for scenario in result["tail_scenario_diagnostics"]:
        total = sum(
            scenario["instrument_losses_usd"].values()
        )

        assert total == pytest.approx(
            scenario["portfolio_loss_usd"],
            abs=0.01,
        )

        assert scenario[
            "reconciliation_difference_usd"
        ] == pytest.approx(0.0, abs=0.01)


def test_tail_scenario_returns_available(result):
    expected_symbols = {
        "SPY", "QQQ", "IEF", "TLT", "LQD", "HYG"
    }

    for scenario in result["tail_scenario_diagnostics"]:
        assert set(
            scenario["instrument_returns"]
        ) == expected_symbols

        assert set(
            scenario["instrument_losses_usd"]
        ) == expected_symbols


def test_tail_scenarios_point_in_time(result):
    assert all(
        scenario["historical_date"] < AS_OF_DATE
        for scenario in result["tail_scenario_diagnostics"]
    )


# ============================================================
# 3. ES concentration diagnostics
# ============================================================

def test_concentration_weights(result):
    diagnostics = result["concentration_diagnostics"]

    total_weight = sum(
        item["portfolio_weight"]
        for item in diagnostics
    )

    assert total_weight == pytest.approx(
        1.0,
        abs=1e-8,
    )


def test_concentration_es_shares(result):
    diagnostics = result["concentration_diagnostics"]

    total_es_share = sum(
        item["es_contribution_share"]
        for item in diagnostics
    )

    assert total_es_share == pytest.approx(
        1.0,
        abs=1e-8,
    )


def test_qqq_tail_risk_concentration(result):
    qqq = next(
        item
        for item in result["concentration_diagnostics"]
        if item["instrument_id"] == "QQQ"
    )

    assert qqq["portfolio_weight"] == pytest.approx(
        0.4278605414,
        abs=1e-8,
    )

    assert qqq["es_contribution_share"] > 0.70

    assert qqq["es_to_weight_ratio"] > 1.5

    assert qqq["es_share_minus_weight"] > 0


# ============================================================
# 4. Leave-one-tail-scenario-out sensitivity
# ============================================================

def test_leave_one_out_count(result):
    sensitivity = result["leave_one_out_sensitivity"]

    assert sensitivity["status"] == "SUCCESS"

    assert len(sensitivity["results"]) == (
        result["summary"]["tail_scenario_count"]
    )


def test_leave_one_out_reconciliation(result):
    sensitivity = result["leave_one_out_sensitivity"]

    for item in sensitivity["results"]:
        total = sum(
            item["instrument_contributions_usd"].values()
        )

        assert total == pytest.approx(
            item["conditional_mean_loss_usd"],
            abs=0.01,
        )


# ============================================================
# 5. Governance and data validation
# ============================================================

def test_small_tail_sample_warning(result):
    warning_codes = {
        warning["code"]
        for warning in result["governance_warnings"]
    }

    assert "SMALL_TAIL_SAMPLE" in warning_codes


def test_unknown_portfolio_early_date():
    """
    Unknown portfolios should raise ValueError rather than
    incorrectly returning INSUFFICIENT_HISTORY.
    """
    with pytest.raises(ValueError):
        get_var_attribution(
            "UNKNOWN_PORTFOLIO",
            "2020-06-01",
        )


def test_enhanced_json_serializable(result):
    json.dumps(result, allow_nan=False)
