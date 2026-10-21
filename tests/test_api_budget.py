
from decimal import Decimal

import pytest

from src.agent.api_config import (
    APIConfigurationError,
    OpenAIConfig,
)
from src.agent.budget import (
    AgentBudget,
    BudgetExceededError,
)


def test_api_disabled_by_default():
    config = OpenAIConfig.from_env({})
    assert config.enabled is False


def test_enabled_requires_key():
    with pytest.raises(APIConfigurationError):
        OpenAIConfig.from_env({
            "OPENAI_ENABLE_API": "true",
            "OPENAI_MODEL": "test-model",
        })


def test_enabled_requires_model():
    with pytest.raises(APIConfigurationError):
        OpenAIConfig.from_env({
            "OPENAI_ENABLE_API": "true",
            "OPENAI_API_KEY": "test-secret",
        })


def test_invalid_enable_flag_rejected():
    with pytest.raises(APIConfigurationError):
        OpenAIConfig.from_env({
            "OPENAI_ENABLE_API": "yes",
        })


def test_config_redacts_key():
    config = OpenAIConfig.from_env({
        "OPENAI_ENABLE_API": "true",
        "OPENAI_MODEL": "test-model",
        "OPENAI_API_KEY": "test-secret",
    })

    assert "test-secret" not in repr(config)


def test_disabled_config_blocks_live_use():
    config = OpenAIConfig.from_env({})

    with pytest.raises(APIConfigurationError):
        config.require_enabled()


def test_token_cost_calculation():
    config = OpenAIConfig.from_env({
        "OPENAI_INPUT_PRICE_PER_MILLION": "2",
        "OPENAI_OUTPUT_PRICE_PER_MILLION": "8",
    })

    cost = config.estimate_cost(1000, 500)

    assert cost == Decimal("0.006")


def test_invalid_price_rejected():
    with pytest.raises(APIConfigurationError):
        OpenAIConfig.from_env({
            "OPENAI_INPUT_PRICE_PER_MILLION": "-1",
        })


def test_reservation_counts_toward_call_limit():
    budget = AgentBudget(max_llm_calls=1)
    budget.reserve_llm_call("0.10")

    with pytest.raises(BudgetExceededError):
        budget.reserve_llm_call("0.01")


def test_reservation_counts_toward_cost_limit():
    budget = AgentBudget(
        max_estimated_cost_usd=Decimal("0.25")
    )

    budget.reserve_llm_call("0.20")

    with pytest.raises(BudgetExceededError):
        budget.reserve_llm_call("0.10")


def test_settlement_reconciles_cost():
    budget = AgentBudget()

    reservation = budget.reserve_llm_call("0.10")
    result = budget.settle_llm_call(
        reservation, "0.07"
    )

    assert budget.llm_calls == 1
    assert budget.estimated_cost_usd == Decimal("0.07")
    assert result["over_reservation"] is False
    assert budget.snapshot()["active_reservations"] == 0


def test_overrun_is_visible():
    budget = AgentBudget(
        max_estimated_cost_usd=Decimal("0.25")
    )

    reservation = budget.reserve_llm_call("0.20")
    result = budget.settle_llm_call(
        reservation, "0.30"
    )

    assert result["over_budget"] is True

    with pytest.raises(BudgetExceededError):
        budget.reserve_llm_call("0.01")


def test_cancel_releases_reservation():
    budget = AgentBudget(max_llm_calls=1)

    reservation = budget.reserve_llm_call("0.10")
    budget.cancel_llm_reservation(reservation)

    assert budget.snapshot()["active_reservations"] == 0
    budget.reserve_llm_call("0.10")


def test_duplicate_settlement_rejected():
    budget = AgentBudget()

    reservation = budget.reserve_llm_call("0.10")
    budget.settle_llm_call(reservation, "0.05")

    with pytest.raises(ValueError):
        budget.settle_llm_call(reservation, "0.05")


def test_negative_actual_cost_rejected():
    budget = AgentBudget()
    reservation = budget.reserve_llm_call("0.10")

    with pytest.raises(ValueError):
        budget.settle_llm_call(reservation, "-0.01")

    assert budget.snapshot()["active_reservations"] == 1
