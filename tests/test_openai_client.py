
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.agent.api_config import (
    APIConfigurationError,
    OpenAIConfig,
)
from src.agent.budget import (
    AgentBudget,
    BudgetExceededError,
)
from src.agent.openai_client import (
    APIRequestUncertainError,
    OpenAIResponsesClient,
    build_live_client,
)


def config(enabled=True):
    return OpenAIConfig.from_env({
        "OPENAI_ENABLE_API": str(enabled).lower(),
        "OPENAI_MODEL": "test-model",
        "OPENAI_API_KEY": "test-secret",
        "OPENAI_INPUT_PRICE_PER_MILLION": "2",
        "OPENAI_OUTPUT_PRICE_PER_MILLION": "8",
    })


class FakeResponses:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeSDK:
    def __init__(self, response=None, error=None):
        self.responses = FakeResponses(response, error)


def mock_response(input_tokens=1000, output_tokens=100):
    return SimpleNamespace(
        id="resp_test_001",
        output=[{
            "type": "function_call",
            "call_id": "call_test_001",
            "name": "analyze_positions",
            "arguments": "{}",
        }],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def create_client(
    *,
    enabled=True,
    verified=True,
    sdk=None,
    budget=None,
):
    return OpenAIResponsesClient(
        config=config(enabled),
        budget=budget or AgentBudget(),
        sdk_client=sdk or FakeSDK(mock_response()),
        pricing_verified=verified,
    )


def send(client):
    return client.create_response(
        instructions="Investigate the exception.",
        input=[{
            "role": "user",
            "content": "Investigate equity exposure.",
        }],
        tools=[],
        max_input_tokens_estimate=2000,
    )


def test_disabled_api_blocks_request():
    sdk = FakeSDK(mock_response())
    client = create_client(enabled=False, sdk=sdk)

    with pytest.raises(APIConfigurationError):
        send(client)

    assert sdk.responses.calls == []


def test_unverified_pricing_blocks_request():
    sdk = FakeSDK(mock_response())
    client = create_client(verified=False, sdk=sdk)

    with pytest.raises(APIConfigurationError):
        send(client)

    assert sdk.responses.calls == []


def test_successful_response():
    sdk = FakeSDK(mock_response())
    budget = AgentBudget()
    client = create_client(sdk=sdk, budget=budget)

    result = send(client)

    assert result.response.response_id == "resp_test_001"
    assert result.response.tool_calls[0].name == "analyze_positions"
    assert budget.llm_calls == 1
    assert budget.snapshot()["active_reservations"] == 0


def test_actual_usage_cost():
    sdk = FakeSDK(mock_response(
        input_tokens=1000,
        output_tokens=100,
    ))
    client = create_client(sdk=sdk)

    result = send(client)

    assert result.estimated_actual_cost_usd == Decimal("0.0028")


def test_request_limits_are_forwarded():
    sdk = FakeSDK(mock_response())
    budget = AgentBudget(
        max_output_tokens_per_call=400
    )
    client = create_client(sdk=sdk, budget=budget)

    send(client)

    kwargs = sdk.responses.calls[0]

    assert kwargs["max_output_tokens"] == 400
    assert kwargs["parallel_tool_calls"] is False
    assert kwargs["store"] is False


def test_budget_exceeded_before_request():
    sdk = FakeSDK(mock_response())
    budget = AgentBudget(
        max_estimated_cost_usd=Decimal("0.000001")
    )
    client = create_client(sdk=sdk, budget=budget)

    with pytest.raises(BudgetExceededError):
        send(client)

    assert sdk.responses.calls == []


def test_uncertain_network_failure_keeps_reservation():
    sdk = FakeSDK(error=TimeoutError("simulated"))
    budget = AgentBudget(max_llm_calls=1)
    client = create_client(sdk=sdk, budget=budget)

    with pytest.raises(APIRequestUncertainError) as exc:
        send(client)

    assert exc.value.reservation_id
    assert budget.snapshot()["active_reservations"] == 1

    with pytest.raises(BudgetExceededError):
        send(client)

    assert len(sdk.responses.calls) == 1


def test_missing_usage_keeps_reservation():
    response = SimpleNamespace(
        id="resp_missing_usage",
        output=[],
        usage=None,
    )
    budget = AgentBudget()
    client = create_client(
        sdk=FakeSDK(response),
        budget=budget,
    )

    with pytest.raises(APIRequestUncertainError):
        send(client)

    assert budget.snapshot()["active_reservations"] == 1


def test_invalid_input_estimate_rejected():
    sdk = FakeSDK(mock_response())
    client = create_client(sdk=sdk)

    with pytest.raises(ValueError):
        client.create_response(
            instructions="Test",
            input=[],
            tools=[],
            max_input_tokens_estimate=-1,
        )

    assert sdk.responses.calls == []


def test_invalid_response_still_counts_usage():
    response = SimpleNamespace(
        id="resp_invalid",
        output=[{
            "type": "function_call",
            "call_id": "call_invalid",
            "name": "analyze_positions",
            "arguments": "{bad json}",
        }],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
        ),
    )
    budget = AgentBudget()
    client = create_client(
        sdk=FakeSDK(response),
        budget=budget,
    )

    with pytest.raises(ValueError):
        send(client)

    assert budget.llm_calls == 1
    assert budget.snapshot()["active_reservations"] == 0


def test_builder_defaults_to_disabled():
    sdk = FakeSDK(mock_response())

    client = build_live_client(
        AgentBudget(),
        sdk_client=sdk,
        environ={},
    )

    with pytest.raises(APIConfigurationError):
        send(client)

    assert sdk.responses.calls == []


def test_builder_requires_verified_pricing():
    sdk = FakeSDK(mock_response())

    client = build_live_client(
        AgentBudget(),
        sdk_client=sdk,
        environ={
            "OPENAI_ENABLE_API": "true",
            "OPENAI_MODEL": "test-model",
            "OPENAI_API_KEY": "test-secret",
            "OPENAI_PRICING_VERIFIED": "false",
        },
    )

    with pytest.raises(APIConfigurationError):
        send(client)

    assert sdk.responses.calls == []
