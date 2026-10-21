
from types import SimpleNamespace

import pytest

from src.agent.budget import AgentBudget
from src.agent.openai_client import (
    APIResponseStatusError,
    OpenAIResponsesClient,
)
from src.agent.payload_limits import (
    PayloadLimitError,
    serialize_tool_result,
)

from tests.test_openai_client import (
    FakeSDK,
    config,
    mock_response,
    send,
)


def make_client(response=None, budget=None):
    return OpenAIResponsesClient(
        config=config(),
        budget=budget or AgentBudget(),
        sdk_client=FakeSDK(
            response if response is not None
            else mock_response()
        ),
        pricing_verified=True,
    )


def test_successful_request_has_audit():
    client = make_client()
    send(client)

    record = client.audit_log[-1]

    assert record.outcome == "COMPLETED"
    assert record.response_id == "resp_test_001"
    assert record.input_tokens == 1000
    assert record.output_tokens == 100


def test_audit_excludes_api_key():
    client = make_client()
    send(client)

    assert "test-secret" not in str(
        client.audit_log[-1].to_dict()
    )


def test_incomplete_response_is_rejected():
    response = mock_response()
    response.status = "incomplete"

    budget = AgentBudget()
    client = make_client(response, budget)

    with pytest.raises(APIResponseStatusError):
        send(client)

    assert budget.llm_calls == 1
    assert budget.snapshot()["active_reservations"] == 0
    assert client.audit_log[-1].outcome == "INCOMPLETE"


def test_failed_response_is_rejected():
    response = mock_response()
    response.status = "failed"

    client = make_client(response)

    with pytest.raises(APIResponseStatusError):
        send(client)

    assert client.audit_log[-1].outcome == "INCOMPLETE"


def test_oversized_request_is_blocked():
    client = make_client()

    with pytest.raises(ValueError):
        client.create_response(
            instructions="X" * 50000,
            input=[],
            tools=[],
            max_input_tokens_estimate=1000,
        )

    assert client.budget.llm_calls == 0


def test_input_estimate_ceiling():
    client = make_client()

    with pytest.raises(ValueError):
        client.create_response(
            instructions="Test",
            input=[],
            tools=[],
            max_input_tokens_estimate=12001,
        )


def test_valid_tool_result_serialization():
    result = {
        "status": "PASS",
        "var_99_usd": 12345.67,
    }

    text = serialize_tool_result(result)

    assert '"status":"PASS"' in text
    assert '"var_99_usd":12345.67' in text


def test_oversized_tool_result_rejected():
    with pytest.raises(PayloadLimitError):
        serialize_tool_result({
            "large_text": "X" * 20000
        })


def test_nonfinite_tool_result_rejected():
    with pytest.raises(PayloadLimitError):
        serialize_tool_result({
            "var": float("nan")
        })


def test_invalid_response_audited():
    response = SimpleNamespace(
        id="resp_invalid",
        status="completed",
        output=[{
            "type": "function_call",
            "call_id": "call_1",
            "name": "analyze_positions",
            "arguments": "{invalid}",
        }],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
        ),
    )

    budget = AgentBudget()
    client = make_client(response, budget)

    with pytest.raises(ValueError):
        send(client)

    assert budget.llm_calls == 1
    assert client.audit_log[-1].outcome == "INVALID_RESPONSE"
