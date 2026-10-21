
import json

import pytest

from src.agent.openai_adapter import (
    AdapterError,
    OfflineOpenAIAdapter,
    build_openai_tools,
    normalize_response,
)


def test_exports_ten_tools():
    tools = build_openai_tools()

    assert len(tools) == 10
    assert len({tool["name"] for tool in tools}) == 10


def test_tool_definitions_are_strict():
    for tool in build_openai_tools():
        assert tool["type"] == "function"
        assert tool["strict"] is True
        assert (
            tool["parameters"]["additionalProperties"]
            is False
        )


def test_no_trusted_context_exposed():
    for tool in build_openai_tools():
        properties = tool["parameters"]["properties"]

        assert "portfolio_id" not in properties
        assert "as_of_date" not in properties

        assert not any(
            name.endswith("_path")
            for name in properties
        )


def test_optional_arguments_become_nullable():
    tools = {
        tool["name"]: tool
        for tool in build_openai_tools()
    }

    schema = tools["analyze_transactions"]["parameters"]

    assert "lookback_days" in schema["required"]
    assert "anyOf" in schema["properties"]["lookback_days"]


def test_required_scenarios_remain_required():
    tools = {
        tool["name"]: tool
        for tool in build_openai_tools()
    }

    schema = tools["run_stress_test"]["parameters"]

    assert "scenarios" in schema["required"]
    assert (
        schema["properties"]["scenarios"]["type"]
        == "array"
    )


def test_normalizes_tool_call():
    response = {
        "id": "resp_mock_001",
        "output": [{
            "type": "function_call",
            "call_id": "call_001",
            "name": "analyze_positions",
            "arguments": "{}",
        }],
        "usage": {
            "input_tokens": 120,
            "output_tokens": 30,
        },
    }

    normalized = normalize_response(response)

    assert normalized.response_id == "resp_mock_001"
    assert len(normalized.tool_calls) == 1
    assert normalized.tool_calls[0].name == "analyze_positions"
    assert normalized.tool_calls[0].arguments == {}
    assert normalized.input_tokens == 120
    assert normalized.output_tokens == 30


def test_normalizes_text_response():
    response = {
        "id": "resp_mock_002",
        "output": [{
            "type": "message",
            "content": [{
                "type": "output_text",
                "text": "Investigation completed.",
            }],
        }],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
        },
    }

    normalized = normalize_response(response)

    assert normalized.text == "Investigation completed."
    assert normalized.tool_calls == []


def test_rejects_invalid_json_arguments():
    response = {
        "id": "resp_mock_003",
        "output": [{
            "type": "function_call",
            "call_id": "call_003",
            "name": "analyze_positions",
            "arguments": "{invalid}",
        }],
    }

    with pytest.raises(AdapterError):
        normalize_response(response)


def test_rejects_duplicate_call_ids():
    call = {
        "type": "function_call",
        "call_id": "same_id",
        "name": "analyze_positions",
        "arguments": "{}",
    }

    with pytest.raises(AdapterError):
        normalize_response({
            "id": "resp_mock_004",
            "output": [call, call],
        })


def test_offline_adapter_consumes_responses():
    mock_response = {
        "id": "resp_mock_005",
        "output": [{
            "type": "function_call",
            "call_id": "call_005",
            "name": "retrieve_risk_policy",
            "arguments": json.dumps({
                "rule_id": "EQUITY_EXPOSURE_LIMIT",
                "metric": None,
            }),
        }],
        "usage": {
            "input_tokens": 150,
            "output_tokens": 40,
        },
    }

    adapter = OfflineOpenAIAdapter([mock_response])

    result = adapter.create_response(
        instructions="Investigate the exception.",
        input=[],
        tools=build_openai_tools(),
    )

    assert result.tool_calls[0].name == "retrieve_risk_policy"
    assert len(adapter.requests) == 1

    with pytest.raises(AdapterError):
        adapter.create_response(
            instructions="Another request",
            input=[],
            tools=[],
        )
