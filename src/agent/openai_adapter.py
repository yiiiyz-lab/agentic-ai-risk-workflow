
"""
OpenAI Responses API adapter.

Step 3A:
- Export validated tool definitions.
- Normalize model tool calls.
- Support deterministic offline testing.

No network requests are made in this module.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from src.agent.tool_registry import ToolRegistry


class AdapterError(ValueError):
    """Raised when a model response cannot be normalized."""


@dataclass(frozen=True)
class ModelToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelResponse:
    response_id: str
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


def _strict_schema(schema):
    """
    Convert registry schemas to strict function schemas.

    Responses API strict mode requires every object property
    to be listed as required. Optional values are represented
    as nullable properties.
    """
    if "anyOf" in schema:
        return {
            "anyOf": [
                _strict_schema(item)
                for item in schema["anyOf"]
            ]
        }

    result = dict(schema)
    kind = result.get("type")

    if kind == "object":
        properties = {
            name: _strict_schema(value)
            for name, value in result.get(
                "properties", {}
            ).items()
        }

        originally_required = set(
            result.get("required", [])
        )

        for name, property_schema in properties.items():
            if name not in originally_required:
                if "anyOf" not in property_schema:
                    properties[name] = {
                        "anyOf": [
                            property_schema,
                            {"type": "null"},
                        ]
                    }

        result["properties"] = properties
        result["required"] = list(properties)
        result["additionalProperties"] = False

    elif kind == "array":
        result["items"] = _strict_schema(
            result["items"]
        )

    # Keep only portable schema constraints here.
    # Our Tool Registry remains the final validator.
    for key in (
        "format",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
    ):
        result.pop(key, None)

    return result


def build_openai_tools(registry=None):
    """Export registered tools for Responses API requests."""
    registry = registry or ToolRegistry()

    definitions = []

    for tool in registry.list_tools():
        definitions.append({
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": _strict_schema(
                tool["parameters"]
            ),
            "strict": True,
        })

    return definitions


def _get(obj, name, default=None):
    """Read a field from a dictionary or SDK-style object."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def normalize_response(response):
    """
    Normalize a Responses API response or offline fixture.

    Tool calls are not executed here. Execution belongs
    exclusively to the validated Tool Registry.
    """
    response_id = _get(response, "id")

    if not isinstance(response_id, str) or not response_id:
        raise AdapterError("Missing response ID")

    output = _get(response, "output", [])

    if not isinstance(output, list):
        raise AdapterError("Invalid response output")

    tool_calls = []
    text_parts = []
    seen_call_ids = set()

    for item in output:
        item_type = _get(item, "type")

        if item_type == "function_call":
            call_id = _get(item, "call_id")
            name = _get(item, "name")
            raw_arguments = _get(item, "arguments")

            if (
                not isinstance(call_id, str)
                or not call_id
                or call_id in seen_call_ids
            ):
                raise AdapterError(
                    "Missing or duplicate tool call ID"
                )

            if not isinstance(name, str) or not name:
                raise AdapterError("Missing tool name")

            if not isinstance(raw_arguments, str):
                raise AdapterError(
                    "Tool arguments must be JSON text"
                )

            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise AdapterError(
                    "Invalid tool arguments JSON"
                ) from exc

            if not isinstance(arguments, dict):
                raise AdapterError(
                    "Tool arguments must be an object"
                )

            seen_call_ids.add(call_id)

            tool_calls.append(
                ModelToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                )
            )

        elif item_type == "message":
            for content in _get(item, "content", []):
                if _get(content, "type") == "output_text":
                    value = _get(content, "text", "")
                    if isinstance(value, str):
                        text_parts.append(value)

    usage = _get(response, "usage", {})

    input_tokens = _get(usage, "input_tokens", 0)
    output_tokens = _get(usage, "output_tokens", 0)

    for count in (input_tokens, output_tokens):
        if type(count) is not int or count < 0:
            raise AdapterError("Invalid token usage")

    return ModelResponse(
        response_id=response_id,
        tool_calls=tool_calls,
        text="\n".join(text_parts),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class OfflineOpenAIAdapter:
    """
    Deterministic adapter for development and tests.

    Each call consumes one preconfigured response.
    No API key or OpenAI SDK is required.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self._index = 0
        self.requests = []

    def create_response(self, *, instructions, input, tools):
        if self._index >= len(self._responses):
            raise AdapterError(
                "No more offline responses configured"
            )

        self.requests.append({
            "instructions": instructions,
            "input": input,
            "tools": tools,
        })

        response = self._responses[self._index]
        self._index += 1

        return normalize_response(response)
