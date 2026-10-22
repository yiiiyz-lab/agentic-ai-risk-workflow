
"""Budget-controlled OpenAI Responses API client.

Supports stateless multi-round investigations by preserving the
provider's response output, including encrypted reasoning items.

Tool execution remains the responsibility of the orchestrator.
"""

import json
import os

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from src.agent.api_config import (
    APIConfigurationError,
    OpenAIConfig,
)
from src.agent.budget import AgentBudget
from src.agent.openai_adapter import (
    ModelResponse,
    normalize_response,
)


class APIRequestUncertainError(RuntimeError):
    """Request outcome or usage is unknown; reservation retained."""

    def __init__(self, reservation_id: str):
        self.reservation_id = reservation_id
        super().__init__(
            "API request outcome is uncertain; "
            "reservation retained for reconciliation"
        )


class APIResponseStatusError(RuntimeError):
    """The provider returned a non-completed response."""

    def __init__(self, status: str):
        self.status = status
        super().__init__(
            f"API response was not completed: {status}"
        )


@dataclass(frozen=True)
class RequestAudit:
    model: str
    outcome: str
    response_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: str | None
    reservation_id: str
    details: str | None = None

    def to_dict(self):
        return {
            "model": self.model,
            "outcome": self.outcome,
            "response_id": self.response_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "reservation_id": self.reservation_id,
            "details": self.details,
        }


@dataclass(frozen=True)
class APIResult:
    response: ModelResponse
    reservation_id: str
    reserved_cost_usd: Decimal
    estimated_actual_cost_usd: Decimal
    settlement: dict[str, Any]

    # Provider output items for stateless conversation continuation.
    # Do not write these items to public logs or reports.
    response_output: list[dict[str, Any]] = field(
        default_factory=list
    )


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _serialize_response_output(raw_response):
    """Convert SDK output items into JSON-compatible dictionaries.

    Preserve complete output items, including reasoning items and
    encrypted_content when returned by the API.
    """
    raw_output = _get(raw_response, "output", [])

    if raw_output is None:
        raw_output = []

    if not isinstance(raw_output, list):
        raise ValueError("Response output must be a list")

    serialized_output = []

    for item in raw_output:
        if isinstance(item, dict):
            serialized_item = item
        elif hasattr(item, "model_dump"):
            serialized_item = item.model_dump(
                mode="json",
                exclude_none=True,
            )
        else:
            raise TypeError(
                "Response output contains an unsupported item"
            )

        if not isinstance(serialized_item, dict):
            raise ValueError(
                "Response output item must be an object"
            )

        serialized_output.append(serialized_item)

    # Validate that the result is safe to pass back as JSON.
    json.dumps(
        serialized_output,
        ensure_ascii=False,
        allow_nan=False,
    )

    return serialized_output


class OpenAIResponsesClient:
    """Explicitly enabled, budget-controlled Responses API client.

    The application owns tool execution, not this client.
    """

    MAX_INPUT_ESTIMATE = 24000
    MAX_REQUEST_CHARS = 80000

    def __init__(
        self,
        config: OpenAIConfig,
        budget: AgentBudget,
        sdk_client=None,
        *,
        pricing_verified: bool = False,
    ):
        self.config = config
        self.budget = budget
        self._sdk_client = sdk_client
        self.pricing_verified = pricing_verified
        self.audit_log: list[RequestAudit] = []

    def _get_sdk_client(self):
        if self._sdk_client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise APIConfigurationError(
                    "Install openai before live API use"
                ) from exc

            self._sdk_client = OpenAI(
                api_key=self.config.api_key,
                max_retries=0,
                timeout=90.0,
            )

        return self._sdk_client

    def _audit(
        self,
        *,
        outcome,
        reservation_id,
        response_id=None,
        input_tokens=None,
        output_tokens=None,
        estimated_cost_usd=None,
        details=None,
    ):
        record = RequestAudit(
            model=self.config.model,
            outcome=outcome,
            response_id=response_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=(
                str(estimated_cost_usd)
                if estimated_cost_usd is not None
                else None
            ),
            reservation_id=reservation_id,
            details=details,
        )

        self.audit_log.append(record)
        return record

    def create_response(
        self,
        *,
        instructions: str,
        input: list,
        tools: list,
        max_input_tokens_estimate: int,
    ) -> APIResult:
        """Send one budget-controlled Responses API request."""

        self.config.require_enabled()

        if not self.pricing_verified:
            raise APIConfigurationError(
                "Model pricing must be verified"
            )

        if (
            type(max_input_tokens_estimate) is not int
            or not 0 < max_input_tokens_estimate
            <= self.MAX_INPUT_ESTIMATE
        ):
            raise ValueError(
                "Input token estimate exceeds allowed range"
            )

        if not isinstance(instructions, str) or not instructions:
            raise ValueError(
                "Instructions must be non-empty"
            )

        if not isinstance(input, list):
            raise ValueError("Input must be a list")

        if not isinstance(tools, list):
            raise ValueError("Tools must be a list")

        # Character-based guard, not an exact token count.
        # Covers instructions, conversation, and tool schemas.
        try:
            serialized = json.dumps(
                {
                    "instructions": instructions,
                    "input": input,
                    "tools": tools,
                },
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Request payload is not valid JSON"
            ) from exc

        if len(serialized) > self.MAX_REQUEST_CHARS:
            raise ValueError(
                f"Request payload exceeds character limit: "
                f"{len(serialized):,} > "
                f"{self.MAX_REQUEST_CHARS:,} characters"
            )

        max_output_tokens = (
            self.budget.max_output_tokens_per_call
        )

        reserved_cost = self.config.estimate_cost(
            max_input_tokens_estimate,
            max_output_tokens,
        )

        sdk = self._get_sdk_client()

        reservation_id = self.budget.reserve_llm_call(
            reserved_cost
        )

        try:
            raw_response = sdk.responses.create(
                model=self.config.model,
                instructions=instructions,
                input=input,
                tools=tools,
                max_output_tokens=max_output_tokens,
                parallel_tool_calls=False,
                store=False,
                include=[
                    "reasoning.encrypted_content",
                ],
            )
        except Exception as exc:
            self._audit(
                outcome="UNCERTAIN",
                reservation_id=reservation_id,
                details=f"Request exception: {type(exc).__name__}",
            )

            raise APIRequestUncertainError(
                reservation_id
            ) from exc

        response_id = _get(raw_response, "id")

        if not isinstance(response_id, str):
            response_id = None

        usage = _get(raw_response, "usage")

        input_tokens = _get(usage, "input_tokens")
        output_tokens = _get(usage, "output_tokens")

        if (
            type(input_tokens) is not int
            or type(output_tokens) is not int
            or input_tokens < 0
            or output_tokens < 0
        ):
            self._audit(
                outcome="UNCERTAIN",
                reservation_id=reservation_id,
                response_id=response_id,
                details="Missing or invalid usage data",
            )

            raise APIRequestUncertainError(
                reservation_id
            )

        actual_cost = self.config.estimate_cost(
            input_tokens,
            output_tokens,
        )

        settlement = self.budget.settle_llm_call(
            reservation_id,
            actual_cost,
        )

        status = _get(raw_response, "status")

        # Preserve compatibility with older offline SDK mocks
        # that omit status. Real API responses include status.
        if status is None:
            status = "completed"

        if status != "completed":
            self._audit(
                outcome="INCOMPLETE",
                reservation_id=reservation_id,
                response_id=response_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=actual_cost,
                details=f"Provider status: {status}",
            )

            raise APIResponseStatusError(str(status))

        try:
            normalized = normalize_response(raw_response)

            response_output = _serialize_response_output(
                raw_response
            )

        except (ValueError, TypeError, AttributeError) as exc:
            self._audit(
                outcome="INVALID_RESPONSE",
                reservation_id=reservation_id,
                response_id=response_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=actual_cost,
                details="Response validation failed",
            )

            raise

        self._audit(
            outcome="COMPLETED",
            reservation_id=reservation_id,
            response_id=response_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=actual_cost,
            details=(
                "Estimated cost exceeded reservation"
                if settlement["over_reservation"]
                else None
            ),
        )

        return APIResult(
            response=normalized,
            reservation_id=reservation_id,
            reserved_cost_usd=reserved_cost,
            estimated_actual_cost_usd=actual_cost,
            settlement=settlement,
            response_output=response_output,
        )


def build_live_client(
    budget: AgentBudget,
    *,
    sdk_client=None,
    environ=None,
):
    """Build a live client from explicitly supplied environment settings."""

    env = os.environ if environ is None else environ

    config = OpenAIConfig.from_env(env)

    verified = env.get(
        "OPENAI_PRICING_VERIFIED", "false"
    ).strip().lower()

    if verified not in ("true", "false"):
        raise APIConfigurationError(
            "OPENAI_PRICING_VERIFIED must be true or false"
        )

    return OpenAIResponsesClient(
        config=config,
        budget=budget,
        sdk_client=sdk_client,
        pricing_verified=(verified == "true"),
    )
