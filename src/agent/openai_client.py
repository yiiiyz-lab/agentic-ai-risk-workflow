
"""Budget-controlled OpenAI Responses API client."""

import os
from dataclasses import dataclass
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
    """
    A request failed without trustworthy usage information.

    Its budget reservation remains active because the
    provider may have received and billed the request.
    """

    def __init__(self, reservation_id: str):
        self.reservation_id = reservation_id
        super().__init__(
            "API request outcome is uncertain; "
            "reservation retained for reconciliation"
        )


@dataclass(frozen=True)
class APIResult:
    response: ModelResponse
    reservation_id: str
    reserved_cost_usd: Decimal
    estimated_actual_cost_usd: Decimal
    settlement: dict[str, Any]


class OpenAIResponsesClient:
    """
    Opt-in OpenAI client with per-request budget reservation.

    SDK clients may be injected for offline tests.
    No SDK or network connection is initialized until
    an enabled request is attempted.
    """

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

    def _get_sdk_client(self):
        if self._sdk_client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise APIConfigurationError(
                    "Install the openai package before "
                    "using live API access"
                ) from exc

            self._sdk_client = OpenAI(
                api_key=self.config.api_key,
                max_retries=0,
                timeout=30.0,
            )

        return self._sdk_client

    def create_response(
        self,
        *,
        instructions: str,
        input: list,
        tools: list,
        max_input_tokens_estimate: int,
    ) -> APIResult:
        """
        Submit one bounded request.

        max_input_tokens_estimate must be a conservative
        upper bound supplied by the calling workflow.

        On an uncertain request failure, the reservation
        is retained and an error containing its ID is raised.
        """
        self.config.require_enabled()

        if not self.pricing_verified:
            raise APIConfigurationError(
                "Model pricing must be verified before "
                "enabling live requests"
            )

        if (
            type(max_input_tokens_estimate) is not int
            or max_input_tokens_estimate <= 0
        ):
            raise ValueError(
                "Input token estimate must be a "
                "positive integer"
            )

        if not isinstance(instructions, str) or not instructions:
            raise ValueError("Instructions must be non-empty")

        if not isinstance(input, list):
            raise ValueError("Input must be a list")

        if not isinstance(tools, list):
            raise ValueError("Tools must be a list")

        max_output_tokens = (
            self.budget.max_output_tokens_per_call
        )

        reserved_cost = self.config.estimate_cost(
            max_input_tokens_estimate,
            max_output_tokens,
        )

        # Initialize the client before reserving budget.
        # A missing SDK or invalid local setup must not
        # leave an unresolved reservation.
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
            )
        except Exception as exc:
            # Do not cancel. A timeout or transport failure
            # does not establish that no billable work occurred.
            raise APIRequestUncertainError(
                reservation_id
            ) from exc

        # A response was returned. Account for usage before
        # normalizing its tool calls or text.
        usage = getattr(raw_response, "usage", None)
        if isinstance(raw_response, dict):
            usage = raw_response.get("usage")

        if usage is None:
            raise APIRequestUncertainError(
                reservation_id
            )

        def read_usage(field):
            if isinstance(usage, dict):
                return usage.get(field)
            return getattr(usage, field, None)

        input_tokens = read_usage("input_tokens")
        output_tokens = read_usage("output_tokens")

        if (
            type(input_tokens) is not int
            or type(output_tokens) is not int
            or input_tokens < 0
            or output_tokens < 0
        ):
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

        # Normalization may fail, but the billable request
        # has already been accounted for.
        response = normalize_response(raw_response)

        return APIResult(
            response=response,
            reservation_id=reservation_id,
            reserved_cost_usd=reserved_cost,
            estimated_actual_cost_usd=actual_cost,
            settlement=settlement,
        )


def build_live_client(
    budget: AgentBudget,
    *,
    sdk_client=None,
    environ=None,
):
    """Construct a client from explicit environment settings."""
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
