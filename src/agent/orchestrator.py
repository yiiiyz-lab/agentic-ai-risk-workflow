
"""Bounded, evidence-grounded AI exception investigation."""

import json
from dataclasses import dataclass

from src.agent.budget import AgentBudget, BudgetExceededError
from src.agent.investigation_state import InvestigationState
from src.agent.openai_adapter import build_openai_tools
from src.agent.openai_client import OpenAIResponsesClient
from src.agent.payload_limits import (
    PayloadLimitError,
    serialize_tool_result,
)
from src.agent.tool_registry import ToolRegistry


INSTRUCTIONS = """
You are an independent market risk investigation assistant.

Investigate the supplied exception using the available tools.
Choose relevant tools dynamically. Do not assume a fixed sequence.

All numerical claims must come from tool evidence.
Distinguish observations, interpretations, and hypotheses.
Do not invent market events, policies, or causal explanations.
Identify data limitations and uncertainties.

The portfolio and as-of date are trusted application context.
Do not attempt to change them.

Finish with a concise investigation report:
- Exception and observed risk
- Supporting evidence
- Potential drivers
- Limitations
- Recommended human review

Cite tool evidence by tool name and evidence index.
The report is a draft, not a final risk approval.
"""


@dataclass
class InvestigationResult:
    state: InvestigationState
    draft_report: str
    budget_snapshot: dict

    def to_dict(self):
        result = self.state.to_dict()
        result["draft_report"] = self.draft_report
        result["budget"] = self.budget_snapshot
        return result


def _clean_arguments(arguments):
    """Remove nullable optional arguments from model calls."""
    return {
        key: value
        for key, value in arguments.items()
        if value is not None
    }


def _initial_input(state, exception):
    return [{
        "role": "user",
        "content": json.dumps(
            {
                "portfolio_id": state.portfolio_id,
                "as_of_date": state.as_of_date,
                "exception": exception,
            },
            allow_nan=False,
            default=str,
        ),
    }]


def _estimate_input_tokens(instructions, messages, tools):
    """Conservative heuristic, not an exact tokenizer count.

    Counts all conversation items and tool schemas.
    The live client separately enforces its character limit.
    """
    payload = json.dumps(
        {
            "instructions": instructions,
            "input": messages,
            "tools": tools,
        },
        ensure_ascii=False,
        allow_nan=False,
    )

    # Conservative approximation for typical English/JSON
    # payloads, plus a fixed overhead allowance.
    estimate = len(payload) // 3 + 1000

    return min(
        OpenAIResponsesClient.MAX_INPUT_ESTIMATE,
        max(1500, estimate),
    )


class InvestigationOrchestrator:
    def __init__(
        self,
        adapter,
        registry=None,
        budget=None,
        max_rounds=6,
    ):
        if type(max_rounds) is not int or max_rounds <= 0:
            raise ValueError("max_rounds must be positive")

        self.adapter = adapter
        self.registry = (
            registry if registry is not None
            else ToolRegistry()
        )

        if isinstance(adapter, OpenAIResponsesClient):
            if budget is None:
                budget = adapter.budget
            elif budget is not adapter.budget:
                raise ValueError(
                    "Live client and orchestrator must "
                    "share the same AgentBudget instance"
                )

        self.budget = (
            budget if budget is not None
            else AgentBudget()
        )
        self.max_rounds = max_rounds

    def investigate(self, exception):
        required = (
            "portfolio_id",
            "date",
            "exception_id",
            "rule_id",
            "status",
        )

        if not isinstance(exception, dict):
            raise ValueError(
                "Exception must be a dictionary"
            )

        if any(not exception.get(key) for key in required):
            raise ValueError(
                "Missing exception identifiers"
            )

        state = InvestigationState(
            portfolio_id=str(exception["portfolio_id"]),
            as_of_date=str(exception["date"]),
            exception_id=str(exception["exception_id"]),
            rule_id=str(exception["rule_id"]),
            exception_status=str(exception["status"]),
        )

        state.set_status("RUNNING")

        messages = _initial_input(state, exception)
        tools = build_openai_tools(self.registry)
        draft_report = ""

        is_live = isinstance(
            self.adapter,
            OpenAIResponsesClient,
        )

        try:
            for round_number in range(self.max_rounds):
                state.add_trace(
                    "MODEL_REQUEST",
                    {"round": round_number + 1},
                )

                if is_live:
                    input_estimate = _estimate_input_tokens(
                        INSTRUCTIONS,
                        messages,
                        tools,
                    )

                    api_result = self.adapter.create_response(
                        instructions=INSTRUCTIONS,
                        input=messages,
                        tools=tools,
                        max_input_tokens_estimate=input_estimate,
                    )

                    response = api_result.response

                    state.add_trace(
                        "MODEL_USAGE",
                        {
                            "response_id": response.response_id,
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "estimated_cost_usd": str(
                                api_result.estimated_actual_cost_usd
                            ),
                        },
                    )

                    # Preserve actual model output, including
                    # encrypted reasoning items where provided.
                    # Keep this only in the working conversation.
                    messages.extend(
                        api_result.response_output
                    )

                else:
                    self.budget.check_llm_call("0")

                    response = self.adapter.create_response(
                        instructions=INSTRUCTIONS,
                        input=messages,
                        tools=tools,
                    )

                    self.budget.record_llm_call("0")

                if response.tool_calls:
                    for call in response.tool_calls:
                        arguments = _clean_arguments(
                            call.arguments
                        )

                        # Tool Registry remains the only
                        # execution entry point.
                        result = self.registry.execute(
                            call.name,
                            arguments,
                            state,
                            self.budget,
                        )

                        serialized = serialize_tool_result(
                            result
                        )

                        if not is_live:
                            # Offline adapter has no raw
                            # provider output to preserve.
                            messages.append({
                                "type": "function_call",
                                "call_id": call.call_id,
                                "name": call.name,
                                "arguments": json.dumps(
                                    call.arguments
                                ),
                            })

                        # Both live and offline paths append
                        # the result for the next model round.
                        messages.append({
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": serialized,
                        })

                elif response.text.strip():
                    if not state.evidence:
                        state.limitations.append(
                            "Model generated a report without "
                            "consulting risk tools."
                        )
                        state.set_status("REQUIRES_REVIEW")
                        break

                    draft_report = response.text.strip()

                    state.add_trace(
                        "DRAFT_REPORT_CREATED",
                        {
                            "response_id": response.response_id,
                            "evidence_count": len(state.evidence),
                        },
                    )

                    state.set_status("REQUIRES_REVIEW")
                    break

                else:
                    state.limitations.append(
                        "Model returned neither tool calls "
                        "nor a final report."
                    )
                    state.set_status("FAILED")
                    break

            else:
                state.limitations.append(
                    "Investigation reached the maximum "
                    "number of model rounds."
                )
                state.set_status("REQUIRES_REVIEW")

        except BudgetExceededError:
            state.limitations.append(
                "Investigation stopped at its budget limit."
            )
            state.set_status("BUDGET_EXCEEDED")

        except PayloadLimitError:
            state.limitations.append(
                "Tool evidence exceeded the model-facing "
                "size limit; full evidence remains recorded."
            )
            state.set_status("REQUIRES_REVIEW")

        except Exception as exc:
            state.add_trace(
                "INVESTIGATION_ERROR",
                {
                    "error_type": type(exc).__name__,
                },
            )

            state.limitations.append(
                "Investigation interrupted: "
                f"{type(exc).__name__}"
            )
            state.set_status("FAILED")

        return InvestigationResult(
            state=state,
            draft_report=draft_report,
            budget_snapshot=self.budget.snapshot(),
        )
