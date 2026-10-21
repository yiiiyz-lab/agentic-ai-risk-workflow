
"""Bounded, evidence-grounded AI exception investigation.

Supports:
- Dynamic risk tool selection
- Stateless OpenAI Responses API continuation
- Full local evidence retention
- Bounded model-facing evidence summaries
- A reserved final report-drafting round
- Shared LLM and tool budgets
- Offline adapter testing
"""

import json
from dataclasses import dataclass

from src.agent.budget import AgentBudget, BudgetExceededError
from src.agent.investigation_state import InvestigationState
from src.agent.openai_adapter import build_openai_tools
from src.agent.openai_client import OpenAIResponsesClient
from src.agent.payload_limits import (
    PayloadLimitError,
    serialize_model_evidence,
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

A summarized tool result is incomplete evidence.
Do not treat omitted fields as passing checks or established facts.
Disclose material evidence limitations.

The report is a draft, not a final risk approval.
"""


DRAFT_INSTRUCTIONS = INSTRUCTIONS + """

You are now in the final report-drafting phase.

Do not request additional tools.
Use only the evidence already supplied in this conversation.

Produce the investigation report with these headings:

1. Exception and Observed Risk
2. Supporting Evidence
3. Potential Drivers
4. Limitations
5. Recommended Human Review

For numerical claims, cite the supporting tool and evidence index.
Distinguish verified observations from possible explanations.

If the evidence is incomplete, explain what remains unverified.
Do not invent missing figures or reconciliation results.

The output must remain a draft requiring human review.
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
    if not isinstance(arguments, dict):
        raise ValueError(
            "Model tool arguments must be a dictionary"
        )

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
    """Estimate request tokens conservatively.

    This is a heuristic, not a tokenizer or a hard cost bound.
    The live client also enforces a request character limit.
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

    estimate = len(payload) // 3 + 1000

    return min(
        OpenAIResponsesClient.MAX_INPUT_ESTIMATE,
        max(1500, estimate),
    )


def _is_summary(serialized):
    """Identify our explicit incomplete-evidence envelope."""
    value = json.loads(serialized)

    return (
        isinstance(value, dict)
        and value.get("evidence_complete") is False
        and value.get("summary_type")
        == "bounded_structural_preview"
    )


class InvestigationOrchestrator:
    def __init__(
        self,
        adapter,
        registry=None,
        budget=None,
        max_rounds=6,
        report_rounds=1,
    ):
        if type(max_rounds) is not int or max_rounds <= 0:
            raise ValueError(
                "max_rounds must be positive"
            )

        if type(report_rounds) is not int or report_rounds != 1:
            raise ValueError(
                "report_rounds must be 1"
            )

        self.adapter = adapter

        self.registry = (
            registry if registry is not None
            else ToolRegistry()
        )

        self.is_live = isinstance(
            adapter,
            OpenAIResponsesClient,
        )

        if self.is_live:
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
        self.report_rounds = report_rounds

    def _request_model(
        self,
        *,
        state,
        messages,
        tools,
        instructions,
        round_number,
        phase,
    ):
        """Execute one model round and account for its usage."""

        state.add_trace(
            "MODEL_REQUEST",
            {
                "round": round_number,
                "phase": phase,
                "tool_count": len(tools),
            },
        )

        if self.is_live:
            input_estimate = _estimate_input_tokens(
                instructions,
                messages,
                tools,
            )

            api_result = self.adapter.create_response(
                instructions=instructions,
                input=messages,
                tools=tools,
                max_input_tokens_estimate=input_estimate,
            )

            response = api_result.response

            state.add_trace(
                "MODEL_USAGE",
                {
                    "round": round_number,
                    "phase": phase,
                    "response_id": response.response_id,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "estimated_cost_usd": str(
                        api_result.estimated_actual_cost_usd
                    ),
                },
            )

            # Preserve original response items for stateless
            # continuation, including encrypted reasoning
            # content where returned by the provider.
            #
            # Never add these raw items to the audit log.
            messages.extend(
                api_result.response_output
            )

            return response

        # Offline adapter path.
        self.budget.check_llm_call("0")

        response = self.adapter.create_response(
            instructions=instructions,
            input=messages,
            tools=tools,
        )

        self.budget.record_llm_call("0")

        return response

    def _execute_tools(
        self,
        *,
        response,
        state,
        messages,
    ):
        """Execute requested tools and append model-facing results."""

        for call in response.tool_calls:
            arguments = _clean_arguments(
                call.arguments
            )

            # ToolRegistry is the only execution entry point.
            # It validates arguments and records full evidence.
            result = self.registry.execute(
                call.name,
                arguments,
                state,
                self.budget,
            )

            evidence_index = len(state.evidence) - 1

            serialized = serialize_model_evidence(
                result,
                tool_name=call.name,
                evidence_index=evidence_index,
            )

            if _is_summary(serialized):
                state.add_trace(
                    "EVIDENCE_SUMMARIZED",
                    {
                        "tool_name": call.name,
                        "evidence_index": evidence_index,
                        "reason": "model_facing_size_limit",
                    },
                )

                limitation = (
                    "Model received incomplete evidence from "
                    f"{call.name} at index {evidence_index}."
                )

                state.limitations.append(limitation)

            if not self.is_live:
                # Offline adapters do not supply original
                # provider response output items.
                messages.append({
                    "type": "function_call",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": json.dumps(
                        call.arguments,
                        allow_nan=False,
                    ),
                })

            messages.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": serialized,
            })

    def _record_report(
        self,
        *,
        response,
        state,
    ):
        """Accept a model report as a draft requiring review."""

        report = response.text.strip()

        if not report:
            state.limitations.append(
                "Model did not generate a report."
            )

            state.add_trace(
                "REPORT_INCOMPLETE",
                {
                    "reason": "empty_model_response",
                },
            )

            state.set_status("REQUIRES_REVIEW")
            return ""

        if not state.evidence:
            state.limitations.append(
                "Model generated a report without "
                "consulting risk tools."
            )

            state.add_trace(
                "REPORT_INCOMPLETE",
                {
                    "reason": "no_tool_evidence",
                },
            )

            state.set_status("REQUIRES_REVIEW")
            return ""

        state.add_trace(
            "DRAFT_REPORT_CREATED",
            {
                "response_id": response.response_id,
                "evidence_count": len(state.evidence),
                "incomplete_evidence": bool(
                    state.limitations
                ),
            },
        )

        state.set_status("REQUIRES_REVIEW")

        return report

    def investigate(self, exception):
        """Investigate one exception with bounded model rounds."""

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

        if any(
            not exception.get(key)
            for key in required
        ):
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

        messages = _initial_input(
            state,
            exception,
        )

        tools = build_openai_tools(
            self.registry
        )

        draft_report = ""

        try:
            # Reserve the last round for report drafting.
            # With max_rounds=10, up to 9 rounds can
            # investigate and round 10 drafts the report.
            investigation_rounds = max(
                0,
                self.max_rounds - self.report_rounds,
            )

            for round_number in range(
                1,
                investigation_rounds + 1,
            ):
                response = self._request_model(
                    state=state,
                    messages=messages,
                    tools=tools,
                    instructions=INSTRUCTIONS,
                    round_number=round_number,
                    phase="INVESTIGATION",
                )

                if response.tool_calls:
                    self._execute_tools(
                        response=response,
                        state=state,
                        messages=messages,
                    )

                    continue

                if response.text.strip():
                    draft_report = self._record_report(
                        response=response,
                        state=state,
                    )
                    break

                state.limitations.append(
                    "Model returned neither tool calls "
                    "nor a final report."
                )

                state.add_trace(
                    "REPORT_INCOMPLETE",
                    {
                        "reason": "empty_investigation_response",
                    },
                )

                state.set_status("FAILED")
                break

            # If investigation did not already finish,
            # use the reserved final round to draft.
            if state.status == "RUNNING":
                state.add_trace(
                    "REPORT_DRAFTING_STARTED",
                    {
                        "evidence_count": len(state.evidence),
                        "tools_enabled": False,
                    },
                )

                if not state.evidence:
                    state.limitations.append(
                        "No risk tool evidence was collected "
                        "before the report-drafting phase."
                    )

                    state.add_trace(
                        "REPORT_INCOMPLETE",
                        {
                            "reason": "no_tool_evidence",
                        },
                    )

                    state.set_status("REQUIRES_REVIEW")

                else:
                    # Explicitly instruct the model that
                    # evidence collection has ended.
                    messages.append({
                        "role": "user",
                        "content": (
                            "Evidence collection is complete. "
                            "Draft the investigation report now "
                            "using the available evidence only. "
                            "Disclose any incomplete summaries "
                            "and unresolved questions. "
                            "Do not request additional tools."
                        ),
                    })

                    response = self._request_model(
                        state=state,
                        messages=messages,
                        tools=[],
                        instructions=DRAFT_INSTRUCTIONS,
                        round_number=self.max_rounds,
                        phase="REPORT_DRAFTING",
                    )

                    if response.tool_calls:
                        state.limitations.append(
                            "Model requested tools during "
                            "the report-only phase."
                        )

                        state.add_trace(
                            "REPORT_INCOMPLETE",
                            {
                                "reason": "unexpected_tool_call",
                            },
                        )

                        state.set_status("REQUIRES_REVIEW")

                    else:
                        draft_report = self._record_report(
                            response=response,
                            state=state,
                        )

        except BudgetExceededError:
            state.limitations.append(
                "Investigation stopped at its budget limit."
            )

            state.add_trace(
                "REPORT_INCOMPLETE",
                {
                    "reason": "budget_exceeded",
                },
            )

            state.set_status("BUDGET_EXCEEDED")

        except PayloadLimitError:
            state.limitations.append(
                "Tool evidence could not be safely "
                "serialized within the model-facing limit. "
                "Full evidence remains recorded."
            )

            state.add_trace(
                "REPORT_INCOMPLETE",
                {
                    "reason": "payload_limit",
                },
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
