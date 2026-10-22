
"""Bounded, evidence-grounded AI exception investigation.

Supports:
- Dynamic risk tool selection
- Stateless OpenAI Responses API continuation
- Full local evidence retention
- Bounded model-facing evidence summaries
- A reserved final report-drafting round
- Shared LLM and tool budgets
- Offline adapter testing
- Exception-specific investigation guidance
"""

import json
import re
from dataclasses import dataclass

from src.agent.budget import AgentBudget, BudgetExceededError
from src.agent.investigation_state import InvestigationState
from src.agent.openai_adapter import build_openai_tools
from src.agent.openai_client import OpenAIResponsesClient
from src.agent.payload_limits import (
    PayloadLimitError,
    serialize_model_evidence,
)
from src.agent.tool_registry import ToolRegistry, ToolValidationError


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
- Executive Summary
- Exception and observed risk
- Supporting evidence
- Potential drivers
- Limitations
- Recommended human review

Cite tool evidence by tool name and evidence index.

A summarized tool result is incomplete evidence.
Do not treat omitted fields as passing checks or established facts.
Disclose material evidence limitations.

"""

DRAFT_INSTRUCTIONS = """
You are preparing an evidence-grounded risk investigation report.

Use only the verified evidence collected during the investigation.
Do not invent market events, causal explanations, or model failures.
Distinguish confirmed findings from potential drivers.
Clearly disclose missing evidence and unresolved questions.

Write the report using exactly these six sections:

## 1. Executive Summary
## 2. Exception and Observed Risk
## 3. Supporting Evidence
## 4. Potential Drivers
## 5. Limitations
## 6. Recommended Human Review

Executive Summary requirements:
- Write 3-5 concise sentences.
- Identify the portfolio, investigation date, and exception.
- Summarize the observed risk metric and breached threshold.
- Explain the primary evidence-supported drivers.
- Mention material offsets or mitigating factors.
- Distinguish confirmed findings from unresolved explanations.
- Reference supporting evidence for material claims.
- Begin directly with the key risk finding.
- Do not start with labels such as "Draft investigation:",
  "Draft report:", or "Preliminary assessment:".
- Use professional risk-management language.
- Avoid repeating disclaimers in the Executive Summary.
- Keep the report's draft status and human-review requirement
  in the Recommended Human Review section. 

Evidence citation requirements:
- Cite every material numerical claim and analytical finding.
- Use the format [tool_name, evidence N].
- Use the actual tool name and zero-based evidence index.
- For multiple sources, use
  [tool_a, evidence 0; tool_b, evidence 1].
- Do not use vague references such as
  "risk metrics evidence" or "attribution evidence".
- Never invent tool names or evidence indices.

The detailed sections must substantiate the Executive Summary.
Do not introduce unsupported conclusions.
The final report requires human review.
"""


REQUIRED_REPORT_SECTIONS = (
    "Executive Summary",
    "Exception and Observed Risk",
    "Supporting Evidence",
    "Potential Drivers",
    "Limitations",
    "Recommended Human Review",
)


def missing_report_sections(report):
    """Return missing headings in the required six-section draft format.

    Supports Markdown headings with or without a leading section number.
    Does not mistake a mention in ordinary prose for a section heading.
    """
    if not isinstance(report, str):
        return list(REQUIRED_REPORT_SECTIONS)

    missing = []
    for section in REQUIRED_REPORT_SECTIONS:
        pattern = (
            r"(?im)^\s{0,3}#{1,4}\s+"
            r"(?:\d+[.)]\s+)?"
            + re.escape(section)
            + r"\s*$"
        )
        if not re.search(pattern, report):
            missing.append(section)
    return missing


def get_exception_guidance(rule_id):
    """
    Return additional investigation guidance for a risk rule.

    Guidance affects model instructions only. It does not
    change tool permissions, execution logic, risk thresholds,
    evidence storage, or investigation statuses.

    Unrecognized rules continue using the original instructions.
    """
    if rule_id == "PNL_VAR_EXCEEDANCE":
        return """

Additional guidance for P&L / VaR exceedance investigations:

1. Exception validation
- Verify the selected date's reported daily P&L.
- Retrieve the prior trading day's 99% VaR estimate.
- Compare the realized loss against prior-day VaR.
- Verify the configured warning and breach thresholds.
- Do not substitute current-day VaR for prior-day VaR.
- Use the existing risk metric and policy tools where relevant.

2. P&L attribution
- Prioritize get_pnl_attribution to identify instrument-level
  and asset-class loss contributors.
- Check whether calculated P&L reconciles to reported P&L.
- Identify the largest loss contributors and any offsets.
- Do not invent instrument contributions or reconciliation
  results that are absent from the evidence.

3. Historical context
- Examine preceding trading days for related P&L losses
  and potential consecutive VaR exceedances.
- When supported by the tool schema, use a multi-day
  attribution window ending on the trusted as-of date.
- Keep the primary investigation date unchanged.
- Distinguish daily P&L from cumulative multi-day P&L.
- Do not apply a one-day VaR estimate directly to a
  cumulative multi-day loss.

4. Positions and transactions
- Use analyze_positions to examine relevant holdings,
  exposure, and concentration.
- Use analyze_transactions when relevant to determine
  whether portfolio positions changed through trading.
- Distinguish active de-risking from market-driven changes
  in portfolio weights.
- Do not infer trading activity from exposure changes alone.

5. VaR and market risk factors
- Use get_var_attribution where relevant to examine
  historical VaR contributors.
- Use analyze_risk_factors to assess available market
  movements and portfolio sensitivities.
- Explain how observed losses relate to the available
  VaR methodology and portfolio exposures.
- Do not invent external market events or macroeconomic
  explanations without supporting evidence.
- Treat a VaR exceedance as a backtesting observation,
  not automatic proof of model failure.

6. Investigation conclusions
- Separate confirmed observations from hypotheses.
- State material data and methodology limitations.
- Identify unresolved questions and appropriate
  human risk-management review actions.
- Keep the report evidence-grounded and concise.
"""

    return ""


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
            # Record the request size before token estimation or
            # client-side validation. This helps diagnose failures
            # that occur before an API request is submitted.
            state.add_trace(
                "MODEL_REQUEST_PREPARATION",
                {
                    "round": round_number,
                    "phase": phase,
                    "message_count": len(messages),
                    "tool_count": len(tools),
                    "instruction_characters": len(instructions),
                },
            )

            input_estimate = _estimate_input_tokens(
                instructions,
                messages,
                tools,
            )

            state.add_trace(
                "MODEL_INPUT_ESTIMATE",
                {
                    "round": round_number,
                    "phase": phase,
                    "estimated_input_tokens": input_estimate,
                },
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
            arguments = _clean_arguments(call.arguments)

            try:
                # The registry validates dates, permissions and schemas.
                # Keep those checks authoritative and unchanged.
                result = self.registry.execute(
                    call.name,
                    arguments,
                    state,
                    self.budget,
                )
            except ToolValidationError as exc:
                # Invalid model arguments are recoverable. They fail
                # before registry execution and do not count as a
                # successful tool invocation or produce evidence.
                error_message = str(exc).replace("\n", " ")[:300]
                state.add_trace(
                    "TOOL_VALIDATION_ERROR",
                    {
                        "tool_name": call.name,
                        "arguments": arguments,
                        "error_type": type(exc).__name__,
                        "error_message": error_message,
                    },
                )
                limitation = (
                    f"Rejected {call.name} tool request: "
                    f"{error_message}"
                )
                if limitation not in state.limitations:
                    state.limitations.append(limitation)
                serialized = json.dumps(
                    {
                        "status": "VALIDATION_ERROR",
                        "tool_name": call.name,
                        "error": error_message,
                        "retry_guidance": (
                            "Correct the arguments or proceed using "
                            "existing evidence. For scenario_dates, "
                            "use only dates strictly before the "
                            "trusted as-of date."
                        ),
                    },
                    allow_nan=False,
                )
            else:
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
                    if limitation not in state.limitations:
                        state.limitations.append(limitation)

            # Every model function call needs exactly one matching
            # function_call_output, including rejected requests.
            if not self.is_live:
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

        missing = missing_report_sections(report)
        if missing and self.is_live:
            state.add_trace(
                "REPORT_INCOMPLETE",
                {
                    "reason": "missing_required_sections",
                    "missing_sections": missing,
                    "response_id": response.response_id,
                },
            )
            limitation = (
                "AI draft is missing required report sections: "
                + ", ".join(missing)
            )
            if limitation not in state.limitations:
                state.limitations.append(limitation)
            state.set_status("REQUIRES_REVIEW")
            # Retain the actual draft for audit and UI review.
            # Do not fabricate a summary or label it complete.
            return report

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

        # Add guidance only for exception types that need it.
        # Existing rules retain their original instructions.
        exception_guidance = get_exception_guidance(
            state.rule_id
        )

        investigation_instructions = (
            INSTRUCTIONS + exception_guidance
        )

        draft_instructions = (
            DRAFT_INSTRUCTIONS + exception_guidance
        )

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
                    instructions=investigation_instructions,
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
                    preliminary = response.text.strip()
                    missing = missing_report_sections(preliminary)
                    if self.is_live and missing:
                        # A model can finish its investigation with an
                        # unstructured five-section answer. Keep the text
                        # in the audit trace only as metadata and use the
                        # reserved drafting round for the required format.
                        state.add_trace(
                            "PRELIMINARY_REPORT_REJECTED",
                            {
                                "reason": "missing_required_sections",
                                "missing_sections": missing,
                                "response_id": response.response_id,
                            },
                        )
                        messages.append({
                            "role": "user",
                            "content": (
                                "Your preliminary report omitted required "
                                "sections: " + ", ".join(missing) + ". "
                                "Evidence collection is complete. "
                                "In the final drafting phase, produce all "
                                "six required Markdown sections in order, "
                                "including Executive Summary. Do not invent "
                                "evidence or request more tools."
                            ),
                        })
                        break
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
                            "using exactly the six required Markdown "
                            "sections, beginning with Executive Summary. "
                            "Use the available evidence only. "
                            "Disclose any incomplete summaries "
                            "and unresolved questions. "
                            "Do not request additional tools."
                        ),
                    })

                    response = self._request_model(
                        state=state,
                        messages=messages,
                        tools=[],
                        instructions=draft_instructions,
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
            # Preserve the failure reason without recording request
            # payloads, credentials, or a potentially sensitive traceback.
            error_type = type(exc).__name__
            error_message = str(exc).replace("\n", " ")[:500]

            state.add_trace(
                "INVESTIGATION_ERROR",
                {
                    "error_type": error_type,
                    "error_message": error_message,
                },
            )

            state.limitations.append(
                "Investigation interrupted: "
                f"{error_type}: {error_message}"
            )

            state.set_status("FAILED")

        return InvestigationResult(
            state=state,
            draft_report=draft_report,
            budget_snapshot=self.budget.snapshot(),
        )
