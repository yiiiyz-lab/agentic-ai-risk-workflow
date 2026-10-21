
import inspect
import json
import math
from dataclasses import dataclass
from datetime import date

from src.agent.budget import AgentBudget
from src.agent.investigation_state import InvestigationState

from src.tools.analyze_positions import analyze_positions
from src.tools.analyze_risk_factors import analyze_risk_factors
from src.tools.analyze_transactions import analyze_transactions
from src.tools.check_data_quality import check_data_quality
from src.tools.get_market_context import get_market_context
from src.tools.get_pnl_attribution import get_pnl_attribution
from src.tools.get_risk_metrics import get_risk_metrics
from src.tools.get_var_attribution import get_var_attribution
from src.tools.retrieve_risk_policy import retrieve_risk_policy
from src.tools.run_stress_test import run_stress_test


class ToolValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ToolSpec:
    function: object
    description: str
    parameters: dict


def integer(minimum, maximum):
    return {
        "type": "integer",
        "minimum": minimum,
        "maximum": maximum,
    }


def string():
    return {"type": "string"}


def array(item_schema, maximum=10):
    return {
        "type": "array",
        "items": item_schema,
        "minItems": 1,
        "maxItems": maximum,
    }


def object_schema(properties=None, required=None):
    return {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
        "additionalProperties": False,
    }


def optional(schema):
    return {"anyOf": [schema, {"type": "null"}]}


DATE = {
    "type": "string",
    "format": "date",
}

WINDOWS = array(integer(2, 252), maximum=5)
SCENARIO_DATES = array(DATE, maximum=10)

# Stress scenarios are deliberately limited to the existing
# ETF universe in this demo. The Python risk engine remains
# independent of these Agent-facing guardrails.
SYMBOLS = ("SPY", "QQQ", "IEF", "TLT", "LQD", "HYG")

SHOCKS = object_schema({
    symbol: {
        "type": "number",
        "minimum": -1,
        "maximum": 5,
    }
    for symbol in SYMBOLS
}, required=list(SYMBOLS))

SCENARIO = {
    "type": "object",
    "properties": {
        "id": string(),
        "name": string(),
        "type": {
            "type": "string",
            "enum": ["historical", "hypothetical"],
        },
        "start_date": DATE,
        "end_date": DATE,
        "shocks": SHOCKS,
    },
    "required": ["id", "name", "type"],
    "additionalProperties": False,
}


TOOL_SPECS = {
    "get_risk_metrics": ToolSpec(
        get_risk_metrics,
        "Retrieve portfolio risk metrics for the investigation date.",
        object_schema(),
    ),
    "analyze_positions": ToolSpec(
        analyze_positions,
        "Analyze positions, exposure and concentration.",
        object_schema(),
    ),
    "analyze_transactions": ToolSpec(
        analyze_transactions,
        "Analyze historical trades and their portfolio impact.",
        object_schema({
            "lookback_days": integer(1, 252),
        }),
    ),
    "get_pnl_attribution": ToolSpec(
        get_pnl_attribution,
        "Attribute portfolio P&L to instruments and asset classes.",
        object_schema({
            "lookback_days": integer(1, 252),
            "start_date": optional(DATE),
        }),
    ),
    "get_var_attribution": ToolSpec(
        get_var_attribution,
        "Explain Historical Simulation VaR and ES contributions.",
        object_schema(),
    ),
    "analyze_risk_factors": ToolSpec(
        analyze_risk_factors,
        "Analyze historical ETF returns, volatility and correlations.",
        object_schema({
            "windows": WINDOWS,
            "scenario_dates": optional(SCENARIO_DATES),
        }),
    ),
    "run_stress_test": ToolSpec(
        run_stress_test,
        "Run selected historical or hypothetical portfolio stress scenarios.",
        object_schema({
            "scenarios": array(SCENARIO, maximum=5),
        }, required=["scenarios"]),
    ),
    "check_data_quality": ToolSpec(
        check_data_quality,
        "Check historical data quality and portfolio reconciliation.",
        object_schema({
            "lookback_days": integer(1, 252),
        }),
    ),
    "retrieve_risk_policy": ToolSpec(
        retrieve_risk_policy,
        "Retrieve configured risk thresholds and policy references.",
        object_schema({
            "rule_id": optional(string()),
            "metric": optional(string()),
        }),
    ),
    "get_market_context": ToolSpec(
        get_market_context,
        "Retrieve historical ETF performance and curated market articles.",
        object_schema({
            "windows": WINDOWS,
            "event_lookback_days": integer(0, 365),
            "topic_tags": optional(
                array(string(), maximum=10)
            ),
        }),
    ),
}


def _validate_date(value, as_of):
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ToolValidationError(
            "Invalid ISO date"
        ) from exc

    if parsed.isoformat() != value:
        raise ToolValidationError(
            "Date must use YYYY-MM-DD"
        )

    if parsed > date.fromisoformat(as_of):
        raise ToolValidationError(
            "Future dates are not allowed"
        )


def _validate(value, schema, as_of):
    if "anyOf" in schema:
        if value is None:
            return
        _validate(value, schema["anyOf"][0], as_of)
        return

    kind = schema["type"]

    if kind == "integer":
        if type(value) is not int:
            raise ToolValidationError("Expected integer")
        if not schema["minimum"] <= value <= schema["maximum"]:
            raise ToolValidationError("Integer outside allowed range")

    elif kind == "number":
        if type(value) not in (int, float):
            raise ToolValidationError("Expected number")
        if not math.isfinite(value):
            raise ToolValidationError("Non-finite number")
        if not schema["minimum"] <= value <= schema["maximum"]:
            raise ToolValidationError("Number outside allowed range")

    elif kind == "string":
        if not isinstance(value, str) or not value.strip():
            raise ToolValidationError("Expected non-empty string")
        if len(value) > 256:
            raise ToolValidationError("String too long")
        if schema.get("format") == "date":
            _validate_date(value, as_of)
        if "enum" in schema and value not in schema["enum"]:
            raise ToolValidationError("Invalid enum value")

    elif kind == "array":
        if not isinstance(value, list):
            raise ToolValidationError("Expected array")
        if not schema["minItems"] <= len(value) <= schema["maxItems"]:
            raise ToolValidationError("Invalid array length")
        for item in value:
            _validate(item, schema["items"], as_of)

    elif kind == "object":
        if not isinstance(value, dict):
            raise ToolValidationError("Expected object")

        properties = schema["properties"]
        if set(value) - set(properties):
            raise ToolValidationError("Unknown argument")

        if set(schema["required"]) - set(value):
            raise ToolValidationError("Missing required argument")

        for key, item in value.items():
            _validate(item, properties[key], as_of)

    else:
        raise ToolValidationError("Unsupported parameter schema")


def _validate_scenarios(scenarios, as_of):
    seen = set()

    for scenario in scenarios:
        scenario_id = scenario["id"]

        if scenario_id in seen:
            raise ToolValidationError("Duplicate scenario ID")
        seen.add(scenario_id)

        if scenario["type"] == "historical":
            if (
                "start_date" not in scenario
                or "end_date" not in scenario
                or "shocks" in scenario
            ):
                raise ToolValidationError(
                    "Historical scenario requires start/end dates"
                )

            start = scenario["start_date"]
            end = scenario["end_date"]

            if not start < end < as_of:
                raise ToolValidationError(
                    "Historical scenario must end before as-of date"
                )

        else:
            if (
                "shocks" not in scenario
                or "start_date" in scenario
                or "end_date" in scenario
            ):
                raise ToolValidationError(
                    "Hypothetical scenario requires shocks"
                )


class ToolRegistry:
    def __init__(self, specs=None):
        self._specs = (
            dict(TOOL_SPECS)
            if specs is None
            else dict(specs)
        )

    def list_tools(self):
        return [
            {
                "name": name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
            for name, spec in self._specs.items()
        ]

    def validate_call(self, name, arguments, state):
        if name not in self._specs:
            raise ToolValidationError(
                f"Tool not permitted: {name}"
            )

        if not isinstance(state, InvestigationState):
            raise ToolValidationError(
                "Invalid investigation state"
            )

        spec = self._specs[name]

        _validate(
            arguments,
            spec.parameters,
            state.as_of_date,
        )

        if name == "run_stress_test":
            _validate_scenarios(
                arguments["scenarios"],
                state.as_of_date,
            )

        # Some risk-factor diagnostics require dates strictly
        # before the investigation date.
        if name == "analyze_risk_factors":
            for value in arguments.get("scenario_dates") or []:
                if value >= state.as_of_date:
                    raise ToolValidationError(
                        "Scenario dates must precede as-of date"
                    )

        return spec

    def execute(
        self,
        name,
        arguments,
        state,
        budget,
    ):
        if not isinstance(budget, AgentBudget):
            raise ToolValidationError("Invalid budget")

        if state.status != "RUNNING":
            raise ToolValidationError(
                "Investigation must be RUNNING"
            )

        spec = self.validate_call(
            name, arguments, state
        )

        budget.check_tool_call()

        # Record the attempted execution before invoking
        # a potentially failing tool.
        budget.record_tool_call()

        state.add_trace(
            "TOOL_CALL",
            {
                "tool_name": name,
                "arguments": arguments,
            },
        )

        try:
            # Inspect the registered tool's accepted parameters.
            signature = inspect.signature(spec.function)

            # Only inject trusted context supported by the function.
            trusted_context = {
                "portfolio_id": state.portfolio_id,
                "as_of_date": state.as_of_date,
            }

            context_arguments = {
                name: value
                for name, value in trusted_context.items()
                if name in signature.parameters
                and signature.parameters[name].kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }

            result = spec.function(
                **context_arguments,
                **arguments,
            )

            # Require tool evidence to be JSON-serializable.
            json.dumps(result, allow_nan=False)

        except Exception as exc:
            state.add_trace(
                "TOOL_ERROR",
                {
                    "tool_name": name,
                    "error_type": type(exc).__name__,
                },
            )
            raise

        state.add_evidence(name, result)

        state.add_trace(
            "TOOL_SUCCESS",
            {
                "tool_name": name,
                "evidence_index": len(state.evidence) - 1,
            },
        )

        return result
