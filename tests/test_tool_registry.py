
import pytest

from src.agent.budget import (
    AgentBudget,
    BudgetExceededError,
)
from src.agent.investigation_state import InvestigationState
from src.agent.tool_registry import (
    ToolRegistry,
    ToolSpec,
    ToolValidationError,
    TOOL_SPECS,
    object_schema,
)


@pytest.fixture
def state():
    result = InvestigationState(
        portfolio_id="PORT_001",
        as_of_date="2026-09-30",
        exception_id="2026-09-30_EQUITY_EXPOSURE_LIMIT",
        rule_id="EQUITY_EXPOSURE_LIMIT",
        exception_status="BREACH",
    )
    result.set_status("RUNNING")
    return result


def test_ten_tools_registered():
    registry = ToolRegistry()
    assert len(registry.list_tools()) == 10


def test_tool_names_unique():
    registry = ToolRegistry()
    names = [item["name"] for item in registry.list_tools()]
    assert len(names) == len(set(names))


def test_unknown_tool_rejected(state):
    registry = ToolRegistry()
    with pytest.raises(ToolValidationError):
        registry.validate_call("delete_files", {}, state)


def test_file_path_injection_rejected(state):
    registry = ToolRegistry()
    with pytest.raises(ToolValidationError):
        registry.validate_call(
            "get_risk_metrics",
            {"metrics_path": "/tmp/other.csv"},
            state,
        )


def test_portfolio_override_rejected(state):
    registry = ToolRegistry()
    with pytest.raises(ToolValidationError):
        registry.validate_call(
            "analyze_positions",
            {"portfolio_id": "OTHER"},
            state,
        )


def test_date_override_rejected(state):
    registry = ToolRegistry()
    with pytest.raises(ToolValidationError):
        registry.validate_call(
            "analyze_positions",
            {"as_of_date": "2026-09-29"},
            state,
        )


def test_valid_lookback(state):
    registry = ToolRegistry()
    registry.validate_call(
        "analyze_transactions",
        {"lookback_days": 20},
        state,
    )


def test_invalid_lookback(state):
    registry = ToolRegistry()
    with pytest.raises(ToolValidationError):
        registry.validate_call(
            "analyze_transactions",
            {"lookback_days": -1},
            state,
        )


def test_future_start_date_rejected(state):
    registry = ToolRegistry()
    with pytest.raises(ToolValidationError):
        registry.validate_call(
            "get_pnl_attribution",
            {"start_date": "2026-10-01"},
            state,
        )


def test_future_scenario_date_rejected(state):
    registry = ToolRegistry()
    with pytest.raises(ToolValidationError):
        registry.validate_call(
            "analyze_risk_factors",
            {"scenario_dates": ["2026-09-30"]},
            state,
        )


def test_valid_historical_stress(state):
    registry = ToolRegistry()
    registry.validate_call(
        "run_stress_test",
        {
            "scenarios": [{
                "id": "COVID",
                "name": "COVID Sell-Off",
                "type": "historical",
                "start_date": "2020-02-19",
                "end_date": "2020-03-23",
            }]
        },
        state,
    )


def test_future_historical_stress_rejected(state):
    registry = ToolRegistry()
    with pytest.raises(ToolValidationError):
        registry.validate_call(
            "run_stress_test",
            {
                "scenarios": [{
                    "id": "FUTURE",
                    "name": "Future Shock",
                    "type": "historical",
                    "start_date": "2026-09-29",
                    "end_date": "2026-09-30",
                }]
            },
            state,
        )


def test_incomplete_hypothetical_stress_rejected(state):
    registry = ToolRegistry()
    with pytest.raises(ToolValidationError):
        registry.validate_call(
            "run_stress_test",
            {
                "scenarios": [{
                    "id": "SHOCK",
                    "name": "Equity Shock",
                    "type": "hypothetical",
                    "shocks": {"SPY": -0.10},
                }]
            },
            state,
        )


def test_tool_execution_and_trace(state):
    calls = []

    def mock_tool(portfolio_id, as_of_date):
        calls.append((portfolio_id, as_of_date))
        return {"status": "PASS", "value": 42}

    registry = ToolRegistry({
        "mock_tool": ToolSpec(
            mock_tool,
            "Mock tool",
            object_schema(),
        )
    })

    budget = AgentBudget()
    result = registry.execute(
        "mock_tool", {}, state, budget
    )

    assert result["value"] == 42
    assert calls == [("PORT_001", "2026-09-30")]
    assert budget.tool_calls == 1
    assert len(state.evidence) == 1

    event_types = [
        event["event_type"] for event in state.trace
    ]
    assert "TOOL_CALL" in event_types
    assert "TOOL_SUCCESS" in event_types


def test_tool_budget_enforced(state):
    def mock_tool(portfolio_id, as_of_date):
        return {"ok": True}

    registry = ToolRegistry({
        "mock_tool": ToolSpec(
            mock_tool,
            "Mock",
            object_schema(),
        )
    })

    budget = AgentBudget(max_tool_calls=1)

    registry.execute("mock_tool", {}, state, budget)

    with pytest.raises(BudgetExceededError):
        registry.execute("mock_tool", {}, state, budget)

    assert budget.tool_calls == 1


def test_tool_failure_recorded(state):
    def failing_tool(portfolio_id, as_of_date):
        raise RuntimeError("Simulated failure")

    registry = ToolRegistry({
        "failing_tool": ToolSpec(
            failing_tool,
            "Mock failure",
            object_schema(),
        )
    })

    budget = AgentBudget()

    with pytest.raises(RuntimeError):
        registry.execute(
            "failing_tool", {}, state, budget
        )

    assert budget.tool_calls == 1
    assert state.evidence == []
    assert state.trace[-1]["event_type"] == "TOOL_ERROR"


def test_investigation_must_be_running(state):
    registry = ToolRegistry()
    state.set_status("COMPLETED")

    with pytest.raises(ToolValidationError):
        registry.execute(
            "get_risk_metrics",
            {},
            state,
            AgentBudget(),
        )

def test_tool_without_as_of_date(state):
    received = {}

    def mock_policy_tool(portfolio_id, rule_id=None):
        received["portfolio_id"] = portfolio_id
        received["rule_id"] = rule_id
        return {"status": "PASS"}

    registry = ToolRegistry({
        "mock_policy_tool": ToolSpec(
            mock_policy_tool,
            "Mock policy lookup",
            object_schema({
                "rule_id": {"type": "string"},
            }),
        )
    })

    result = registry.execute(
        "mock_policy_tool",
        {"rule_id": "EQUITY_EXPOSURE_LIMIT"},
        state,
        AgentBudget(),
    )

    assert result["status"] == "PASS"
    assert received == {
        "portfolio_id": "PORT_001",
        "rule_id": "EQUITY_EXPOSURE_LIMIT",
    }