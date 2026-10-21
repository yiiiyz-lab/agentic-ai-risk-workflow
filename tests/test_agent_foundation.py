
from decimal import Decimal

import pytest

from src.agent.budget import (
    AgentBudget,
    BudgetExceededError,
)
from src.agent.investigation_state import (
    InvestigationState,
)


@pytest.fixture
def state():
    return InvestigationState(
        portfolio_id="PORT_001",
        as_of_date="2026-09-30",
        exception_id=(
            "2026-09-30_EQUITY_EXPOSURE_LIMIT"
        ),
        rule_id="EQUITY_EXPOSURE_LIMIT",
        exception_status="BREACH",
    )


def test_initial_investigation_state(state):
    assert state.status == "PENDING"
    assert state.trace == []
    assert state.evidence == []


def test_status_transition_recorded(state):
    state.set_status("RUNNING")
    assert state.status == "RUNNING"
    assert state.trace[-1]["event_type"] == (
        "STATUS_CHANGE"
    )


def test_invalid_status_rejected(state):
    with pytest.raises(ValueError):
        state.set_status("UNKNOWN")


def test_pass_exception_rejected():
    with pytest.raises(ValueError):
        InvestigationState(
            "PORT_001",
            "2026-09-30",
            "exception",
            "rule",
            "PASS",
        )


def test_invalid_date_rejected():
    with pytest.raises(ValueError):
        InvestigationState(
            "PORT_001",
            "2026-09-31",
            "exception",
            "rule",
            "BREACH",
        )


def test_evidence_recorded(state):
    state.add_evidence(
        "retrieve_risk_policy",
        {"breach_threshold": 0.65},
    )
    assert len(state.evidence) == 1
    assert state.trace[-1]["event_type"] == (
        "EVIDENCE_RECORDED"
    )


def test_finding_requires_evidence(state):
    with pytest.raises(ValueError):
        state.add_finding(
            "Equity exposure exceeds its limit",
            [0],
        )


def test_finding_with_valid_evidence(state):
    state.add_evidence(
        "retrieve_risk_policy",
        {"breach_threshold": 0.65},
    )
    state.add_finding(
        "Policy threshold is 65%",
        [0],
    )
    assert len(state.findings) == 1
    assert state.findings[0]["review_status"] == (
        "PENDING"
    )


def test_budget_initial_state():
    budget = AgentBudget()
    assert budget.llm_calls == 0
    assert budget.tool_calls == 0
    assert budget.estimated_cost_usd == Decimal("0")


def test_llm_call_limit():
    budget = AgentBudget(max_llm_calls=1)
    budget.check_llm_call("0.01")
    budget.record_llm_call("0.01")

    with pytest.raises(BudgetExceededError):
        budget.check_llm_call("0.01")


def test_spending_limit():
    budget = AgentBudget(
        max_estimated_cost_usd=Decimal("0.05")
    )
    budget.record_llm_call("0.04")

    with pytest.raises(BudgetExceededError):
        budget.check_llm_call("0.02")


def test_tool_call_limit():
    budget = AgentBudget(max_tool_calls=2)
    budget.record_tool_call()
    budget.record_tool_call()

    with pytest.raises(BudgetExceededError):
        budget.record_tool_call()


def test_budget_snapshot():
    budget = AgentBudget()
    snapshot = budget.snapshot()

    assert snapshot["max_llm_calls"] == 8
    assert snapshot["max_tool_calls"] == 20
    assert snapshot["max_estimated_cost_usd"] == "0.25"
