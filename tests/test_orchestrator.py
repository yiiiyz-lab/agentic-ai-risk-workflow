
import json

from src.agent.budget import AgentBudget
from src.agent.openai_adapter import OfflineOpenAIAdapter
from src.agent.orchestrator import InvestigationOrchestrator


def test_offline_investigation_with_real_tools():
    responses = [
        {
            "id": "resp_001",
            "output": [{
                "type": "function_call",
                "call_id": "call_001",
                "name": "retrieve_risk_policy",
                "arguments": json.dumps({
                    "rule_id": "EQUITY_EXPOSURE_LIMIT",
                    "metric": None,
                }),
            }],
        },
        {
            "id": "resp_002",
            "output": [{
                "type": "function_call",
                "call_id": "call_002",
                "name": "get_risk_metrics",
                "arguments": "{}",
            }],
        },
        {
            "id": "resp_003",
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": (
                        "Draft investigation: equity exposure "
                        "requires risk manager review."
                    ),
                }],
            }],
        },
    ]

    adapter = OfflineOpenAIAdapter(responses)
    budget = AgentBudget()

    orchestrator = InvestigationOrchestrator(
        adapter=adapter,
        budget=budget,
    )

    result = orchestrator.investigate({
        "portfolio_id": "PORT_001",
        "date": "2026-09-30",
        "exception_id": (
            "2026-09-30_EQUITY_EXPOSURE_LIMIT"
        ),
        "rule_id": "EQUITY_EXPOSURE_LIMIT",
        "status": "BREACH",
        "metric": "equity_exposure_pct",
        "metric_value": 0.704633,
    })

    assert result.state.status == "REQUIRES_REVIEW"
    assert len(result.state.evidence) == 2
    assert result.budget_snapshot["tool_calls"] == 2
    assert result.budget_snapshot["llm_calls"] == 3
    assert "risk manager review" in result.draft_report

    assert [
        item["tool_name"]
        for item in result.state.evidence
    ] == [
        "retrieve_risk_policy",
        "get_risk_metrics",
    ]
