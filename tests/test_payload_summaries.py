
import json

import pytest

from src.agent.payload_limits import (
    PayloadLimitError,
    serialize_model_evidence,
    serialize_tool_result,
)


def test_small_result_remains_complete():
    result = {"status": "PASS", "pnl_usd": 125.50}

    serialized = serialize_model_evidence(
        result,
        tool_name="get_pnl_attribution",
        evidence_index=0,
    )

    assert json.loads(serialized) == result


def test_oversized_result_has_explicit_warning():
    result = {
        "status": "PASS",
        "observations": [
            {"date": f"2026-09-{i:02d}", "value": i}
            for i in range(1, 500)
        ],
    }

    serialized = serialize_model_evidence(
        result,
        tool_name="get_pnl_attribution",
        evidence_index=3,
        max_chars=3000,
    )

    summary = json.loads(serialized)

    assert summary["evidence_complete"] is False
    assert summary["evidence_index"] == 3
    assert summary["tool_name"] == "get_pnl_attribution"
    assert summary["full_result_sha256"]
    assert "incomplete" in summary["warning"].lower()
    assert len(serialized) <= 3000


def test_original_serializer_still_rejects_oversized():
    with pytest.raises(PayloadLimitError):
        serialize_tool_result(
            {"data": "x" * 1000},
            max_chars=100,
        )


def test_invalid_numbers_are_rejected():
    with pytest.raises(PayloadLimitError):
        serialize_model_evidence(
            {"value": float("nan")},
            tool_name="get_risk_metrics",
            evidence_index=0,
        )
