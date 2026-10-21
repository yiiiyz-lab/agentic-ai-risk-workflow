
"""Deterministic limits and summaries for model-facing tool evidence."""

import json
import hashlib


class PayloadLimitError(ValueError):
    """Evidence cannot be safely serialized within configured limits."""


MAX_TOOL_RESULT_CHARS = 12000

# Limit nested expansion when constructing an oversized-result summary.
MAX_SUMMARY_DEPTH = 3
MAX_DICT_ITEMS = 12
MAX_LIST_ITEMS = 5
MAX_STRING_CHARS = 300


def _json_dumps(value):
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise PayloadLimitError(
            "Tool result is not JSON serializable"
        ) from exc


def serialize_tool_result(
    result,
    max_chars=MAX_TOOL_RESULT_CHARS,
):
    """Serialize complete evidence, rejecting oversized results."""
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("max_chars must be positive")

    serialized = _json_dumps(result)

    if len(serialized) > max_chars:
        raise PayloadLimitError(
            "Tool result exceeds model-facing size limit"
        )

    return serialized


def _summarize(value, depth=0):
    """Produce a bounded structural view without inventing values."""
    if depth >= MAX_SUMMARY_DEPTH:
        if isinstance(value, dict):
            return {
                "_type": "object",
                "_total_keys": len(value),
                "_details_omitted": True,
            }

        if isinstance(value, list):
            return {
                "_type": "array",
                "_total_items": len(value),
                "_details_omitted": True,
            }

        if isinstance(value, str) and len(value) > MAX_STRING_CHARS:
            return {
                "_type": "string",
                "_length": len(value),
                "_details_omitted": True,
            }

        return value

    if isinstance(value, dict):
        items = list(value.items())

        result = {
            str(key): _summarize(item, depth + 1)
            for key, item in items[:MAX_DICT_ITEMS]
        }

        if len(items) > MAX_DICT_ITEMS:
            result["_summary_metadata"] = {
                "total_keys": len(items),
                "omitted_keys": len(items) - MAX_DICT_ITEMS,
            }

        return result

    if isinstance(value, list):
        items = [
            _summarize(item, depth + 1)
            for item in value[:MAX_LIST_ITEMS]
        ]

        if len(value) > MAX_LIST_ITEMS:
            return {
                "_type": "array",
                "_total_items": len(value),
                "_sample": items,
                "_omitted_items": len(value) - MAX_LIST_ITEMS,
            }

        return items

    if isinstance(value, str) and len(value) > MAX_STRING_CHARS:
        return {
            "_type": "string",
            "_length": len(value),
            "_preview": value[:MAX_STRING_CHARS],
            "_details_omitted": True,
        }

    return value


def serialize_model_evidence(
    result,
    *,
    tool_name,
    evidence_index,
    max_chars=MAX_TOOL_RESULT_CHARS,
):
    """Serialize evidence for a model, summarizing oversized outputs.

    The original result must already be retained by InvestigationState.
    """
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("max_chars must be positive")

    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("tool_name must be non-empty")

    if type(evidence_index) is not int or evidence_index < 0:
        raise ValueError("evidence_index must be non-negative")

    full_json = _json_dumps(result)

    if len(full_json) <= max_chars:
        return full_json

    digest = hashlib.sha256(
        full_json.encode("utf-8")
    ).hexdigest()

    summary = {
        "tool_name": tool_name,
        "evidence_index": evidence_index,
        "evidence_complete": False,
        "summary_type": "bounded_structural_preview",
        "full_result_sha256": digest,
        "full_result_characters": len(full_json),
        "warning": (
            "This is an incomplete structural preview. "
            "Omitted details may include material financial "
            "values, reconciliation failures, or limitations. "
            "Do not conclude that omitted checks passed. "
            "Refer to full stored evidence for review."
        ),
        "preview": _summarize(result),
    }

    serialized = _json_dumps(summary)

    if len(serialized) > max_chars:
        raise PayloadLimitError(
            "Evidence summary exceeds model-facing size limit"
        )

    return serialized
