
"""Deterministic limits for model-facing tool evidence."""

import json


class PayloadLimitError(ValueError):
    pass


MAX_TOOL_RESULT_CHARS = 12000


def serialize_tool_result(result, max_chars=MAX_TOOL_RESULT_CHARS):
    """
    Serialize tool output for an LLM.

    Reject oversized evidence instead of silently truncating
    JSON or removing numerical fields.
    """
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("max_chars must be positive")

    try:
        serialized = json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise PayloadLimitError(
            "Tool result is not JSON serializable"
        ) from exc

    if len(serialized) > max_chars:
        raise PayloadLimitError(
            "Tool result exceeds model-facing size limit"
        )

    return serialized
