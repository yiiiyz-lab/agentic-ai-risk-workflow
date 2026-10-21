
"""Safe, offline-first OpenAI API configuration."""

import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


class APIConfigurationError(ValueError):
    pass


def _positive_decimal(value, name):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise APIConfigurationError(
            f"Invalid {name}"
        ) from exc

    if not result.is_finite() or result <= 0:
        raise APIConfigurationError(
            f"{name} must be positive and finite"
        )

    return result


@dataclass(frozen=True, repr=False)
class OpenAIConfig:
    enabled: bool
    model: str
    api_key: str | None
    input_price_per_million: Decimal
    output_price_per_million: Decimal

    def __repr__(self):
        return (
            "OpenAIConfig("
            f"enabled={self.enabled}, "
            f"model={self.model!r}, "
            "api_key=<redacted>, "
            "pricing_configured=True)"
        )

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ

        enabled_raw = env.get(
            "OPENAI_ENABLE_API", "false"
        ).strip().lower()

        if enabled_raw not in ("true", "false"):
            raise APIConfigurationError(
                "OPENAI_ENABLE_API must be true or false"
            )

        enabled = enabled_raw == "true"
        model = env.get("OPENAI_MODEL", "").strip()
        key = env.get("OPENAI_API_KEY", "").strip()

        if enabled and (not model or not key):
            raise APIConfigurationError(
                "Enabled API requires model and API key"
            )

        input_price = _positive_decimal(
            env.get(
                "OPENAI_INPUT_PRICE_PER_MILLION",
                "1",
            ),
            "input token price",
        )
        output_price = _positive_decimal(
            env.get(
                "OPENAI_OUTPUT_PRICE_PER_MILLION",
                "1",
            ),
            "output token price",
        )

        return cls(
            enabled=enabled,
            model=model,
            api_key=key or None,
            input_price_per_million=input_price,
            output_price_per_million=output_price,
        )

    def require_enabled(self):
        if not self.enabled:
            raise APIConfigurationError(
                "Live API access is disabled"
            )

    def estimate_cost(self, input_tokens, output_tokens):
        for count in (input_tokens, output_tokens):
            if type(count) is not int or count < 0:
                raise ValueError(
                    "Token counts must be non-negative integers"
                )

        return (
            Decimal(input_tokens)
            * self.input_price_per_million
            + Decimal(output_tokens)
            * self.output_price_per_million
        ) / Decimal("1000000")
