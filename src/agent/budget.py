
from dataclasses import dataclass, field
from decimal import Decimal


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class AgentBudget:
    max_llm_calls: int = 8
    max_tool_calls: int = 20
    max_output_tokens_per_call: int = 1500
    max_estimated_cost_usd: Decimal = Decimal("0.25")

    llm_calls: int = 0
    tool_calls: int = 0
    estimated_cost_usd: Decimal = field(
        default_factory=lambda: Decimal("0")
    )

    def __post_init__(self):
        if (
            self.max_llm_calls <= 0
            or self.max_tool_calls <= 0
            or self.max_output_tokens_per_call <= 0
            or self.max_estimated_cost_usd <= 0
        ):
            raise ValueError("Budget limits must be positive")

    def check_llm_call(self, reserved_cost_usd):
        reserved = Decimal(str(reserved_cost_usd))

        if reserved < 0:
            raise ValueError("Reserved cost cannot be negative")

        if self.llm_calls >= self.max_llm_calls:
            raise BudgetExceededError(
                "Maximum LLM calls reached"
            )

        if (
            self.estimated_cost_usd + reserved
            > self.max_estimated_cost_usd
        ):
            raise BudgetExceededError(
                "Estimated spending budget exceeded"
            )

    def record_llm_call(self, cost_usd):
        cost = Decimal(str(cost_usd))

        if cost < 0:
            raise ValueError("Cost cannot be negative")

        self.estimated_cost_usd += cost
        self.llm_calls += 1

    def check_tool_call(self):
        if self.tool_calls >= self.max_tool_calls:
            raise BudgetExceededError(
                "Maximum tool calls reached"
            )

    def record_tool_call(self):
        self.check_tool_call()
        self.tool_calls += 1

    def snapshot(self):
        return {
            "llm_calls": self.llm_calls,
            "max_llm_calls": self.max_llm_calls,
            "tool_calls": self.tool_calls,
            "max_tool_calls": self.max_tool_calls,
            "max_output_tokens_per_call": (
                self.max_output_tokens_per_call
            ),
            "estimated_cost_usd": str(
                self.estimated_cost_usd
            ),
            "max_estimated_cost_usd": str(
                self.max_estimated_cost_usd
            ),
        }
