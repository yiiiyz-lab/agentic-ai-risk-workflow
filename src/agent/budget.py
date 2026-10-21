
"""Per-investigation API and tool execution budget."""

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from threading import RLock
from uuid import uuid4


class BudgetExceededError(RuntimeError):
    pass


def _money(value):
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("Invalid monetary amount") from exc

    if not amount.is_finite() or amount < 0:
        raise ValueError("Amount must be finite and non-negative")

    return amount


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

    _reservations: dict = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: RLock = field(
        default_factory=RLock, init=False, repr=False
    )

    def __post_init__(self):
        for name in (
            "max_llm_calls",
            "max_tool_calls",
            "max_output_tokens_per_call",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")

        self.max_estimated_cost_usd = _money(
            self.max_estimated_cost_usd
        )
        self.estimated_cost_usd = _money(
            self.estimated_cost_usd
        )

        if self.max_estimated_cost_usd == 0:
            raise ValueError("Budget limit must be positive")

        if type(self.llm_calls) is not int or self.llm_calls < 0:
            raise ValueError("Invalid LLM call count")
        if type(self.tool_calls) is not int or self.tool_calls < 0:
            raise ValueError("Invalid tool call count")

    def _reserved_total(self):
        return sum(
            self._reservations.values(),
            Decimal("0"),
        )

    def _check_llm_locked(self, amount):
        if (
            self.llm_calls + len(self._reservations)
            >= self.max_llm_calls
        ):
            raise BudgetExceededError(
                "Maximum LLM calls reached or reserved"
            )

        projected = (
            self.estimated_cost_usd
            + self._reserved_total()
            + amount
        )

        if projected > self.max_estimated_cost_usd:
            raise BudgetExceededError(
                "Estimated spending budget exceeded"
            )

    def check_llm_call(self, reserved_cost_usd):
        amount = _money(reserved_cost_usd)
        with self._lock:
            self._check_llm_locked(amount)

    def reserve_llm_call(self, estimated_cost_usd):
        """Atomically reserve one call and its estimated cost."""
        amount = _money(estimated_cost_usd)

        with self._lock:
            self._check_llm_locked(amount)
            reservation_id = str(uuid4())
            self._reservations[reservation_id] = amount
            return reservation_id

    def settle_llm_call(self, reservation_id, actual_cost_usd):
        """
        Replace a reservation with measured estimated cost.

        The call counts even if actual cost exceeds the
        reserved estimate. Any overrun remains visible.
        """
        amount = _money(actual_cost_usd)

        with self._lock:
            if reservation_id not in self._reservations:
                raise ValueError("Unknown reservation")

            reserved = self._reservations.pop(reservation_id)
            self.llm_calls += 1
            self.estimated_cost_usd += amount

            return {
                "reserved_cost_usd": str(reserved),
                "actual_estimated_cost_usd": str(amount),
                "over_reservation": amount > reserved,
                "over_budget": (
                    self.estimated_cost_usd
                    + self._reserved_total()
                    > self.max_estimated_cost_usd
                ),
            }

    def cancel_llm_reservation(self, reservation_id):
        """
        Cancel only when no billable API request was sent.

        After an uncertain network failure, do not cancel
        until usage/billing status is resolved.
        """
        with self._lock:
            if reservation_id not in self._reservations:
                raise ValueError("Unknown reservation")
            del self._reservations[reservation_id]

    def record_llm_call(self, cost_usd):
        """Legacy direct accounting; prefer reserve/settle."""
        amount = _money(cost_usd)

        with self._lock:
            self._check_llm_locked(amount)
            self.llm_calls += 1
            self.estimated_cost_usd += amount

    def check_tool_call(self):
        with self._lock:
            if self.tool_calls >= self.max_tool_calls:
                raise BudgetExceededError(
                    "Maximum tool calls reached"
                )

    def record_tool_call(self):
        with self._lock:
            if self.tool_calls >= self.max_tool_calls:
                raise BudgetExceededError(
                    "Maximum tool calls reached"
                )
            self.tool_calls += 1

    def snapshot(self):
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "max_llm_calls": self.max_llm_calls,
                "tool_calls": self.tool_calls,
                "max_tool_calls": self.max_tool_calls,
                "max_output_tokens_per_call":
                    self.max_output_tokens_per_call,
                "estimated_cost_usd":
                    str(self.estimated_cost_usd),
                "max_estimated_cost_usd":
                    str(self.max_estimated_cost_usd),
                "active_reservations":
                    len(self._reservations),
                "reserved_cost_usd":
                    str(self._reserved_total()),
            }
