"""Execution budget enforcement for the task agent."""

from __future__ import annotations

import time


class BudgetExceeded(RuntimeError):
    """Raised before an operation that would exceed a configured budget."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class BudgetGuard:
    """Track steps and estimated paid cost before any tool call starts."""

    def __init__(self, max_steps: int, max_cost_inr: float, max_seconds: float) -> None:
        self.max_steps = max_steps
        self.max_cost_inr = max_cost_inr
        self.max_seconds = max_seconds
        self.started = time.monotonic()
        self.steps = 0
        self.cost_inr = 0.0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def reserve(self, *, estimated_cost: float = 0.0, paid: bool = False) -> None:
        if self.steps >= self.max_steps:
            raise BudgetExceeded("max_steps", "The maximum number of execution steps was reached.")
        if self.elapsed >= self.max_seconds:
            raise BudgetExceeded("max_seconds", "The maximum execution time was reached.")
        if paid and self.cost_inr + estimated_cost > self.max_cost_inr:
            raise BudgetExceeded(
                "max_cost_inr",
                f"Estimated cost INR {estimated_cost:.6f} exceeds the remaining budget.",
            )
        self.steps += 1
        if paid:
            self.cost_inr += estimated_cost

    def ensure_time(self) -> None:
        """Fail after an operation that ran beyond the wall-clock budget."""
        if self.elapsed >= self.max_seconds:
            raise BudgetExceeded("max_seconds", "The maximum execution time was reached.")

    def reconcile(self, estimate: float, actual: float) -> None:
        self.cost_inr = max(0.0, self.cost_inr - estimate + actual)


def estimate_tool_cost(name: str, arguments: dict[str, object]) -> float:
    """Estimate Bodhan spend conservatively enough to guard a call."""
    if name == "synthesize":
        return len(str(arguments.get("text", ""))) * 6 / 10_000
    if name == "translate":
        return len(str(arguments.get("text", ""))) / 10_000
    return 0.0
