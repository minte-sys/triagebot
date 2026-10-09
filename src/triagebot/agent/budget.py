"""Limits that stop a run before it gets slow or expensive."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Budget:
    max_steps: int = 8
    max_seconds: float = 180
    max_tokens: int = 40_000

    def exceeded(self, steps: int, seconds: float, tokens: int) -> str | None:
        """Return why the budget is used up, or None if the run may continue."""
        if steps >= self.max_steps:
            return f"step limit reached ({self.max_steps} model calls)"
        if seconds >= self.max_seconds:
            return f"time limit reached ({self.max_seconds:.0f} s)"
        if tokens >= self.max_tokens:
            return f"token limit reached ({tokens} of {self.max_tokens})"
        return None
