"""Budget guard utilities for API-based extraction strategies.

This module provides a small helper class used by vision-augmented strategies
to track token usage and enforce per-document cost ceilings.

It is deliberately independent of any particular API provider. The caller is
responsible for converting API-specific ``usage`` metadata into
``prompt_tokens`` / ``completion_tokens`` counts and passing them to
``record_usage``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional


logger = logging.getLogger(__name__)


@dataclass
class BudgetGuard:
    """Track token usage and enforce a maximum dollar spend per document.

    Args:
        max_cost_usd: Hard ceiling per document in USD (e.g. 0.50).
        cost_per_1k_tokens_usd: Effective blended cost per 1,000 tokens for the
            chosen model. This does not need to be exact; it is used for
            *preventative* gating before hitting external limits.
    """

    max_cost_usd: float = 0.50
    cost_per_1k_tokens_usd: float = 0.002  # conservative default

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0

    # Simple backoff configuration for transient failures (e.g. rate limiting).
    max_retries: int = 3
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 8.0

    # Optional identifier for logging (e.g. document_id).
    context_id: Optional[str] = field(default=None)

    # ------------------------------------------------------------------ #
    # Cost tracking
    # ------------------------------------------------------------------ #
    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage from a single API call and update cost."""
        self.total_prompt_tokens += max(0, prompt_tokens)
        self.total_completion_tokens += max(0, completion_tokens)

        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        self.total_cost_usd = (total_tokens / 1000.0) * self.cost_per_1k_tokens_usd

        logger.info(
            "BudgetGuard usage update",
            extra={
                "context_id": self.context_id,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_cost_usd": self.total_cost_usd,
                "max_cost_usd": self.max_cost_usd,
            },
        )

    @property
    def remaining_budget_usd(self) -> float:
        """Return remaining budget before hitting ``max_cost_usd``."""
        return max(0.0, self.max_cost_usd - self.total_cost_usd)

    def can_spend_more(self) -> bool:
        """Return True if further API calls are allowed under the budget."""
        return self.total_cost_usd < self.max_cost_usd

    # ------------------------------------------------------------------ #
    # Backoff utilities
    # ------------------------------------------------------------------ #
    def get_backoff_time(self, attempt: int) -> float:
        """Compute exponential backoff time for the given attempt (0-based)."""
        # attempt=0 → base_backoff_seconds, attempt=1 → 2x, etc.
        delay = self.base_backoff_seconds * (2 ** attempt)
        return min(delay, self.max_backoff_seconds)

    def sleep_before_retry(self, attempt: int) -> None:
        """Sleep for an appropriate duration before retrying a request."""
        delay = self.get_backoff_time(attempt)
        logger.warning(
            "Rate limit or transient error; backing off",
            extra={
                "context_id": self.context_id,
                "attempt": attempt,
                "sleep_seconds": delay,
            },
        )
        time.sleep(delay)


__all__ = ["BudgetGuard"]

