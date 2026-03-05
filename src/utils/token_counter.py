"""Token counting utilities for chunking.

This module provides token counting functionality using tiktoken (OpenAI-compatible)
with a fallback to simple word/4 approximation.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import tiktoken
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning(
        "tiktoken not available. Using word/4 approximation for token counting."
    )


class TokenCounter:
    """Token counter with tiktoken support and fallback.

    Uses tiktoken for OpenAI-compatible token counting when available,
    otherwise falls back to word/4 approximation.
    """

    def __init__(self, model: str = "gpt-4"):
        """Initialize the TokenCounter.

        Args:
            model: OpenAI model name for tiktoken encoding. Defaults to "gpt-4".
                Ignored if tiktoken is not available.
        """
        self.model = model
        self.encoding: Optional[tiktoken.Encoding] = None

        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.encoding_for_model(model)
                logger.debug(f"Initialized tiktoken encoding for model: {model}")
            except KeyError:
                logger.warning(
                    f"Model {model} not found in tiktoken. Using default encoding."
                )
                try:
                    self.encoding = tiktoken.get_encoding("cl100k_base")
                except Exception as e:
                    logger.error(f"Failed to initialize tiktoken: {e}")
                    self.encoding = None
        else:
            logger.info("Using word/4 approximation for token counting")

    def count(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0

        if self.encoding is not None:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logger.warning(f"tiktoken encoding failed: {e}. Using fallback.")
                return self._fallback_count(text)

        return self._fallback_count(text)

    def _fallback_count(self, text: str) -> int:
        """Fallback token counting using word/4 approximation.

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated token count (words / 4).
        """
        # Simple approximation: words / 4
        # This is a rough estimate that works reasonably well for English text
        words = len(text.split())
        return max(1, words // 4)  # At least 1 token


# Global token counter instance
_default_counter: Optional[TokenCounter] = None


def get_token_counter(model: str = "gpt-4") -> TokenCounter:
    """Get or create the default token counter instance.

    Args:
        model: OpenAI model name for tiktoken encoding.

    Returns:
        TokenCounter instance.
    """
    global _default_counter
    if _default_counter is None:
        _default_counter = TokenCounter(model=model)
    return _default_counter


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Convenience function to count tokens in text.

    Args:
        text: Text to count tokens for.
        model: OpenAI model name for tiktoken encoding.

    Returns:
        Estimated token count.
    """
    counter = get_token_counter(model)
    return counter.count(text)
