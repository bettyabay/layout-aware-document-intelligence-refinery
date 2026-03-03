"""Abstract base class for extraction strategies.

This module defines the ``ExtractionStrategy`` ABC that all concrete extraction
strategies (fast text, layout-aware, vision-augmented) must implement. It
ensures a consistent interface for the extraction router to delegate to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from src.models.document_profile import DocumentProfile
from src.models.extracted_document import ExtractedDocument


class ExtractionStrategy(ABC):
    """Abstract base class for document extraction strategies.

    All extraction strategies must implement this interface to ensure consistent
    behavior across the pipeline. The router uses ``can_handle()`` to select
    strategies, and ``extract()`` to perform the actual extraction.

    Attributes:
        name: Human-readable name of the strategy (for logging).
    """

    def __init__(self, name: str) -> None:
        """Initialize the extraction strategy.

        Args:
            name: Human-readable name for this strategy.
        """
        self.name = name

    @abstractmethod
    def extract(self, document_path: str) -> ExtractedDocument:
        """Extract content from a document.

        This is the main extraction method that all strategies must implement.
        It should return an ``ExtractedDocument`` with text blocks, tables,
        figures, and optionally reading order.

        Args:
            document_path: Path to the PDF or document file to extract.

        Returns:
            ExtractedDocument with all extracted content and spatial metadata.

        Raises:
            FileNotFoundError: If the document file does not exist.
            ValueError: If the document cannot be processed by this strategy.
            RuntimeError: If extraction fails for any other reason.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.extract() must be implemented"
        )

    @abstractmethod
    def confidence_score(self, document_path: str) -> float:
        """Calculate confidence score for extraction on a document.

        This method should analyze the document and return a confidence score
        between 0.0 and 1.0 indicating how well this strategy can handle it.
        Higher scores indicate higher confidence that extraction will succeed.

        Args:
            document_path: Path to the document file.

        Returns:
            Confidence score between 0.0 and 1.0.

        Raises:
            FileNotFoundError: If the document file does not exist.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.confidence_score() must be implemented"
        )

    @abstractmethod
    def cost_estimate(self, document_path: str) -> Dict[str, float]:
        """Estimate extraction cost for a document.

        Returns a dictionary with cost breakdown, including:
        - ``total_cost_usd``: Total estimated cost in USD
        - ``cost_per_page``: Average cost per page
        - Additional strategy-specific cost components

        Args:
            document_path: Path to the document file.

        Returns:
            Dictionary with cost estimates. Must include ``total_cost_usd`` key.

        Raises:
            FileNotFoundError: If the document file does not exist.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.cost_estimate() must be implemented"
        )

    @abstractmethod
    def can_handle(self, profile: DocumentProfile) -> bool:
        """Check if this strategy can handle a document based on its profile.

        The router uses this method to determine which strategies are eligible
        for a given document. Multiple strategies may return True, in which case
        the router will select based on confidence scores and cost.

        Args:
            profile: DocumentProfile from the triage agent.

        Returns:
            True if this strategy can handle the document, False otherwise.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.can_handle() must be implemented"
        )

    def __repr__(self) -> str:
        """Return string representation of the strategy."""
        return f"{self.__class__.__name__}(name={self.name!r})"


__all__ = ["ExtractionStrategy"]
