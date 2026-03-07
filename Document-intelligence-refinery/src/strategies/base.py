"""Base extraction strategy interface."""

from abc import ABC, abstractmethod
from pathlib import Path

from src.models import DocumentProfile, ExtractedDocument


class ExtractionStrategy(ABC):
    """Abstract base class for extraction strategies."""

    name: str

    @abstractmethod
    def extract(
        self, pdf_path: Path, profile: DocumentProfile, rules: dict
    ) -> tuple[ExtractedDocument, float, float]:
        """
        Extract document content.
        
        Returns:
            tuple: (extracted_document, confidence_score, cost_estimate_usd)
        """
        pass

    def confidence_score(
        self, extracted: ExtractedDocument, profile: DocumentProfile, rules: dict
    ) -> float:
        """Calculate confidence score for extraction."""
        # Default implementation
        return 0.5
