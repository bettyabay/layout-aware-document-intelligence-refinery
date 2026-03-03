"""Layout-aware extraction strategy using Docling (Strategy B).

This strategy is intended for documents with more complex layouts – multi-column
text, table-heavy reports, and documents with significant figures. It uses
Docling to recover structure and then normalises the result into the internal
``ExtractedDocument`` schema via :class:`DoclingAdapter`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pdfplumber

from src.models.document_profile import DocumentProfile
from src.models.extracted_document import ExtractedDocument
from src.strategies.base import ExtractionStrategy
from src.utils.adapters import DoclingAdapter
from src.utils.confidence_scorer import combined_weighted_score

try:
    # Docling is an optional dependency; import lazily.
    from docling.document_converter import DocumentConverter  # type: ignore[import]
except Exception as exc:  # pragma: no cover - runtime fallback
    DocumentConverter = None  # type: ignore[assignment]
    _DOC_CONVERTER_IMPORT_ERROR = exc
else:
    _DOC_CONVERTER_IMPORT_ERROR = None


class LayoutExtractor(ExtractionStrategy):
    """Layout-aware extraction using Docling."""

    def __init__(self) -> None:
        super().__init__(name="layout_aware")
        if DocumentConverter is None:
            raise ImportError(
                "docling is not installed or could not be imported. "
                "Install it with `pip install docling` to use LayoutExtractor."
            ) from _DOC_CONVERTER_IMPORT_ERROR
        self._converter = DocumentConverter()

    # ------------------------------------------------------------------ #
    # Core extraction
    # ------------------------------------------------------------------ #
    def extract(self, document_path: str) -> ExtractedDocument:
        """Extract structured content using Docling.

        Args:
            document_path: Path to the PDF/document file.

        Returns:
            ``ExtractedDocument`` with text blocks, tables, figures, and reading order.
        """
        pdf_path = Path(document_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"Document not found: {pdf_path}")

        try:
            # Docling performs layout-aware parsing and returns a DoclingDocument
            result = self._converter.convert(str(pdf_path))
        except Exception as exc:  # pragma: no cover - external library failure
            raise RuntimeError(f"Docling conversion failed for {pdf_path}") from exc

        docling_document = getattr(result, "document", result)
        return DoclingAdapter.to_extracted_document(docling_document)

    # ------------------------------------------------------------------ #
    # Confidence scoring
    # ------------------------------------------------------------------ #
    def confidence_score(self, document_path: str) -> float:
        """Compute confidence score for layout-aware extraction.

        We reuse the generic ``combined_weighted_score`` but bias it towards
        layout and table quality, since this strategy is chosen primarily for
        structural fidelity rather than raw throughput.
        """
        pdf_path = Path(document_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"Document not found: {pdf_path}")

        weights = {
            "character_density": 0.3,
            "layout_preservation": 0.4,
            "table_extraction": 0.3,
        }
        return combined_weighted_score(str(pdf_path), weights=weights)

    # ------------------------------------------------------------------ #
    # Cost estimation
    # ------------------------------------------------------------------ #
    def cost_estimate(self, document_path: str) -> Dict[str, float]:
        """Estimate cost for layout-aware extraction.

        Docling is CPU-bound but heavier than ``FastTextExtractor``. We model a
        small but non-zero notional cost per page so the router can reason about
        trade-offs, even though local execution does not incur token charges.
        """
        pdf_path = Path(document_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"Document not found: {pdf_path}")

        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = len(pdf.pages)

        # Nominal cost model: slightly higher than fast text but still cheap.
        cost_per_page = 0.0005
        total_cost = cost_per_page * pages
        return {
            "total_cost_usd": total_cost,
            "cost_per_page": cost_per_page if pages > 0 else 0.0,
        }

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #
    def can_handle(self, profile: DocumentProfile) -> bool:
        """Return True when the profile suggests a layout model is appropriate.

        Triggers when:
            * Layout is multi-column, table-heavy, figure-heavy, or mixed.
            * Origin is native_digital or mixed (pure scanned documents should
              be handled by the vision-augmented strategy).
        """
        if profile.origin_type not in ("native_digital", "mixed"):
            return False

        return profile.layout_complexity in (
            "multi_column",
            "table_heavy",
            "figure_heavy",
            "mixed",
        )


__all__ = ["LayoutExtractor"]

