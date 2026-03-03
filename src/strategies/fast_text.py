"""Fast text extraction strategy using pdfplumber.

This strategy is intended for native digital, relatively simple-layout PDFs.
It uses pdfplumber to extract text blocks, tables, and basic image metadata
and returns an :class:`ExtractedDocument`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pdfplumber

from src.models.document_profile import DocumentProfile
from src.models.extracted_document import (
    BoundingBox,
    ExtractedDocument,
    Figure,
    Table,
    TextBlock,
)
from src.strategies.base import ExtractionStrategy
from src.utils.confidence_scorer import (
    character_density_score,
    combined_weighted_score,
)


class FastTextExtractor(ExtractionStrategy):
    """Fast text extraction using pdfplumber."""

    def __init__(self) -> None:
        super().__init__(name="fast_text")

    # ------------------------------------------------------------------ #
    # Core extraction
    # ------------------------------------------------------------------ #
    def extract(self, document_path: str) -> ExtractedDocument:
        """Extract text blocks, tables, and figures from a PDF using pdfplumber."""
        pdf_path = Path(document_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"Document not found: {pdf_path}")

        text_blocks: List[TextBlock] = []
        tables: List[Table] = []
        figures: List[Figure] = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                # Text blocks – use words as minimal blocks in reading order
                words = page.extract_words(
                    use_text_flow=True, keep_blank_chars=False
                ) or []
                for w in words:
                    bbox = BoundingBox(
                        x0=float(w.get("x0", 0.0)),
                        y0=float(w.get("bottom", 0.0)),
                        x1=float(w.get("x1", 0.0)),
                        y1=float(w.get("top", 0.0)),
                    )
                    text_blocks.append(
                        TextBlock(
                            content=w.get("text", ""),
                            bbox=bbox,
                            page_num=page_index,
                        )
                    )

                # Tables – use pdfplumber's table finding
                try:
                    found_tables = page.find_tables() or []
                except Exception:
                    found_tables = []

                for t in found_tables:
                    try:
                        raw_rows = t.extract() or []
                    except Exception:
                        raw_rows = []

                    if not raw_rows:
                        continue

                    headers = [str(c) for c in raw_rows[0]]
                    rows = [[str(c) for c in r] for r in raw_rows[1:]]
                    x0, top, x1, bottom = t.bbox
                    bbox = BoundingBox(
                        x0=float(x0),
                        y0=float(bottom),
                        x1=float(x1),
                        y1=float(top),
                    )
                    tables.append(
                        Table(
                            headers=headers,
                            rows=rows,
                            bbox=bbox,
                            page_num=page_index,
                        )
                    )

                # Figures – capture image regions as figures with empty captions
                images = getattr(page, "images", []) or []
                for img in images:
                    x0 = float(img.get("x0", 0.0))
                    x1 = float(img.get("x1", x0))
                    top = float(img.get("top", 0.0))
                    bottom = float(img.get("bottom", top))
                    bbox = BoundingBox(x0=x0, y0=bottom, x1=x1, y1=top)
                    figures.append(
                        Figure(
                            caption="",
                            bbox=bbox,
                            page_num=page_index,
                        )
                    )

        # Reading order: indices in order of addition
        reading_order = list(range(len(text_blocks)))

        return ExtractedDocument(
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            reading_order=reading_order,
        )

    # ------------------------------------------------------------------ #
    # Confidence scoring
    # ------------------------------------------------------------------ #
    def confidence_score(self, document_path: str) -> float:
        """Compute combined confidence score for fast text extraction.

        Components:
            * Character density (via ``character_density_score``)
            * Layout preservation (simple heuristic favouring simple layouts)
            * Table extraction (fraction of tables parsed successfully)
        """
        pdf_path = Path(document_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"Document not found: {pdf_path}")

        # Use generic combined scorer, but adjust weights for fast text
        weights = {
            "character_density": 0.5,
            "layout_preservation": 0.3,
            "table_extraction": 0.2,
        }
        return combined_weighted_score(str(pdf_path), weights=weights)

    def cost_estimate(self, document_path: str) -> Dict[str, float]:
        """Estimate cost for fast text extraction (CPU-only, effectively free)."""
        pdf_path = Path(document_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"Document not found: {pdf_path}")

        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = len(pdf.pages)

        # Assume negligible marginal cost; keep numbers explicit for logging
        total_cost = 0.0
        return {
            "total_cost_usd": total_cost,
            "cost_per_page": total_cost / pages if pages > 0 else 0.0,
        }

    def can_handle(self, profile: DocumentProfile) -> bool:
        """Return True when the profile suggests fast text is sufficient."""
        return (
            profile.origin_type == "native_digital"
            and profile.layout_complexity == "single_column"
        )


__all__ = ["FastTextExtractor"]

