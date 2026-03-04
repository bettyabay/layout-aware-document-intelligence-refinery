"""Layout-aware extraction strategy using MinerU (Strategy B alternative).

This strategy mirrors :class:`LayoutExtractor` (Docling-based) but delegates the
heavy lifting to MinerU's PDF-Extract-Kit pipeline. Because MinerU's concrete
CLI / Python APIs and output schemas can change between versions and are often
heavily configurable, this implementation is intentionally conservative:

* It does **not** hard-code MinerU's CLI arguments or Python API calls.
* Instead, it assumes that you will run MinerU with an output format that can
  be normalised into the schema documented in :class:`MinerUAdapter`.
* The path to that JSON is passed into this extractor, or looked up from a
  simple convention.

In practice, you should:

1. Install MinerU (and its optional extras), for example::

       uv pip install -U \"mineru[all]\"

2. Configure your MinerU invocation (CLI or API) to write a JSON file per
   document that follows the normalised schema in ``MinerUAdapter``.
3. Point this extractor at that JSON (via `mineru_json_path` or a naming
   convention), so it can be converted into ``ExtractedDocument``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import json
import pdfplumber

from src.models.document_profile import DocumentProfile
from src.models.extracted_document import ExtractedDocument
from src.strategies.base import ExtractionStrategy
from src.utils.adapters import MinerUAdapter
from src.utils.confidence_scorer import combined_weighted_score


class MinerUExtractor(ExtractionStrategy):
    """Layout-aware extraction using MinerU output plus :class:`MinerUAdapter`.

    This class does **not** invoke MinerU directly, as doing so would require
    hard-coding external CLI/API contracts that may diverge from the version
    you have installed. Instead, it focuses solely on:

    * Validating the input document path.
    * Locating and loading a MinerU-produced JSON payload.
    * Converting that payload into the internal :class:`ExtractedDocument`.
    * Providing confidence and cost estimates compatible with other strategies.
    """

    def __init__(self, mineru_json_path: Optional[Path] = None) -> None:
        super().__init__(name="layout_mineru")
        self._mineru_json_path = mineru_json_path

    # ------------------------------------------------------------------ #
    # Core extraction
    # ------------------------------------------------------------------ #
    def extract(self, document_path: str) -> ExtractedDocument:
        """Convert MinerU JSON output for ``document_path`` into ``ExtractedDocument``.

        Args:
            document_path: Path to the original PDF that was processed by MinerU.

        Returns:
            Normalised :class:`ExtractedDocument`.

        Notes:
            This method assumes that MinerU has already been run for the given
            document and that a corresponding JSON file exists. By default the
            extractor looks for a JSON file:

            * At ``self._mineru_json_path`` if provided, or
            * Next to the PDF, named ``<stem>.mineru.json``.

            If neither exists, a clear error is raised with guidance on how to
            integrate MinerU into your pipeline.
        """
        pdf_path = Path(document_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"Document not found: {pdf_path}")

        json_path = self._resolve_mineru_json_path(pdf_path)
        if not json_path.exists():
            raise FileNotFoundError(
                "MinerU JSON output not found.\n"
                f"Expected at: {json_path}\n\n"
                "Run MinerU on the PDF first (CLI or API) and emit JSON matching "
                "the normalised schema documented in `MinerUAdapter`, or pass an "
                "explicit `mineru_json_path` when constructing `MinerUExtractor`."
            )

        with json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        # Delegate structural normalisation to the adapter.
        return MinerUAdapter.from_mineru_json(raw)

    def _resolve_mineru_json_path(self, pdf_path: Path) -> Path:
        """Resolve the expected MinerU JSON path for a given PDF."""
        if self._mineru_json_path is not None:
            return self._mineru_json_path.resolve()
        # Default convention: alongside the PDF, with a `.mineru.json` suffix.
        return pdf_path.with_suffix(pdf_path.suffix + ".mineru.json")

    # ------------------------------------------------------------------ #
    # Confidence scoring
    # ------------------------------------------------------------------ #
    def confidence_score(self, document_path: str) -> float:
        """Compute confidence score for MinerU-based layout-aware extraction.

        We reuse ``combined_weighted_score`` with a bias similar to the Docling
        extractor, but with slightly higher emphasis on table extraction and
        formulas—the areas where MinerU is particularly strong.
        """
        pdf_path = Path(document_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"Document not found: {pdf_path}")

        weights = {
            "character_density": 0.25,
            "layout_preservation": 0.35,
            "table_extraction": 0.40,
        }
        return combined_weighted_score(str(pdf_path), weights=weights)

    # ------------------------------------------------------------------ #
    # Cost estimation
    # ------------------------------------------------------------------ #
    def cost_estimate(self, document_path: str) -> Dict[str, float]:
        """Estimate cost for MinerU-based layout-aware extraction.

        MinerU is typically more expensive than simple text extraction, but
        comparable in cost to other layout-aware strategies. We model a small
        per-page notional cost to allow the router to compare against Docling.
        """
        pdf_path = Path(document_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"Document not found: {pdf_path}")

        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = len(pdf.pages)

        # Nominal cost model: slightly higher than Docling to capture OCR overhead.
        cost_per_page = 0.0007
        total_cost = cost_per_page * pages
        return {
            "total_cost_usd": total_cost,
            "cost_per_page": cost_per_page if pages > 0 else 0.0,
        }

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #
    def can_handle(self, profile: DocumentProfile) -> bool:
        """Return True when the profile suggests MinerU is a good fit.

        MinerU is particularly strong on:

        * Table-heavy financial/technical documents.
        * Documents with significant formulas (converted to LaTeX).
        * Multi-language or OCR-heavy content (scanned PDFs).

        We therefore allow it to handle:

        * Any document with ``table_heavy`` or ``mixed`` layout.
        * Any document whose origin is ``mixed`` or ``scanned_image``.
        """
        if profile.layout_complexity in ("table_heavy", "mixed"):
            return True

        if profile.origin_type in ("mixed", "scanned_image"):
            return True

        return False


__all__ = ["MinerUExtractor"]

