"""Triage Agent – document profiling and classification.

This module implements the Stage 1 Triage Agent responsible for producing a
``DocumentProfile`` that governs downstream extraction strategy selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Protocol, Tuple
from collections import Counter

import pdfplumber

from src.models.document_profile import (
    DocumentProfile,
    DomainHint,
    EstimatedCost,
    FileMetadata,
    LayoutComplexity,
    OriginType,
)

logger = logging.getLogger(__name__)


@dataclass
class LayoutStats:
    """Aggregated layout statistics used for layout_complexity detection."""

    multi_column_pages: int = 0
    table_like_pages: int = 0
    figure_pages: int = 0
    total_pages: int = 0


class DomainClassifier(Protocol):
    """Protocol for pluggable domain classification strategies."""

    def classify(self, text: str) -> Tuple[DomainHint, float]:
        """Classify document domain from text."""


class KeywordDomainClassifier:
    """Simple keyword-based domain classifier.

    This implementation is intentionally lightweight and pluggable; a future
    VLM-based classifier can implement the same :class:`DomainClassifier`
    interface and be injected into :class:`TriageAgent`.
    """

    def __init__(self) -> None:
        # Lowercase keyword lists for each domain
        self._keywords: Dict[DomainHint, List[str]] = {
            "financial": [
                "revenue",
                "profit",
                "loss",
                "balance sheet",
                "income statement",
                "assets",
                "liabilities",
                "equity",
                "cash flow",
            ],
            "legal": [
                "hereby",
                "whereas",
                "clause",
                "section",
                "agreement",
                "party",
                "jurisdiction",
            ],
            "technical": [
                "architecture",
                "algorithm",
                "implementation",
                "framework",
                "methodology",
                "protocol",
            ],
            "medical": [
                "patient",
                "diagnosis",
                "treatment",
                "clinical",
                "medication",
                "symptom",
            ],
            "general": [],
        }

    def classify(self, text: str) -> Tuple[DomainHint, float]:
        """Classify domain based on keyword hits.

        Args:
            text: Sample text from the document (lowercased internally).

        Returns:
            Tuple of (domain_hint, confidence).
        """
        text_lc = text.lower()
        scores: Dict[DomainHint, int] = {k: 0 for k in self._keywords}

        for domain, keywords in self._keywords.items():
            for kw in keywords:
                if kw in text_lc:
                    scores[domain] += 1

        # Remove general from scoring – it is the fallback
        scores_no_general = {k: v for k, v in scores.items() if k != "general"}
        best_domain: DomainHint = "general"
        best_score = 0
        for domain, score in scores_no_general.items():
            if score > best_score:
                best_score = score
                best_domain = domain

        # Confidence is a simple function of keyword hits
        confidence = min(1.0, best_score / 5.0) if best_score > 0 else 0.2
        return best_domain, confidence


class TriageAgent:
    """Document Triage Agent for classification and profiling.

    Responsibilities:
        * Compute ``origin_type`` from character density and image ratio.
        * Estimate ``layout_complexity`` using simple column/table/figure heuristics.
        * Classify ``domain_hint`` using a pluggable classifier.
        * Estimate ``estimated_cost`` for extraction.
        * Persist :class:`DocumentProfile` to ``.refinery/profiles``.
    """

    def __init__(
        self,
        profiles_dir: Path,
        domain_classifier: DomainClassifier | None = None,
    ) -> None:
        self._profiles_dir = profiles_dir
        self._profiles_dir.mkdir(parents=True, exist_ok=True)
        self._domain_classifier = domain_classifier or KeywordDomainClassifier()
        logger.info("Initialised TriageAgent with profiles_dir=%s", profiles_dir)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def classify_document(self, pdf_path: Path) -> DocumentProfile:
        """Classify a PDF and return a :class:`DocumentProfile`.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            DocumentProfile instance.
        """
        pdf_path = pdf_path.resolve()
        if not pdf_path.exists():
            msg = f"PDF file does not exist: {pdf_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info("Classifying document: %s", pdf_path)
        doc_id = self._profile_doc_id(pdf_path)
        with pdfplumber.open(str(pdf_path)) as pdf:
            (
                origin_type,
                origin_by_page,
                layout_complexity,
                layout_by_page,
            ) = self._classify_pages(pdf)
            language, language_confidence = self._detect_language(pdf)
            domain_hint, domain_confidence = self._classify_domain(pdf)
            estimated_cost = self._estimate_extraction_cost(
                origin_type, layout_complexity
            )

            # Attach per-page classifications into metadata (extra fields are allowed)
            metadata = FileMetadata(
                path=str(pdf_path),
                size_bytes=pdf_path.stat().st_size,
                page_count=len(pdf.pages),
                mime_type="application/pdf",
                origin_by_page=origin_by_page,
                layout_by_page=layout_by_page,
            )

            profile = DocumentProfile(
                doc_id=doc_id,
                origin_type=origin_type,
                layout_complexity=layout_complexity,
                language=language,
                language_confidence=language_confidence,
                domain_hint=domain_hint,
                estimated_cost=estimated_cost,
                metadata=metadata,
            )

        # Persist profile
        profile_path = self._profiles_dir / f"{doc_id}.json"
        profile_path.write_text(profile.to_json(indent=2), encoding="utf-8")
        logger.info("Saved DocumentProfile to %s", profile_path)

        return profile

    # ------------------------------------------------------------------ #
    # Page-wise classification and aggregation
    # ------------------------------------------------------------------ #
    def _classify_pages(
        self, pdf: pdfplumber.PDF
    ) -> Tuple[OriginType, List[OriginType], LayoutComplexity, List[LayoutComplexity]]:
        """Classify each page, then aggregate to document-level labels.

        Returns:
            Tuple of:
                (document_origin_type, per_page_origin_types,
                 document_layout_complexity, per_page_layout_complexities)
        """
        origin_by_page: List[OriginType] = []
        layout_by_page: List[LayoutComplexity] = []

        for page in pdf.pages:
            origin_by_page.append(self._classify_page_origin(page))
            layout_by_page.append(self._classify_page_layout(page))

        doc_origin = self._aggregate_origin_type(origin_by_page)
        doc_layout = self._aggregate_layout_complexity(layout_by_page)
        return doc_origin, origin_by_page, doc_layout, layout_by_page

    @staticmethod
    def _classify_page_origin(page: pdfplumber.page.Page) -> OriginType:
        """Classify a single page's origin type.

        Heuristic (page-level):
            * ``scanned_image`` if chars_on_page < 50 AND image_area_ratio > 0.8
            * otherwise ``native_digital``.

        The document-level ``mixed`` label is derived when both
        ``native_digital`` and ``scanned_image`` pages are present.
        """
        text = page.extract_text() or ""
        chars_on_page = len(text)

        width, height = page.width, page.height
        page_area = float(width * height) if width and height else 0.0

        images = getattr(page, "images", []) or []
        image_area = sum(
            float(img.get("width", 0.0)) * float(img.get("height", 0.0))
            for img in images
        )
        image_ratio = image_area / page_area if page_area > 0 else 0.0

        if chars_on_page < 50 and image_ratio > 0.8:
            return "scanned_image"

        return "native_digital"

    @staticmethod
    def _aggregate_origin_type(origins: List[OriginType]) -> OriginType:
        """Aggregate per-page origin labels into a document-level label."""
        if not origins:
            return "native_digital"

        counts = Counter(origins)
        total = len(origins)

        scanned = counts.get("scanned_image", 0)
        native = counts.get("native_digital", 0)

        # Mostly scanned pages → scanned_image
        if scanned / total >= 0.8:
            return "scanned_image"

        # Mix of scanned and native pages → mixed
        if scanned > 0 and native > 0:
            return "mixed"

        # Fallbacks
        if scanned > 0:
            return "scanned_image"

        return "native_digital"

    def _classify_page_layout(
        self, page: pdfplumber.page.Page
    ) -> LayoutComplexity:
        """Classify a single page's layout complexity."""
        width = page.width or 0.0
        chars = getattr(page, "chars", []) or []
        images = getattr(page, "images", []) or []

        x_positions = [c.get("x0", 0.0) for c in chars]
        unique_columns = self._estimate_column_count(x_positions, width)
        has_table = self._looks_like_table(chars)
        has_figures = bool(images)

        if has_table:
            return "table_heavy"
        if has_figures:
            return "figure_heavy"
        if unique_columns >= 2:
            return "multi_column"
        return "single_column"

    @staticmethod
    def _aggregate_layout_complexity(
        layouts: List[LayoutComplexity],
    ) -> LayoutComplexity:
        """Aggregate per-page layout labels into a document-level label."""
        if not layouts:
            return "single_column"

        counts = Counter(layouts)

        # Prefer more complex labels when present
        for label in ("table_heavy", "figure_heavy", "multi_column", "mixed"):
            if counts.get(label, 0) > 0:
                return label  # type: ignore[return-value]

        return "single_column"

    # ------------------------------------------------------------------ #
    # Origin type detection
    # ------------------------------------------------------------------ #
    def _detect_origin_type(self, pdf: pdfplumber.PDF) -> OriginType:
        """Detect origin type using character density and image ratio.

        Heuristic:
            * ``scanned_image`` if average chars/page < 50 AND avg image ratio > 0.8
            * Otherwise ``native_digital`` or ``mixed`` based on intermediate values.
        """
        total_chars = 0
        total_pages = len(pdf.pages)
        total_image_area = 0.0
        total_page_area = 0.0
        pages_with_fonts = 0

        for page in pdf.pages:
            text = page.extract_text() or ""
            total_chars += len(text)

            width, height = page.width, page.height
            area = float(width * height) if width and height else 0.0
            total_page_area += area

            images = getattr(page, "images", []) or []
            image_area = sum(
                float(img.get("width", 0.0)) * float(img.get("height", 0.0))
                for img in images
            )
            total_image_area += image_area

            chars = getattr(page, "chars", []) or []
            if chars:
                pages_with_fonts += 1

        avg_chars_per_page = total_chars / total_pages if total_pages else 0.0
        avg_image_ratio = (
            total_image_area / total_page_area if total_page_area > 0 else 0.0
        )

        logger.debug(
            "Origin metrics: chars=%s, pages=%s, avg_chars/page=%.2f, image_ratio=%.3f, "
            "font_pages=%s",
            total_chars,
            total_pages,
            avg_chars_per_page,
            avg_image_ratio,
            pages_with_fonts,
        )

        if avg_chars_per_page < 50 and avg_image_ratio > 0.8:
            return "scanned_image"

        # Mixed if significant images or low font coverage
        font_ratio = pages_with_fonts / total_pages if total_pages else 0.0
        if avg_image_ratio > 0.3 or font_ratio < 0.5:
            return "mixed"

        return "native_digital"

    # ------------------------------------------------------------------ #
    # Layout complexity detection
    # ------------------------------------------------------------------ #
    def _detect_layout_complexity(self, pdf: pdfplumber.PDF) -> LayoutComplexity:
        """Detect layout complexity using simple spatial heuristics."""
        stats = LayoutStats(total_pages=len(pdf.pages))

        for page in pdf.pages:
            width = page.width or 0.0
            chars = getattr(page, "chars", []) or []
            images = getattr(page, "images", []) or []

            # Column detection: cluster x positions into left/right groups
            x_positions = [c.get("x0", 0.0) for c in chars]
            unique_columns = self._estimate_column_count(x_positions, width)
            if unique_columns >= 2:
                stats.multi_column_pages += 1

            # Table detection: many aligned x positions on the same y lines
            if self._looks_like_table(chars):
                stats.table_like_pages += 1

            # Figure detection: images present
            if images:
                stats.figure_pages += 1

        logger.debug("Layout stats: %s", stats)

        # Decide layout_complexity based on relative counts
        if stats.table_like_pages > max(stats.multi_column_pages, stats.figure_pages):
            return "table_heavy"
        if stats.figure_pages > max(stats.multi_column_pages, stats.table_like_pages):
            return "figure_heavy"
        if stats.multi_column_pages > 0:
            return "multi_column"
        return "single_column"

    @staticmethod
    def _estimate_column_count(x_positions: Iterable[float], page_width: float) -> int:
        """Estimate the number of text columns on a page."""
        if not x_positions or page_width <= 0:
            return 1

        # Bucket x positions into coarse bins
        bin_width = page_width / 4  # 4 buckets across the page
        bins: Dict[int, int] = {}
        for x in x_positions:
            idx = int(x // bin_width)
            bins[idx] = bins.get(idx, 0) + 1

        # Columns are bins with significant text
        threshold = max(bins.values()) * 0.2 if bins else 0
        column_bins = [b for b, count in bins.items() if count >= threshold]
        return max(1, len(column_bins))

    @staticmethod
    def _looks_like_table(chars: List[Dict[str, float]]) -> bool:
        """Heuristic to detect table-like structure on a page."""
        if not chars:
            return False

        # Group chars by approximate y (row)
        rows: Dict[int, List[Dict[str, float]]] = {}
        for ch in chars:
            y = int(ch.get("top", 0.0) // 10)  # bucket height of ~10 points
            rows.setdefault(y, []).append(ch)

        # Count rows with multiple distinct x positions
        table_like_rows = 0
        for row_chars in rows.values():
            xs = {int(c.get("x0", 0.0) // 20) for c in row_chars}  # 20-point buckets
            if len(xs) >= 3:
                table_like_rows += 1

        return table_like_rows >= 3

    # ------------------------------------------------------------------ #
    # Language & domain detection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _detect_language(pdf: pdfplumber.PDF) -> Tuple[str, float]:
        """Lightweight heuristic language detection.

        This implementation is intentionally simple and optimised for the
        current corpus (primarily English and Amharic):

        * Sample text from the first few pages.
        * Compute the proportion of Ethiopic (Amharic) Unicode codepoints.
        * If Ethiopic characters dominate, classify as ``\"am\"``.
        * Otherwise default to ``\"en\"``.

        This can be replaced with a dedicated language detection library
        (e.g. :mod:`langdetect`) in a later phase without changing the
        :class:`DocumentProfile` schema.
        """
        sample_text_parts: List[str] = []
        for page in pdf.pages[:5]:
            text = page.extract_text() or ""
            sample_text_parts.append(text)
        sample_text = "\n".join(sample_text_parts)

        if not sample_text.strip():
            # No reliable signal; fall back to English with low-ish confidence
            return "en", 0.6

        total_chars = 0
        ethiopic_chars = 0
        for ch in sample_text:
            code = ord(ch)
            # Skip obvious whitespace/control chars
            if ch.isspace() or code < 32:
                continue
            total_chars += 1
            # Ethiopic block: U+1200–U+137F
            if 0x1200 <= code <= 0x137F:
                ethiopic_chars += 1

        if total_chars == 0:
            return "en", 0.6

        ethiopic_ratio = ethiopic_chars / total_chars

        if ethiopic_ratio >= 0.3:
            # Dominantly Amharic / Ethiopic script
            confidence = 0.8 + 0.2 * min(1.0, (ethiopic_ratio - 0.3) / 0.2)
            return "am", min(1.0, confidence)

        # Default to English; confidence scales with *absence* of Ethiopic chars
        confidence = 0.7 + 0.3 * (1.0 - ethiopic_ratio)
        return "en", min(1.0, confidence)

    def _classify_domain(self, pdf: pdfplumber.PDF) -> Tuple[DomainHint, float]:
        """Classify document domain using the configured classifier."""
        # Concatenate text from first few pages as a representative sample
        sample_text_parts: List[str] = []
        for page in pdf.pages[:10]:
            text = page.extract_text() or ""
            sample_text_parts.append(text)
        sample_text = "\n".join(sample_text_parts)

        return self._domain_classifier.classify(sample_text)

    # ------------------------------------------------------------------ #
    # Cost estimation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _estimate_extraction_cost(
        origin_type: OriginType,
        layout_complexity: LayoutComplexity,
    ) -> EstimatedCost:
        """Estimate extraction cost tier from origin and layout."""
        if origin_type == "scanned_image":
            return "needs_vision_model"
        if layout_complexity in ("multi_column", "table_heavy", "figure_heavy", "mixed"):
            return "needs_layout_model"
        return "fast_text_sufficient"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _profile_doc_id(pdf_path: Path) -> str:
        """Generate a simple, stable document ID from the filename."""
        return pdf_path.stem


__all__ = ["TriageAgent", "KeywordDomainClassifier", "DomainClassifier"]

