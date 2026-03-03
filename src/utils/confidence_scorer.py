"""Confidence scoring utilities for extraction strategies.

This module provides functions to calculate confidence scores for document
extraction. Confidence scores are used by the extraction router to select
the best strategy and to determine when to escalate to more sophisticated
(and expensive) strategies.

All scoring functions return values between 0.0 and 1.0, where:
- 0.0 = Very low confidence (extraction likely to fail)
- 1.0 = Very high confidence (extraction should succeed)
"""

from __future__ import annotations

from typing import Dict, List

import pdfplumber


def character_density_score(
    pdf_path: str, min_chars_per_page: int = 100, min_density: float = 0.01
) -> float:
    """Calculate confidence score based on character density.

    Higher character density indicates a native digital PDF with extractable
    text. Low density suggests a scanned document that may need OCR.

    Args:
        pdf_path: Path to the PDF file.
        min_chars_per_page: Minimum characters per page for high confidence.
        min_density: Minimum character density (chars/point²) for high confidence.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_chars = 0
            total_area = 0
            page_count = len(pdf.pages)

            for page in pdf.pages:
                text = page.extract_text() or ""
                char_count = len(text)
                total_chars += char_count

                width = page.width
                height = page.height
                area = width * height
                total_area += area

            if page_count == 0 or total_area == 0:
                return 0.0

            avg_chars_per_page = total_chars / page_count
            avg_density = total_chars / total_area

            # Score based on both metrics
            chars_score = min(1.0, avg_chars_per_page / min_chars_per_page)
            density_score = min(1.0, avg_density / min_density)

            # Combined score (weighted average)
            score = 0.6 * chars_score + 0.4 * density_score
            return min(1.0, max(0.0, score))

    except Exception:
        return 0.0


def layout_preservation_score(
    pdf_path: str, sample_pages: int = 5
) -> float:
    """Calculate confidence score based on layout preservation capability.

    This score estimates how well the extraction strategy can preserve document
    layout (columns, reading order, spatial relationships). It analyzes text
    block positions to detect multi-column layouts and spatial structure.

    Args:
        pdf_path: Path to the PDF file.
        sample_pages: Number of pages to sample for analysis.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            if page_count == 0:
                return 0.0

            # Sample pages (first, middle, last)
            sample_indices = set()
            if page_count <= sample_pages:
                sample_indices = set(range(page_count))
            else:
                sample_indices.add(0)
                sample_indices.add(page_count - 1)
                # Add evenly spaced middle pages
                step = (page_count - 2) // (sample_pages - 2)
                for i in range(1, page_count - 1, step):
                    sample_indices.add(i)
                    if len(sample_indices) >= sample_pages:
                        break

            multi_column_pages = 0
            structured_pages = 0

            for idx in sample_indices:
                page = pdf.pages[idx]
                chars = page.chars if hasattr(page, "chars") else []

                if len(chars) < 10:
                    continue

                # Detect multi-column layout by analyzing x-coordinates
                x_positions = [char.get("x0", 0) for char in chars]
                if len(x_positions) > 0:
                    x_min = min(x_positions)
                    x_max = max(x_positions)
                    x_range = x_max - x_min

                    # Bin x-positions to detect columns
                    if x_range > 0:
                        bins: Dict[int, int] = {}
                        num_bins = max(2, int(page.width / 200))  # ~200pt bins

                        for x in x_positions:
                            bin_idx = int((x - x_min) / x_range * num_bins)
                            bins[bin_idx] = bins.get(bin_idx, 0) + 1

                        # If we have 2+ significant bins, likely multi-column
                        significant_bins = sum(1 for count in bins.values() if count > len(chars) * 0.1)
                        if significant_bins >= 2:
                            multi_column_pages += 1

                # Detect structured content (tables, lists) by alignment patterns
                y_positions = [char.get("top", 0) for char in chars]
                if len(y_positions) > 10:
                    # Count distinct y-positions (rows)
                    unique_y = len(set(round(y, 1) for y in y_positions))
                    if unique_y > 5:  # Multiple rows suggest structure
                        structured_pages += 1

            if len(sample_indices) == 0:
                return 0.0

            # Score: higher if we detect layout complexity
            # For simple layouts, fast text extraction is sufficient (high score)
            # For complex layouts, we need layout-aware extraction (medium score)
            multi_column_ratio = multi_column_pages / len(sample_indices)
            structured_ratio = structured_pages / len(sample_indices)

            # If no complex layout detected, high confidence for simple extraction
            if multi_column_ratio < 0.2 and structured_ratio < 0.3:
                return 0.9

            # If complex layout detected, medium confidence (needs layout-aware)
            if multi_column_ratio > 0.5 or structured_ratio > 0.5:
                return 0.6

            # Mixed case
            return 0.75

    except Exception:
        return 0.5  # Default to medium confidence on error


def table_extraction_score(
    pdf_path: str, sample_pages: int = 10
) -> float:
    """Calculate confidence score for table extraction capability.

    This score estimates how well the extraction strategy can extract tables
    as structured data (with headers and rows) rather than plain text.

    Args:
        pdf_path: Path to the PDF file.
        sample_pages: Number of pages to sample for table detection.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            if page_count == 0:
                return 0.0

            # Sample pages
            sample_indices = set()
            if page_count <= sample_pages:
                sample_indices = set(range(page_count))
            else:
                step = page_count // sample_pages
                for i in range(0, page_count, step):
                    sample_indices.add(i)
                    if len(sample_indices) >= sample_pages:
                        break

            table_like_pages = 0

            for idx in sample_indices:
                page = pdf.pages[idx]
                chars = page.chars if hasattr(page, "chars") else []

                if len(chars) < 20:
                    continue

                # Detect table-like structure by analyzing alignment
                # Tables have characters aligned in rows and columns
                x_positions = [char.get("x0", 0) for char in chars]
                y_positions = [char.get("top", 0) for char in chars]

                if len(x_positions) < 20 or len(y_positions) < 20:
                    continue

                # Group characters by y-position (rows)
                rows: Dict[float, List[float]] = {}
                for char, y_pos in zip(chars, y_positions):
                    y_rounded = round(y_pos, 1)
                    if y_rounded not in rows:
                        rows[y_rounded] = []
                    rows[y_rounded].append(char.get("x0", 0))

                # Check if we have multiple rows with multiple distinct x-positions
                rows_with_multiple_x = 0
                for row_x_positions in rows.values():
                    unique_x = len(set(round(x, 1) for x in row_x_positions))
                    if unique_x >= 3:  # At least 3 columns
                        rows_with_multiple_x += 1

                # If we have 3+ rows with 3+ columns, likely a table
                if rows_with_multiple_x >= 3:
                    table_like_pages += 1

            if len(sample_indices) == 0:
                return 0.0

            table_ratio = table_like_pages / len(sample_indices)

            # Score: high if no tables (simple extraction sufficient)
            # Medium if tables detected (needs table extraction capability)
            if table_ratio < 0.1:
                return 0.9  # No tables, simple extraction OK
            elif table_ratio > 0.5:
                return 0.5  # Many tables, needs specialized extraction
            else:
                return 0.7  # Some tables, moderate confidence

    except Exception:
        return 0.5  # Default to medium confidence on error


def combined_weighted_score(
    pdf_path: str,
    weights: Dict[str, float] | None = None,
    **kwargs,
) -> float:
    """Calculate combined confidence score using weighted components.

    This function combines multiple confidence scores into a single weighted
    score. The default weights are tuned for fast text extraction strategies,
    but can be customized for different strategy types.

    Args:
        pdf_path: Path to the PDF file.
        weights: Optional dictionary with weights for each component.
            Keys: 'character_density', 'layout_preservation', 'table_extraction'
            Default: {'character_density': 0.4, 'layout_preservation': 0.3, 'table_extraction': 0.3}
        **kwargs: Additional arguments passed to individual scoring functions.

    Returns:
        Combined confidence score between 0.0 and 1.0.
    """
    if weights is None:
        weights = {
            "character_density": 0.4,
            "layout_preservation": 0.3,
            "table_extraction": 0.3,
        }

    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    # Calculate individual scores
    scores: Dict[str, float] = {}

    if "character_density" in normalized_weights:
        scores["character_density"] = character_density_score(
            pdf_path, **kwargs.get("character_density", {})
        )

    if "layout_preservation" in normalized_weights:
        scores["layout_preservation"] = layout_preservation_score(
            pdf_path, **kwargs.get("layout_preservation", {})
        )

    if "table_extraction" in normalized_weights:
        scores["table_extraction"] = table_extraction_score(
            pdf_path, **kwargs.get("table_extraction", {})
        )

    # Calculate weighted average
    combined_score = sum(
        normalized_weights.get(key, 0.0) * score
        for key, score in scores.items()
    )

    return min(1.0, max(0.0, combined_score))


__all__ = [
    "character_density_score",
    "layout_preservation_score",
    "table_extraction_score",
    "combined_weighted_score",
]
