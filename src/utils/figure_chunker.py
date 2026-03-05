"""Figure Chunker for Rule 2: Figure Caption Integrity.

This module implements specialized figure chunking logic to ensure that figure
captions are always stored as metadata of their parent figure chunk, never as
separate chunks.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from src.models.extracted_document import ExtractedDocument, Figure, TextBlock
from src.models.ldu import LDU
from src.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

# Spatial thresholds for caption detection
CAPTION_VERTICAL_THRESHOLD = 100.0  # PDF points - max distance below/above figure
CAPTION_HORIZONTAL_OVERLAP_THRESHOLD = 0.5  # 50% horizontal overlap required

# Common caption patterns
CAPTION_PATTERNS = [
    r"^Figure\s+\d+[.:]?\s+",  # "Figure 1: " or "Figure 1. "
    r"^Fig\.\s+\d+[.:]?\s+",  # "Fig. 1: "
    r"^Fig\s+\d+[.:]?\s+",  # "Fig 1: "
    r"^Image\s+\d+[.:]?\s+",  # "Image 1: "
    r"^Chart\s+\d+[.:]?\s+",  # "Chart 1: "
    r"^Graph\s+\d+[.:]?\s+",  # "Graph 1: "
]


class FigureChunker:
    """Specialized chunker for figure structures.

    Ensures that figure captions are always paired with their figures and stored
    as metadata, never as separate chunks.
    """

    def __init__(
        self,
        vertical_threshold: float = CAPTION_VERTICAL_THRESHOLD,
        horizontal_overlap_threshold: float = CAPTION_HORIZONTAL_OVERLAP_THRESHOLD,
    ):
        """Initialize the FigureChunker.

        Args:
            vertical_threshold: Maximum vertical distance (PDF points) between
                figure and caption for pairing.
            horizontal_overlap_threshold: Minimum horizontal overlap ratio (0-1)
                required for caption to be considered associated with figure.
        """
        self.vertical_threshold = vertical_threshold
        self.horizontal_overlap_threshold = horizontal_overlap_threshold

    def find_caption_for_figure(
        self, figure: Figure, extracted_document: ExtractedDocument
    ) -> Optional[TextBlock]:
        """Find the caption text block associated with a figure.

        Searches for captions using:
        1. Spatial proximity (captions are usually below/above figures)
        2. Reading order clues (captions follow figures in reading order)
        3. Pattern matching (common caption patterns like "Figure 1:")

        Args:
            figure: The figure to find a caption for.
            extracted_document: The extracted document containing text blocks.

        Returns:
            The TextBlock that is the caption, or None if no caption found.
        """
        # First, check if figure already has a caption
        if figure.caption and figure.caption.strip():
            # Try to find the text block that matches this caption
            for text_block in extracted_document.text_blocks:
                if (
                    text_block.page_num == figure.page_num
                    and self._is_caption_text(text_block.content)
                    and self._is_spatially_proximate(figure, text_block)
                ):
                    # Verify it matches the figure's caption
                    if figure.caption.strip().lower() in text_block.content.lower():
                        return text_block
            # If we can't find the text block, that's okay - use the caption from figure
            return None

        # Search for captions near the figure
        candidates = []

        for text_block in extracted_document.text_blocks:
            # Must be on same page or adjacent page
            if abs(text_block.page_num - figure.page_num) > 1:
                continue

            # Check if text looks like a caption
            if not self._is_caption_text(text_block.content):
                continue

            # Check spatial proximity
            if not self._is_spatially_proximate(figure, text_block):
                continue

            # Calculate proximity score
            score = self._calculate_proximity_score(figure, text_block)
            candidates.append((score, text_block))

        if not candidates:
            return None

        # Return the closest candidate
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_candidate = candidates[0][1]

        logger.debug(
            f"Found caption for figure on page {figure.page_num}: "
            f"{best_candidate.content[:50]}..."
        )

        return best_candidate

    def pair_figure_with_caption(
        self, figure: Figure, caption: Optional[TextBlock] = None
    ) -> LDU:
        """Create an LDU for a figure with its caption stored as metadata.

        Args:
            figure: The figure object.
            caption: Optional TextBlock that is the caption. If None, uses
                figure.caption if available.

        Returns:
            An LDU with the figure content and caption in metadata.
        """
        # Determine caption text
        if caption:
            caption_text = caption.content.strip()
        elif figure.caption:
            caption_text = figure.caption.strip()
        else:
            caption_text = ""

        # Figure content is minimal - just the caption or placeholder
        content = caption_text if caption_text else "[Figure]"

        # Create metadata with caption information
        metadata = {
            "caption": caption_text,
            "has_caption": bool(caption_text),
            "caption_source": "text_block" if caption else ("figure_object" if figure.caption else "none"),
        }

        # If we found a caption from text blocks, add spatial info
        if caption:
            metadata["caption_bbox"] = {
                "x0": caption.bbox.x0,
                "y0": caption.bbox.y0,
                "x1": caption.bbox.x1,
                "y1": caption.bbox.y1,
            }
            metadata["caption_page"] = caption.page_num

        # Determine page refs (include caption page if different)
        page_refs = [figure.page_num]
        if caption and caption.page_num != figure.page_num:
            page_refs.append(caption.page_num)
            page_refs.sort()

        # Calculate bounding box that includes both figure and caption
        bbox = {
            "x0": figure.bbox.x0,
            "y0": figure.bbox.y0,
            "x1": figure.bbox.x1,
            "y1": figure.bbox.y1,
        }

        if caption:
            bbox["x0"] = min(bbox["x0"], caption.bbox.x0)
            bbox["y0"] = min(bbox["y0"], caption.bbox.y0)
            bbox["x1"] = max(bbox["x1"], caption.bbox.x1)
            bbox["y1"] = max(bbox["y1"], caption.bbox.y1)

        ldu = LDU(
            content=content,
            chunk_type="figure",
            page_refs=page_refs,
            bounding_box=bbox,
            token_count=count_tokens(content),
            metadata=metadata,
        )

        return ldu

    def _is_caption_text(self, text: str) -> bool:
        """Check if text looks like a figure caption.

        Args:
            text: Text to check.

        Returns:
            True if text matches common caption patterns.
        """
        text_stripped = text.strip()
        if not text_stripped:
            return False

        # Check against common caption patterns
        for pattern in CAPTION_PATTERNS:
            if re.match(pattern, text_stripped, re.IGNORECASE):
                return True

        # Check for short text (captions are usually short)
        # and common caption keywords
        if len(text_stripped) < 200:
            text_lower = text_stripped.lower()
            caption_keywords = [
                "figure",
                "fig.",
                "image",
                "chart",
                "graph",
                "diagram",
                "illustration",
            ]
            if any(keyword in text_lower for keyword in caption_keywords):
                # Check if it starts with a number (common in captions)
                if re.match(r"^\d+", text_stripped):
                    return True

        return False

    def _is_spatially_proximate(
        self, figure: Figure, text_block: TextBlock
    ) -> bool:
        """Check if a text block is spatially proximate to a figure.

        Args:
            figure: The figure.
            text_block: The text block to check.

        Returns:
            True if text block is close enough to be considered a caption.
        """
        # Must be on same page or adjacent page
        if abs(text_block.page_num - figure.page_num) > 1:
            return False

        # Calculate vertical distance
        # Captions are usually below figures (higher y0 in PDF coordinates)
        # or above figures (lower y0)
        figure_bottom = figure.bbox.y0
        figure_top = figure.bbox.y1
        text_bottom = text_block.bbox.y0
        text_top = text_block.bbox.y1

        # Vertical distance: caption below figure
        distance_below = abs(text_bottom - figure_top)
        # Vertical distance: caption above figure
        distance_above = abs(figure_bottom - text_top)

        min_vertical_distance = min(distance_below, distance_above)

        if min_vertical_distance > self.vertical_threshold:
            return False

        # Check horizontal overlap
        horizontal_overlap = self._calculate_horizontal_overlap(figure, text_block)

        return horizontal_overlap >= self.horizontal_overlap_threshold

    def _calculate_horizontal_overlap(
        self, figure: Figure, text_block: TextBlock
    ) -> float:
        """Calculate horizontal overlap ratio between figure and text block.

        Args:
            figure: The figure.
            text_block: The text block.

        Returns:
            Overlap ratio from 0.0 to 1.0.
        """
        fig_left = figure.bbox.x0
        fig_right = figure.bbox.x1
        text_left = text_block.bbox.x0
        text_right = text_block.bbox.x1

        # Calculate overlap
        overlap_left = max(fig_left, text_left)
        overlap_right = min(fig_right, text_right)

        if overlap_left >= overlap_right:
            return 0.0

        overlap_width = overlap_right - overlap_left
        text_width = text_right - text_left

        if text_width == 0:
            return 0.0

        return overlap_width / text_width

    def _calculate_proximity_score(
        self, figure: Figure, text_block: TextBlock
    ) -> float:
        """Calculate a proximity score for caption matching.

        Higher score = more likely to be the caption.

        Args:
            figure: The figure.
            text_block: The text block candidate.

        Returns:
            Proximity score (higher is better).
        """
        score = 0.0

        # Same page gets high score
        if text_block.page_num == figure.page_num:
            score += 10.0
        else:
            score += 5.0  # Adjacent page

        # Vertical proximity (closer = better)
        figure_top = figure.bbox.y1
        text_bottom = text_block.bbox.y0
        vertical_distance = abs(text_bottom - figure_top)

        if vertical_distance < self.vertical_threshold:
            # Normalize to 0-1 and add to score
            proximity_factor = 1.0 - (vertical_distance / self.vertical_threshold)
            score += proximity_factor * 5.0

        # Horizontal overlap (more overlap = better)
        horizontal_overlap = self._calculate_horizontal_overlap(figure, text_block)
        score += horizontal_overlap * 3.0

        # Pattern matching bonus
        if self._is_caption_text(text_block.content):
            score += 2.0

        return score


def identify_figure_captions(
    extracted_document: ExtractedDocument,
) -> List[Tuple[Figure, Optional[TextBlock]]]:
    """Identify all figures and their associated captions.

    Args:
        extracted_document: The ExtractedDocument to analyze.

    Returns:
        List of tuples (figure, caption_text_block) for each figure.
    """
    chunker = FigureChunker()
    pairs = []
    for figure in extracted_document.figures:
        caption = chunker.find_caption_for_figure(figure, extracted_document)
        pairs.append((figure, caption))
    return pairs
