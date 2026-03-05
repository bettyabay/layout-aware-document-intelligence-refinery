"""Content Hash Generator for Provenance Tracking.

This module provides utilities for generating deterministic, spatial, content-aware
hashes for document chunks. These hashes are used for provenance tracking and
verification, enabling the system to verify that extracted content still exists
in the source document even after re-extraction or document updates.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, List, Optional, Tuple

from src.models.extracted_document import ExtractedDocument
from src.models.ldu import LDU

logger = logging.getLogger(__name__)

# Default content preview length for hashing
DEFAULT_CONTENT_PREVIEW_LENGTH = 100

# Spatial tolerance for verification (in PDF points)
DEFAULT_SPATIAL_TOLERANCE = 10.0

# Content similarity threshold for fuzzy matching
DEFAULT_CONTENT_SIMILARITY_THRESHOLD = 0.8


class ContentHasher:
    """Content hash generator and verifier for provenance tracking.

    This class provides methods to:
    - Generate deterministic hashes from content and spatial information
    - Verify that a chunk's content still exists in a document
    - Handle slight variations in position and content with fuzzy matching
    """

    def __init__(
        self,
        content_preview_length: int = DEFAULT_CONTENT_PREVIEW_LENGTH,
        spatial_tolerance: float = DEFAULT_SPATIAL_TOLERANCE,
        content_similarity_threshold: float = DEFAULT_CONTENT_SIMILARITY_THRESHOLD,
    ):
        """Initialize the ContentHasher.

        Args:
            content_preview_length: Number of characters to include in content
                preview for hashing. Longer previews are more content-aware but
                less robust to minor edits.
            spatial_tolerance: Maximum distance (in PDF points) for considering
                two bounding boxes as matching during verification.
            content_similarity_threshold: Minimum similarity score (0-1) for
                considering content as matching during verification.
        """
        self.content_preview_length = content_preview_length
        self.spatial_tolerance = spatial_tolerance
        self.content_similarity_threshold = content_similarity_threshold

    def generate_content_hash(
        self, content: str, page_num: int, bbox: Dict[str, float]
    ) -> str:
        """Generate a deterministic content hash from content, page, and bbox.

        The hash is:
        - Deterministic: same input always produces same hash
        - Spatial: includes position information (page and bbox)
        - Content-aware: includes text preview
        - Collision-resistant: uses SHA-256

        Args:
            content: The text content to hash.
            page_num: 1-indexed page number.
            bbox: Bounding box dictionary with keys 'x0', 'y0', 'x1', 'y1'.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        # Normalize content preview (first N characters, stripped)
        content_preview = content[: self.content_preview_length].strip()

        # Normalize bounding box (ensure all values are floats)
        normalized_bbox = {
            "x0": float(bbox.get("x0", 0.0)),
            "y0": float(bbox.get("y0", 0.0)),
            "x1": float(bbox.get("x1", 0.0)),
            "y1": float(bbox.get("y1", 0.0)),
        }

        # Create deterministic payload
        payload = {
            "content_preview": content_preview,
            "page_num": int(page_num),
            "bbox": normalized_bbox,
        }

        # Serialize to JSON with sorted keys for determinism
        data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")

        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(data)
        return hash_obj.hexdigest()

    def generate_spatial_hash(
        self, text: str, coordinates: Dict[str, float]
    ) -> str:
        """Generate a spatial hash from text and coordinates.

        This is a lighter-weight hash that focuses primarily on spatial
        information, useful for position-based lookups.

        Args:
            text: Text content (may be truncated for hashing).
            coordinates: Coordinate dictionary. Expected keys include 'x0', 'y0',
                'x1', 'y1', and optionally 'page'.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        # Normalize text (first N characters)
        text_preview = text[: self.content_preview_length].strip()

        # Normalize coordinates
        normalized_coords = {k: float(v) for k, v in coordinates.items()}

        payload = {
            "text_preview": text_preview,
            "coordinates": normalized_coords,
        }

        data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        hash_obj = hashlib.sha256(data)
        return hash_obj.hexdigest()

    def verify_hash(
        self, chunk: LDU, document: ExtractedDocument
    ) -> Tuple[bool, float]:
        """Verify that a chunk's content still exists in the document.

        This method searches the document for content matching the chunk, with
        tolerance for slight variations in position and content. It returns both
        a boolean (found/not found) and a confidence score (0.0 to 1.0).

        Args:
            chunk: The LDU chunk to verify.
            document: The ExtractedDocument to search in.

        Returns:
            Tuple of (found: bool, confidence: float).
            - found: True if matching content is found, False otherwise.
            - confidence: Confidence score from 0.0 to 1.0, based on:
              * Content similarity (exact match = 1.0, fuzzy match = 0.8-0.99)
              * Position proximity (exact position = 1.0, nearby = 0.7-0.99)
        """
        # Search for matching content in text blocks
        best_match: Optional[Tuple[float, Dict]] = None

        for text_block in document.text_blocks:
            # Check if page matches (within tolerance for multi-page chunks)
            if text_block.page_num not in chunk.page_refs:
                continue

            # Calculate content similarity
            content_sim = self._calculate_content_similarity(
                chunk.content, text_block.content
            )

            if content_sim < self.content_similarity_threshold:
                continue

            # Calculate spatial proximity
            spatial_prox = self._calculate_spatial_proximity(
                chunk.bounding_box,
                {
                    "x0": text_block.bbox.x0,
                    "y0": text_block.bbox.y0,
                    "x1": text_block.bbox.x1,
                    "y1": text_block.bbox.y1,
                },
            )

            # Combined confidence score (weighted average)
            confidence = (content_sim * 0.7) + (spatial_prox * 0.3)

            if best_match is None or confidence > best_match[0]:
                best_match = (confidence, {"text_block": text_block})

        # Also check tables if chunk is a table
        if chunk.chunk_type == "table":
            for table in document.tables:
                if table.page_num not in chunk.page_refs:
                    continue

                # Convert table to text for comparison
                table_text = self._table_to_text(table)
                content_sim = self._calculate_content_similarity(
                    chunk.content, table_text
                )

                if content_sim < self.content_similarity_threshold:
                    continue

                spatial_prox = self._calculate_spatial_proximity(
                    chunk.bounding_box,
                    {
                        "x0": table.bbox.x0,
                        "y0": table.bbox.y0,
                        "x1": table.bbox.x1,
                        "y1": table.bbox.y1,
                    },
                )

                confidence = (content_sim * 0.7) + (spatial_prox * 0.3)

                if best_match is None or confidence > best_match[0]:
                    best_match = (confidence, {"table": table})

        # Also check figures if chunk is a figure
        if chunk.chunk_type == "figure":
            for figure in document.figures:
                if figure.page_num not in chunk.page_refs:
                    continue

                # Compare captions
                chunk_caption = chunk.metadata.get("caption", chunk.content)
                figure_caption = figure.caption or ""

                content_sim = self._calculate_content_similarity(
                    chunk_caption, figure_caption
                )

                if content_sim < self.content_similarity_threshold:
                    continue

                spatial_prox = self._calculate_spatial_proximity(
                    chunk.bounding_box,
                    {
                        "x0": figure.bbox.x0,
                        "y0": figure.bbox.y0,
                        "x1": figure.bbox.x1,
                        "y1": figure.bbox.y1,
                    },
                )

                confidence = (content_sim * 0.7) + (spatial_prox * 0.3)

                if best_match is None or confidence > best_match[0]:
                    best_match = (confidence, {"figure": figure})

        if best_match is None:
            logger.debug(
                f"Chunk {chunk.content_hash[:8]} not found in document"
            )
            return False, 0.0

        confidence = best_match[0]
        found = confidence >= self.content_similarity_threshold

        logger.debug(
            f"Chunk {chunk.content_hash[:8]} verification: found={found}, "
            f"confidence={confidence:.3f}"
        )

        return found, confidence

    def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings.

        Uses a combination of exact matching and fuzzy matching (character overlap).

        Args:
            text1: First text string.
            text2: Second text string.

        Returns:
            Similarity score from 0.0 to 1.0.
        """
        # Normalize whitespace
        text1_norm = " ".join(text1.split())
        text2_norm = " ".join(text2.split())

        # Exact match
        if text1_norm == text2_norm:
            return 1.0

        # Check if one is a substring of the other
        if text1_norm in text2_norm or text2_norm in text1_norm:
            # Calculate overlap ratio
            shorter = min(len(text1_norm), len(text2_norm))
            longer = max(len(text1_norm), len(text2_norm))
            return shorter / longer if longer > 0 else 0.0

        # Character-level similarity (Jaccard-like)
        set1 = set(text1_norm.lower())
        set2 = set(text2_norm.lower())

        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _calculate_spatial_proximity(
        self, bbox1: Dict[str, float], bbox2: Dict[str, float]
    ) -> float:
        """Calculate spatial proximity between two bounding boxes.

        Args:
            bbox1: First bounding box with keys 'x0', 'y0', 'x1', 'y1'.
            bbox2: Second bounding box with keys 'x0', 'y0', 'x1', 'y1'.

        Returns:
            Proximity score from 0.0 to 1.0. 1.0 = exact match, 0.0 = far apart.
        """
        # Calculate center points
        center1_x = (bbox1.get("x0", 0) + bbox1.get("x1", 0)) / 2
        center1_y = (bbox1.get("y0", 0) + bbox1.get("y1", 0)) / 2
        center2_x = (bbox2.get("x0", 0) + bbox2.get("x1", 0)) / 2
        center2_y = (bbox2.get("y0", 0) + bbox2.get("y1", 0)) / 2

        # Calculate distance between centers
        dx = center1_x - center2_x
        dy = center1_y - center2_y
        distance = (dx**2 + dy**2) ** 0.5

        # Calculate size similarity
        width1 = abs(bbox1.get("x1", 0) - bbox1.get("x0", 0))
        height1 = abs(bbox1.get("y1", 0) - bbox1.get("y0", 0))
        width2 = abs(bbox2.get("x1", 0) - bbox2.get("x0", 0))
        height2 = abs(bbox2.get("y1", 0) - bbox2.get("y0", 0))

        size1 = width1 * height1
        size2 = width2 * height2

        if size1 == 0 and size2 == 0:
            size_sim = 1.0
        elif size1 == 0 or size2 == 0:
            size_sim = 0.0
        else:
            size_sim = min(size1, size2) / max(size1, size2)

        # Distance-based proximity (exponential decay)
        # If distance is within tolerance, proximity is high
        if distance <= self.spatial_tolerance:
            distance_prox = 1.0
        else:
            # Exponential decay: proximity decreases as distance increases
            # Scale by tolerance so that 2x tolerance = ~0.6, 3x = ~0.4, etc.
            distance_prox = max(0.0, 1.0 - (distance / (self.spatial_tolerance * 3)))

        # Combined proximity (weighted average)
        proximity = (distance_prox * 0.7) + (size_sim * 0.3)

        return proximity

    def _table_to_text(self, table) -> str:
        """Convert a table to text representation for comparison.

        Args:
            table: Table object with headers and rows.

        Returns:
            Text representation of the table.
        """
        lines = []
        if table.headers:
            lines.append(" | ".join(str(h) for h in table.headers))
        for row in table.rows:
            lines.append(" | ".join(str(cell) for cell in row))
        return "\n".join(lines)


# Convenience functions for direct use
def generate_content_hash(
    content: str, page_num: int, bbox: Dict[str, float]
) -> str:
    """Generate a content hash (convenience function).

    Args:
        content: The text content to hash.
        page_num: 1-indexed page number.
        bbox: Bounding box dictionary.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    hasher = ContentHasher()
    return hasher.generate_content_hash(content, page_num, bbox)


def generate_spatial_hash(text: str, coordinates: Dict[str, float]) -> str:
    """Generate a spatial hash (convenience function).

    Args:
        text: Text content.
        coordinates: Coordinate dictionary.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    hasher = ContentHasher()
    return hasher.generate_spatial_hash(text, coordinates)


def verify_hash(chunk: LDU, document: ExtractedDocument) -> Tuple[bool, float]:
    """Verify a chunk's hash against a document (convenience function).

    Args:
        chunk: The LDU chunk to verify.
        document: The ExtractedDocument to search in.

    Returns:
        Tuple of (found: bool, confidence: float).
    """
    hasher = ContentHasher()
    return hasher.verify_hash(chunk, document)
