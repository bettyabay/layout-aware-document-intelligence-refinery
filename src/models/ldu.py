"""Models for Logical Document Units (LDUs).

LDUs are the atomic, semantically coherent chunks that the semantic chunking
engine emits. They preserve spatial provenance and structural context while
remaining RAG-friendly.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator

ChunkType = Literal["paragraph", "table", "figure", "list", "header", "footnote"]


class CrossReference(BaseModel):
    """Cross-reference to another chunk or document element.

    Attributes:
        target_id: ID of the referenced chunk (content_hash or custom ID).
        reference_type: Type of reference (e.g., "table", "figure", "section").
        anchor_text: The text that triggered the reference (e.g., "see Table 3").
    """

    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(..., description="ID of the referenced chunk")
    reference_type: str = Field(..., description="Type of reference")
    anchor_text: str = Field(..., description="Text that triggered the reference")


class LDU(BaseModel):
    """Logical Document Unit (LDU).

    Attributes:
        content: Raw text content for this chunk.
        chunk_type: Semantic type of the chunk.
        page_refs: 1-indexed page numbers where this chunk appears.
        bounding_box: Spatial bounding box (typically of the dominant region),
            encoded as a dictionary with keys like ``x0``, ``y0``, ``x1``, ``y1``,
            and optionally ``page``.
        parent_section: Optional title of the parent section.
        token_count: Token count estimate for this chunk (for budgeting).
        content_hash: Spatial hash of the content and coordinates used for
            provenance and deduplication.
        metadata: Additional metadata (e.g., captions, headers, formatting).
        children: List of content hashes or IDs of child chunks.
        cross_references: List of cross-references to other chunks or elements.
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., description="Chunk text content")
    chunk_type: ChunkType = Field(..., description="Semantic type of the chunk")
    page_refs: List[int] = Field(
        ...,
        min_length=1,
        description="1-indexed page numbers where this chunk appears",
    )
    bounding_box: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Spatial bounding box for the chunk. Expected keys include "
            "'x0', 'y0', 'x1', 'y1', and optionally 'page'."
        ),
    )
    parent_section: Optional[str] = Field(
        default=None,
        description="Title of the parent section, if known",
    )
    token_count: int = Field(
        ...,
        ge=0,
        description="Token count estimate for this chunk",
    )
    content_hash: str = Field(
        default="",
        description=(
            "Spatial hash of the chunk content and coordinates. If not provided, "
            "it is computed automatically."
        ),
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional metadata for this chunk. Common keys include: "
            "'caption' (for figures), 'header_level' (for headers), "
            "'list_type' (for lists), 'table_headers' (for tables)."
        ),
    )
    children: List[str] = Field(
        default_factory=list,
        description=(
            "List of content hashes or IDs of child chunks. Used for hierarchical "
            "relationships (e.g., section header -> paragraph chunks)."
        ),
    )
    cross_references: List[CrossReference] = Field(
        default_factory=list,
        description=(
            "List of cross-references to other chunks or document elements. "
            "Resolved references (e.g., 'see Table 3') are stored here."
        ),
    )

    @model_validator(mode="after")
    def _ensure_content_hash(self) -> "LDU":
        """Ensure that ``content_hash`` is populated with a spatial hash.

        The hash is computed over both the textual content and the spatial
        coordinates (if present) using SHA256. This makes the hash robust to
        re-pagination while still tied to the *semantic* content.
        """
        if not self.content_hash:
            self.content_hash = self.compute_hash()
        return self

    # ------------------------------------------------------------------
    # Spatial hashing
    # ------------------------------------------------------------------
    def compute_hash(self) -> str:
        """Compute a deterministic spatial hash for this chunk.

        The hash is based on:

        * ``content`` (normalised to UTF-8)
        * ``page_refs`` (sorted)
        * ``bounding_box`` (sorted keys)

        Returns:
            Hex-encoded SHA256 hash string.
        """
        payload: Dict[str, Any] = {
            "content": self.content,
            "page_refs": sorted(self.page_refs),
            "bounding_box": self._normalised_bbox(),
        }
        data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def _normalised_bbox(self) -> Dict[str, float]:
        """Normalise the bounding box representation.

        Returns:
            Bounding box dict with only serialisable keys and float values.
        """
        return {k: float(v) for k, v in self.bounding_box.items()}


__all__ = [
    "ChunkType",
    "CrossReference",
    "LDU",
]

