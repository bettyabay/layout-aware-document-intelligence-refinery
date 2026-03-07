"""Extracted document models with spatial information."""

import hashlib
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .common import StrategyName

ChunkType = Literal["paragraph", "table", "figure", "list", "heading", "mixed"]


class BBox(BaseModel):
    """Bounding box coordinates."""

    x0: float = Field(description="Left coordinate")
    y0: float = Field(description="Top coordinate")
    x1: float = Field(description="Right coordinate")
    y1: float = Field(description="Bottom coordinate")

    @model_validator(mode="after")
    def validate_order(self) -> "BBox":
        """Validate bounding box coordinates."""
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("bbox must satisfy x1>=x0 and y1>=y0")
        return self


class ProvenanceChain(BaseModel):
    """Provenance chain entry for tracking source of information."""

    document_name: str = Field(min_length=1, description="Source document name")
    page_number: int = Field(ge=1, description="Page number (1-indexed)")
    bbox: BBox = Field(description="Bounding box coordinates")
    content_hash: str = Field(min_length=8, description="Content hash for verification")
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Extraction confidence score"
    )
    text_excerpt: Optional[str] = Field(default=None, description="Text excerpt for display")


class LDU(BaseModel):
    """Logical Document Unit - semantic chunk with metadata."""

    id: str = Field(min_length=1, description="Unique LDU identifier")
    text: str = Field(default="", description="Text content")
    content_hash: str = Field(min_length=8, description="Content hash")
    chunk_type: ChunkType = Field(default="paragraph", description="Chunk type")
    bounding_box: Optional[BBox] = Field(default=None, description="Bounding box")
    token_count: Optional[int] = Field(default=None, description="Token count")
    parent_section: Optional[str] = Field(default=None, description="Parent section identifier")
    previous_chunk_id: Optional[str] = Field(default=None, description="Previous chunk ID")
    next_chunk_id: Optional[str] = Field(default=None, description="Next chunk ID")
    reference_ids: list[str] = Field(default_factory=list, description="Referenced chunk IDs")
    page_refs: list[int] = Field(default_factory=list, description="Page references")
    provenance_chain: list[ProvenanceChain] = Field(
        default_factory=list, description="Provenance chain"
    )

    @model_validator(mode="after")
    def validate_page_refs(self) -> "LDU":
        """Validate page references."""
        if any(page < 1 for page in self.page_refs):
            raise ValueError("page_refs must contain page numbers >= 1")
        return self


class PageIndexNode(BaseModel):
    """PageIndex tree node for hierarchical navigation."""

    id: str = Field(description="Node identifier")
    node_type: str = Field(description="Node type (section, subsection, etc.)")
    label: Optional[str] = Field(default=None, description="Node label")
    page_number: Optional[int] = Field(default=None, ge=1, description="Page number")
    bbox: Optional[BBox] = Field(default=None, description="Bounding box")
    summary: Optional[str] = Field(default=None, description="LLM-generated summary")
    children: list["PageIndexNode"] = Field(default_factory=list, description="Child nodes")


PageIndexNode.model_rebuild()


class TextBlock(BaseModel):
    """Extracted text block with spatial information."""

    id: str = Field(description="Text block identifier")
    text: str = Field(description="Text content")
    bbox: BBox = Field(description="Bounding box")
    reading_order: int = Field(ge=0, description="Reading order index")


class TableObject(BaseModel):
    """Extracted table with structured data."""

    id: str = Field(description="Table identifier")
    title: Optional[str] = Field(default=None, description="Table title")
    headers: list[str] = Field(default_factory=list, description="Column headers")
    rows: list[list[str]] = Field(default_factory=list, description="Table rows")
    bbox: BBox = Field(description="Bounding box")
    reading_order: int = Field(default=0, ge=0, description="Reading order index")


class FigureObject(BaseModel):
    """Extracted figure with caption."""

    id: str = Field(description="Figure identifier")
    caption: Optional[str] = Field(default=None, description="Figure caption")
    bbox: BBox = Field(description="Bounding box")
    references: list[str] = Field(default_factory=list, description="Reference IDs")
    reading_order: int = Field(default=0, ge=0, description="Reading order index")


class ExtractedPage(BaseModel):
    """Extracted page with all elements."""

    page_number: int = Field(ge=1, description="Page number (1-indexed)")
    width: float = Field(gt=0, description="Page width")
    height: float = Field(gt=0, description="Page height")
    text_blocks: list[TextBlock] = Field(default_factory=list, description="Text blocks")
    tables: list[TableObject] = Field(default_factory=list, description="Tables")
    figures: list[FigureObject] = Field(default_factory=list, description="Figures")
    ldu_ids: list[str] = Field(default_factory=list, description="Associated LDU IDs")


class ExtractedMetadata(BaseModel):
    """Metadata for extracted document."""

    source_strategy: StrategyName = Field(description="Final extraction strategy used")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence score")
    strategy_sequence: list[StrategyName] = Field(
        default_factory=list, description="Strategy escalation sequence"
    )


class ExtractedDocument(BaseModel):
    """Complete extracted document with all elements."""

    doc_id: str = Field(min_length=4, description="Document identifier")
    document_name: str = Field(min_length=1, description="Document name")
    pages: list[ExtractedPage] = Field(default_factory=list, description="Extracted pages")
    metadata: ExtractedMetadata = Field(description="Extraction metadata")
    ldus: list[LDU] = Field(default_factory=list, description="Logical Document Units")
    page_index: Optional[PageIndexNode] = Field(default=None, description="PageIndex tree")
    provenance_chains: list[ProvenanceChain] = Field(
        default_factory=list, description="Provenance chains"
    )


def content_hash_for_text(text: str) -> str:
    """Generate content hash for text."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def estimate_token_count(text: str) -> int:
    """Estimate token count for text (rough approximation)."""
    if not text:
        return 0
    # Rough approximation: 1 token ≈ 0.75 words
    word_count = len((text or "").split())
    return max(1, int(word_count * 1.33))
