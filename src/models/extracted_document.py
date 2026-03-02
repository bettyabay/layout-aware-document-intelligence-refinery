"""Models for extracted document structure.

This module defines the ``ExtractedDocument`` model – the normalised
representation that all extraction strategies must emit before semantic
chunking. It is intentionally simple and focuses on the minimal fields needed
for downstream stages.
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field, ConfigDict


class BoundingBox(BaseModel):
    """Bounding box coordinates in PDF points.

    Attributes:
        x0: Left coordinate.
        y0: Bottom coordinate.
        x1: Right coordinate.
        y1: Top coordinate.
    """

    model_config = ConfigDict(frozen=True)

    x0: float = Field(..., description="Left coordinate (points)")
    y0: float = Field(..., description="Bottom coordinate (points)")
    x1: float = Field(..., description="Right coordinate (points)")
    y1: float = Field(..., description="Top coordinate (points)")


class TextBlock(BaseModel):
    """Text block with spatial information.

    Attributes:
        content: Raw text content of the block.
        bbox: Bounding box of the text block.
        page_num: 1-indexed page number.
    """

    model_config = ConfigDict(frozen=True)

    content: str = Field(..., description="Extracted text content")
    bbox: BoundingBox = Field(..., description="Bounding box for the text block")
    page_num: int = Field(..., ge=1, description="1-indexed page number")


class Table(BaseModel):
    """Structured table extracted from a document.

    Attributes:
        headers: Header row cells.
        rows: Table rows (list of list of cell values).
        bbox: Bounding box of the entire table.
        page_num: 1-indexed page number.
    """

    model_config = ConfigDict(frozen=True)

    headers: List[str] = Field(..., description="Header row cells")
    rows: List[List[str]] = Field(..., description="Table body rows")
    bbox: BoundingBox = Field(..., description="Bounding box of the table")
    page_num: int = Field(..., ge=1, description="1-indexed page number")


class Figure(BaseModel):
    """Figure or image region extracted from a document.

    Attributes:
        caption: Figure caption text, if any.
        bbox: Bounding box of the figure.
        page_num: 1-indexed page number.
    """

    model_config = ConfigDict(frozen=True)

    caption: str = Field("", description="Figure caption text (may be empty)")
    bbox: BoundingBox = Field(..., description="Bounding box of the figure")
    page_num: int = Field(..., ge=1, description="1-indexed page number")


class ExtractedDocument(BaseModel):
    """Unified representation of a document after extraction.

    Attributes:
        text_blocks: List of text blocks with coordinates and page numbers.
        tables: List of table structures with headers and rows.
        figures: List of figures with captions.
        reading_order: Optional reading order indices referring to ``text_blocks``.
    """

    model_config = ConfigDict(extra="forbid")

    text_blocks: List[TextBlock] = Field(
        default_factory=list,
        description="Text blocks extracted from the document",
    )
    tables: List[Table] = Field(
        default_factory=list,
        description="Tables extracted from the document",
    )
    figures: List[Figure] = Field(
        default_factory=list,
        description="Figures extracted from the document",
    )
    reading_order: List[int] = Field(
        default_factory=list,
        description=(
            "Optional reading-order indices into 'text_blocks'. If empty, the "
            "consumer may infer order from page_num + spatial position."
        ),
    )


__all__ = [
    "BoundingBox",
    "TextBlock",
    "Table",
    "Figure",
    "ExtractedDocument",
]

