"""Models for PageIndex navigation structure.

The PageIndex is a hierarchical, section-based navigation tree that allows
agents to locate relevant parts of a long document before performing vector
search. It is inspired by VectifyAI's PageIndex.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, ConfigDict

DataType = Literal["table", "figure", "equation", "list", "text"]


class Section(BaseModel):
    """A section node in the PageIndex tree.

    Attributes:
        title: Section title.
        page_start: First page (1-indexed) where this section appears.
        page_end: Last page (1-indexed) where this section appears.
        child_sections: Nested subsections.
        key_entities: Named entities relevant to this section.
        summary: Short LLM-generated summary (2–3 sentences).
        data_types_present: Types of data present in this section.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., description="Section title")
    page_start: int = Field(..., ge=1, description="First page of the section (1-indexed)")
    page_end: int = Field(..., ge=1, description="Last page of the section (1-indexed)")
    child_sections: List["Section"] = Field(
        default_factory=list,
        description="Nested subsections",
    )
    key_entities: List[str] = Field(
        default_factory=list,
        description="Named entities relevant to this section",
    )
    summary: str = Field(
        "",
        description="LLM-generated summary (2–3 sentences)",
    )
    data_types_present: List[DataType] = Field(
        default_factory=list,
        description="Types of data present (tables, figures, equations, lists, text)",
    )


# Enable recursive typing for child_sections
Section.model_rebuild()


class PageIndex(BaseModel):
    """Hierarchical navigation index for a document.

    Attributes:
        doc_id: Stable identifier for the document.
        doc_name: Human-readable document name (e.g. filename).
        root_sections: Top-level sections in the document.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(..., description="Stable identifier for the document")
    doc_name: str = Field(..., description="Document filename or display name")
    root_sections: List[Section] = Field(
        default_factory=list,
        description="Top-level sections in the document",
    )


__all__ = [
    "DataType",
    "Section",
    "PageIndex",
]

