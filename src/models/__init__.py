"""Pydantic models for the Document Intelligence Refinery.

This package exposes the core data structures used across the 5-stage
pipeline: document profiles, extracted documents, logical document units,
PageIndex trees, and provenance chains.
"""

from .document_profile import (
    DocumentProfile,
    DomainHint,
    EstimatedCost,
    LayoutComplexity,
    OriginType,
    FileMetadata,
)
from .extracted_document import (
    BoundingBox,
    ExtractedDocument,
    Figure,
    Table,
    TextBlock,
)
from .ldu import ChunkType, LDU
from .page_index import DataType, PageIndex, Section
from .provenance import ProvenanceChain

__all__ = [
    # Document profile
    "OriginType",
    "LayoutComplexity",
    "DomainHint",
    "EstimatedCost",
    "FileMetadata",
    "DocumentProfile",
    # Extracted document
    "BoundingBox",
    "TextBlock",
    "Table",
    "Figure",
    "ExtractedDocument",
    # Logical Document Units
    "ChunkType",
    "LDU",
    # PageIndex
    "DataType",
    "Section",
    "PageIndex",
    # Provenance
    "ProvenanceChain",
]

