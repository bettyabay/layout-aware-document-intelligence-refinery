"""Models for provenance tracking and citation.

The provenance layer allows every answer from the Query Agent to be traced back
to concrete regions in the source documents.
"""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field, ConfigDict


class ProvenanceChain(BaseModel):
    """Provenance information for a single citation or answer.

    Attributes:
        document_name: Name of the source document.
        page_number: 1-indexed page number where the evidence resides.
        bbox: Bounding box of the cited region (keys like 'x0', 'y0', 'x1', 'y1').
        content_hash: Spatial hash of the cited content, used for verification.
        verification_status: Whether the claim has been independently verified.
    """

    model_config = ConfigDict(extra="forbid")

    document_name: str = Field(..., description="Name of the source document")
    page_number: int = Field(
        ...,
        ge=1,
        description="1-indexed page number of the cited content",
    )
    bbox: Dict[str, float] = Field(
        default_factory=dict,
        description="Bounding box of the cited region in PDF points",
    )
    content_hash: str = Field(
        ...,
        min_length=1,
        description="Spatial hash of the cited content for verification",
    )
    verification_status: bool = Field(
        default=False,
        description="True if the claim has been verified against the source",
    )


__all__ = [
    "ProvenanceChain",
]

