"""Document profile models for triage classification.

This module defines the ``DocumentProfile`` Pydantic model, which is the typed
output of the Stage 1 Triage Agent. It captures the key classification
dimensions that drive downstream extraction strategy selection.
"""

from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator

# Typed aliases for classification dimensions
OriginType = Literal["native_digital", "scanned_image", "mixed", "form_fillable"]
LayoutComplexity = Literal[
    "single_column", "multi_column", "table_heavy", "figure_heavy", "mixed"
]
DomainHint = Literal["financial", "legal", "technical", "medical", "general"]
EstimatedCost = Literal[
    "fast_text_sufficient",
    "needs_layout_model",
    "needs_vision_model",
]


class FileMetadata(BaseModel):
    """File-level metadata used by the triage and extraction stages.

    This metadata is *advisory* – it does not drive routing directly, but it
    provides useful context for logging, auditing, and debugging.

    Attributes:
        path: Absolute or workspace-relative path to the file.
        size_bytes: File size in bytes.
        page_count: Number of pages in the document.
        mime_type: Detected MIME type for the document.
        checksum: Optional checksum (e.g. SHA256) of the raw file contents.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    path: str = Field(..., description="Absolute or workspace-relative file path")
    size_bytes: int = Field(..., ge=0, description="File size in bytes")
    page_count: int = Field(..., ge=1, description="Number of pages in the document")
    mime_type: str = Field(..., description="Detected MIME type for the document")
    checksum: str | None = Field(
        default=None,
        description="Optional checksum (e.g. SHA256) of the raw file contents",
    )


class DocumentProfile(BaseModel):
    """Document classification profile produced by the Triage Agent.

    The ``DocumentProfile`` encapsulates all the high-level characteristics of a
    document that influence how it should be processed by the extraction
    pipeline.

    It is designed to be:

    * **Serializable** – can be stored in ``.refinery/profiles`` as JSON.
    * **Stable** – classification decisions are deterministic for a given file.
    * **Actionable** – each field maps directly to strategy choices.

    Attributes:
        origin_type: Detected document origin type.
        layout_complexity: Detected layout complexity category.
        language: ISO 639-1 language code.
        language_confidence: Confidence score for ``language`` in [0.0, 1.0].
        domain_hint: High-level domain classification of the document.
        estimated_cost: Estimated extraction cost tier.
        metadata: File metadata and additional key–value information.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    origin_type: OriginType = Field(
        ...,
        description=(
            "Document origin classification: 'native_digital', 'scanned_image', "
            "'mixed', or 'form_fillable'."
        ),
    )
    layout_complexity: LayoutComplexity = Field(
        ...,
        description=(
            "Layout complexity classification: 'single_column', 'multi_column', "
            "'table_heavy', 'figure_heavy', or 'mixed'."
        ),
    )

    language: str = Field(
        ...,
        min_length=2,
        max_length=10,
        description="Detected language code (typically ISO 639-1, e.g. 'en').",
    )
    language_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Language detection confidence score in [0.0, 1.0].",
    )

    domain_hint: DomainHint = Field(
        ...,
        description=(
            "Domain classification hint used for prompt and strategy selection. "
            "One of: 'financial', 'legal', 'technical', 'medical', 'general'."
        ),
    )

    estimated_cost: EstimatedCost = Field(
        ...,
        description=(
            "Estimated extraction cost tier: "
            "'fast_text_sufficient', 'needs_layout_model', or 'needs_vision_model'."
        ),
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary file metadata. Expected keys include: 'file' (path), "
            "'size_bytes', 'page_count', 'mime_type', 'checksum'."
        ),
    )

    @field_validator("language")
    @classmethod
    def _normalise_language(cls, value: str) -> str:
        """Normalise language codes to lowercase.

        Args:
            value: Language string supplied by the caller.

        Returns:
            Normalised language string.
        """
        return value.strip().lower()

    @field_validator("metadata")
    @classmethod
    def _ensure_core_metadata(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure that mandatory metadata keys are present if possible.

        This validator does **not** raise if metadata is incomplete – it simply
        normalises and provides sensible defaults where necessary.

        Args:
            value: Raw metadata dictionary.

        Returns:
            Normalised metadata dictionary.
        """
        normalised = dict(value)
        # Normalise key names
        if "path" in normalised and "file" not in normalised:
            normalised["file"] = normalised["path"]
        if "file" in normalised and "path" not in normalised:
            normalised["path"] = normalised["file"]
        return normalised

    # ---------------------------------------------------------------------
    # Convenience serialization helpers
    # ---------------------------------------------------------------------
    def to_json(self, *, indent: int = 2) -> str:
        """Serialise the profile to a JSON string.

        Args:
            indent: Indentation level for pretty-printing.

        Returns:
            JSON string representation of the profile.
        """
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "DocumentProfile":
        """Deserialise a profile from a JSON string.

        Args:
            data: JSON string produced by :meth:`to_json`.

        Returns:
            Parsed :class:`DocumentProfile` instance.
        """
        return cls.model_validate_json(data)


__all__ = [
    "OriginType",
    "LayoutComplexity",
    "DomainHint",
    "EstimatedCost",
    "FileMetadata",
    "DocumentProfile",
]

