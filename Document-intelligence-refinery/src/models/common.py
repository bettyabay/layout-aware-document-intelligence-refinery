"""Common enums and types for Document Intelligence Refinery."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class OriginType(str, Enum):
    """Document origin type classification."""

    NATIVE_DIGITAL = "native_digital"
    SCANNED_IMAGE = "scanned_image"
    MIXED = "mixed"
    FORM_FILLABLE = "form_fillable"


class LayoutComplexity(str, Enum):
    """Layout complexity classification."""

    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE_HEAVY = "table_heavy"
    FIGURE_HEAVY = "figure_heavy"
    MIXED = "mixed"


class DomainHint(str, Enum):
    """Domain classification hints."""

    FINANCIAL = "financial"
    LEGAL = "legal"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    ACADEMIC = "academic"
    GENERAL = "general"


class StrategyName(str, Enum):
    """Extraction strategy names."""

    A = "fast_text"  # Strategy A: Fast text extraction
    B = "layout_aware"  # Strategy B: Layout-aware extraction
    C = "vision_augmented"  # Strategy C: Vision-augmented extraction


class EstimatedExtractionCost(str, Enum):
    """Estimated extraction cost category."""

    FREE = "free"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    NEEDS_LAYOUT_MODEL = "needs_layout_model"
    NEEDS_VISION_MODEL = "needs_vision_model"


class LanguageInfo(BaseModel):
    """Language information."""

    code: str = Field(default="en", description="Language code (ISO 639-1)")
    name: str = Field(default="English", description="Language name")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence")

    def __repr__(self) -> str:
        return f"LanguageInfo(code={self.code}, name={self.name}, confidence={self.confidence})"
