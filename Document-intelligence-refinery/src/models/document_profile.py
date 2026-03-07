"""Document profile model for triage classification."""

from pydantic import BaseModel, Field

from .common import (
    DomainHint,
    EstimatedExtractionCost,
    LanguageInfo,
    LayoutComplexity,
    OriginType,
    StrategyName,
)


class TriageSignals(BaseModel):
    """Signals collected during triage analysis."""

    avg_char_density: float = Field(ge=0.0, description="Average character density per page")
    avg_whitespace_ratio: float = Field(ge=0.0, le=1.0, description="Average whitespace ratio")
    avg_image_area_ratio: float = Field(ge=0.0, le=1.0, description="Average image area ratio")
    table_density: float = Field(ge=0.0, description="Table density (tables per page)")
    figure_density: float = Field(ge=0.0, description="Figure density (figures per page)")


class DocumentProfile(BaseModel):
    """Document classification profile from triage agent."""

    doc_id: str = Field(min_length=4, description="Unique document identifier")
    document_name: str = Field(min_length=1, description="Document filename")
    origin_type: OriginType = Field(description="Document origin type")
    layout_complexity: LayoutComplexity = Field(description="Layout complexity classification")
    language: LanguageInfo = Field(description="Detected language information")
    domain_hint: DomainHint = Field(description="Domain classification hint")
    estimated_extraction_cost: EstimatedExtractionCost = Field(
        description="Estimated extraction cost category"
    )
    triage_signals: TriageSignals = Field(description="Raw triage signals")
    selected_strategy: StrategyName = Field(description="Recommended extraction strategy")
    triage_confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence score for triage classification"
    )
