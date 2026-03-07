"""Extraction ledger entry model."""

from pydantic import BaseModel, Field

from .common import StrategyName


class ExtractionLedgerEntry(BaseModel):
    """Entry in extraction ledger for audit trail."""

    timestamp: str = Field(description="ISO timestamp")
    doc_id: str = Field(description="Document identifier")
    document_name: str = Field(description="Document name")
    strategy_sequence: list[StrategyName] = Field(description="Strategy escalation sequence")
    final_strategy: StrategyName = Field(description="Final strategy used")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Final confidence score")
    cost_estimate_usd: float = Field(ge=0.0, description="Total cost estimate")
    processing_time_ms: int = Field(ge=0, description="Processing time in milliseconds")
    budget_cap_usd: float = Field(ge=0.0, description="Budget cap")
    budget_status: str = Field(description="Budget status")
    notes: str | None = Field(default=None, description="Additional notes")
