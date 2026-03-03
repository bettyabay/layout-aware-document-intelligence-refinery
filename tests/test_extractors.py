"""Unit tests for extraction strategies."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.triage import TriageAgent
from src.models.document_profile import DocumentProfile
from src.strategies.fast_text import FastTextExtractor


DATA_DIR = Path("data")


@pytest.fixture
def fast_text_extractor() -> FastTextExtractor:
    """FastTextExtractor instance for tests."""
    return FastTextExtractor()


@pytest.fixture
def profiles_dir(tmp_path) -> Path:
    """Temporary profiles directory for tests."""
    return tmp_path / "profiles"


@pytest.fixture
def triage_agent(profiles_dir: Path) -> TriageAgent:
    """TriageAgent instance for creating profiles."""
    return TriageAgent(profiles_dir=profiles_dir)


@pytest.mark.parametrize(
    "relative_path",
    [
        "class_a/CBE_ANNUAL_REPORT_2023-24.pdf",
        "class_d/tax_expenditure_ethiopia_2021_22.pdf",
    ],
)
def test_fast_text_extraction_basic(
    fast_text_extractor: FastTextExtractor,
    relative_path: str,
) -> None:
    """Test that FastTextExtractor can extract basic content from native digital PDFs."""
    pdf_path = DATA_DIR / relative_path
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")

    # Extract document
    extracted = fast_text_extractor.extract(str(pdf_path))

    # Verify structure
    assert extracted is not None
    assert len(extracted.text_blocks) > 0, "Should extract at least some text blocks"
    assert len(extracted.reading_order) == len(extracted.text_blocks)

    # Verify text blocks have required fields
    for block in extracted.text_blocks[:10]:  # Check first 10
        assert block.content, "Text block should have content"
        assert block.page_num >= 1, "Page number should be >= 1"
        assert block.bbox.x0 < block.bbox.x1, "Bounding box should be valid"
        assert block.bbox.y0 < block.bbox.y1, "Bounding box should be valid"


def test_fast_text_extraction_tables(
    fast_text_extractor: FastTextExtractor,
) -> None:
    """Test table extraction from documents with tables."""
    pdf_path = DATA_DIR / "class_d/tax_expenditure_ethiopia_2021_22.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")

    extracted = fast_text_extractor.extract(str(pdf_path))

    # This document should have tables
    # Note: May be 0 if pdfplumber doesn't detect tables, but structure should be valid
    for table in extracted.tables:
        assert len(table.headers) > 0, "Table should have headers"
        assert len(table.rows) >= 0, "Table rows can be empty"
        assert table.page_num >= 1, "Table should have valid page number"
        assert table.bbox.x0 < table.bbox.x1, "Table bbox should be valid"


def test_fast_text_extraction_figures(
    fast_text_extractor: FastTextExtractor,
) -> None:
    """Test figure/image extraction."""
    pdf_path = DATA_DIR / "class_c/fta_performance_survey_final_report_2022.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")

    extracted = fast_text_extractor.extract(str(pdf_path))

    # Figures may or may not be present, but structure should be valid
    for figure in extracted.figures:
        assert figure.page_num >= 1, "Figure should have valid page number"
        assert figure.bbox.x0 < figure.bbox.x1, "Figure bbox should be valid"
        assert figure.bbox.y0 < figure.bbox.y1, "Figure bbox should be valid"


def test_fast_text_confidence_score(
    fast_text_extractor: FastTextExtractor,
) -> None:
    """Test confidence scoring for fast text extraction."""
    pdf_path = DATA_DIR / "class_a/CBE_ANNUAL_REPORT_2023-24.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")

    confidence = fast_text_extractor.confidence_score(str(pdf_path))

    assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"
    # Native digital PDFs should have reasonable confidence
    assert confidence > 0.3, "Native digital PDF should have some confidence"


def test_fast_text_cost_estimate(
    fast_text_extractor: FastTextExtractor,
) -> None:
    """Test cost estimation for fast text extraction."""
    pdf_path = DATA_DIR / "class_a/CBE_ANNUAL_REPORT_2023-24.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")

    cost = fast_text_extractor.cost_estimate(str(pdf_path))

    assert "total_cost_usd" in cost, "Cost estimate should include total_cost_usd"
    assert "cost_per_page" in cost, "Cost estimate should include cost_per_page"
    assert cost["total_cost_usd"] == 0.0, "Fast text extraction should be free"
    assert cost["cost_per_page"] == 0.0, "Fast text extraction should be free per page"


def test_fast_text_can_handle(
    fast_text_extractor: FastTextExtractor,
    triage_agent: TriageAgent,
) -> None:
    """Test that can_handle correctly identifies suitable documents."""
    pdf_path = DATA_DIR / "class_a/CBE_ANNUAL_REPORT_2023-24.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")

    # Create profile
    profile = triage_agent.classify_document(pdf_path)

    # Check if fast text can handle it
    can_handle = fast_text_extractor.can_handle(profile)

    # Should handle if native_digital and single_column
    expected = (
        profile.origin_type == "native_digital"
        and profile.layout_complexity == "single_column"
    )
    assert can_handle == expected, f"can_handle should return {expected} for this profile"


def test_fast_text_cannot_handle_scanned(
    fast_text_extractor: FastTextExtractor,
    triage_agent: TriageAgent,
) -> None:
    """Test that fast text cannot handle scanned documents."""
    pdf_path = DATA_DIR / "class_b/Annual_Report_JUNE-2023.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")

    # Create profile
    profile = triage_agent.classify_document(pdf_path)

    # Fast text should not handle scanned documents
    can_handle = fast_text_extractor.can_handle(profile)

    if profile.origin_type == "scanned_image":
        assert not can_handle, "Fast text should not handle scanned documents"


def test_fast_text_extraction_file_not_found(
    fast_text_extractor: FastTextExtractor,
) -> None:
    """Test that extract raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        fast_text_extractor.extract("nonexistent_file.pdf")


def test_fast_text_confidence_file_not_found(
    fast_text_extractor: FastTextExtractor,
) -> None:
    """Test that confidence_score raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        fast_text_extractor.confidence_score("nonexistent_file.pdf")


def test_fast_text_cost_file_not_found(
    fast_text_extractor: FastTextExtractor,
) -> None:
    """Test that cost_estimate raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        fast_text_extractor.cost_estimate("nonexistent_file.pdf")
