"""Comprehensive tests for all document classes.

This test suite validates the extraction pipeline on all document classes:
- Class A: Annual financial reports (mixed origin, complex layouts)
- Class B: Scanned documents (scanned_image origin)
- Class C: Technical assessment reports (mixed layouts)
- Class D: Structured data reports (table-heavy, native digital)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.extractor import ExtractionRouter
from src.agents.triage import TriageAgent
from src.models.document_profile import DocumentProfile
from src.models.extracted_document import ExtractedDocument

DATA_DIR = Path("data")


@pytest.fixture(scope="module")
def profiles_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary profiles directory for tests."""
    return tmp_path_factory.mktemp("profiles")


@pytest.fixture(scope="module")
def ledger_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary ledger directory for tests."""
    return tmp_path_factory.mktemp("ledger")


@pytest.fixture(scope="module")
def triage_agent(profiles_dir: Path) -> TriageAgent:
    """TriageAgent instance for tests."""
    return TriageAgent(profiles_dir=profiles_dir)


@pytest.fixture(scope="module")
def extraction_router(ledger_dir: Path) -> ExtractionRouter:
    """ExtractionRouter instance for tests."""
    ledger_path = ledger_dir / "extraction_ledger.jsonl"
    return ExtractionRouter(ledger_path=ledger_path, confidence_threshold=0.74)


def get_ledger_entries(ledger_path: Path, doc_id: str) -> list[dict]:
    """Get ledger entries for a document."""
    entries = []
    if ledger_path.exists():
        with ledger_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("document_id") == doc_id:
                        entries.append(entry)
    return entries


def get_strategy_by_page(ledger_path: Path, doc_id: str) -> dict[int, str]:
    """Get strategy used for each page from ledger."""
    entries = get_ledger_entries(ledger_path, doc_id)
    # Group by page and get the final successful strategy for each page
    page_strategies: dict[int, str] = {}
    for entry in entries:
        page_num = entry.get("page_num")
        if page_num and entry.get("success"):
            # Use the last successful strategy for each page
            page_strategies[page_num] = entry.get("strategy_used", "unknown")
    return page_strategies


def test_class_a_annual_report(
    triage_agent: TriageAgent,
    extraction_router: ExtractionRouter,
    ledger_dir: Path,
) -> None:
    """Test on CBE Annual Report (Class A).

    Expected behavior:
    - Mixed origin type (some pages native, some scanned)
    - Complex layout (multi-column, tables, figures)
    - Should use layout-aware or vision strategies for complex pages
    """
    doc_path = DATA_DIR / "class_a" / "CBE_ANNUAL_REPORT_2023-24.pdf"
    if not doc_path.exists():
        pytest.skip(f"Test PDF not found: {doc_path}")

    # Stage 1: Triage
    profile = triage_agent.classify_document(doc_path)
    assert isinstance(profile, DocumentProfile)
    assert profile.origin_type in ("native_digital", "mixed", "scanned_image")
    assert profile.layout_complexity != "single_column"  # Should be complex

    # Stage 2: Extraction
    extracted = extraction_router.extract(profile, str(doc_path))
    assert isinstance(extracted, ExtractedDocument)
    assert len(extracted.text_blocks) > 0

    # Check ledger for strategy usage
    ledger_path = ledger_dir / "extraction_ledger.jsonl"
    entries = get_ledger_entries(ledger_path, profile.doc_id)
    assert len(entries) > 0, "Ledger should have entries"

    # Get the final strategy used (from successful entries)
    successful_entries = [e for e in entries if e.get("success")]
    if successful_entries:
        final_strategy = successful_entries[-1].get("strategy_used")
        assert final_strategy in (
            "fast_text",
            "layout_aware",
            "layout_mineru",
            "vision_augmented",
        ), f"Unexpected strategy: {final_strategy}"

        # For complex documents, should not use fast_text (or escalate from it)
        escalation_paths = [e.get("escalation_path", []) for e in successful_entries]
        if any("fast_text" in path for path in escalation_paths):
            # If fast_text was tried, should have escalated
            assert any(
                len(path) > 1 for path in escalation_paths
            ), "Should have escalated from fast_text for complex document"

    # Check that tables were extracted
    assert len(extracted.tables) > 0, "Should extract tables from annual report"


def test_class_b_scanned(
    triage_agent: TriageAgent,
    extraction_router: ExtractionRouter,
    ledger_dir: Path,
) -> None:
    """Test on scanned document (Class B).

    Expected behavior:
    - Should detect scanned_image origin (0% font metadata)
    - Should use vision_augmented strategy for all pages
    """
    doc_path = DATA_DIR / "class_b" / "Annual_Report_JUNE-2023.pdf"
    if not doc_path.exists():
        pytest.skip(f"Test PDF not found: {doc_path}")

    # Stage 1: Triage
    profile = triage_agent.classify_document(doc_path)
    assert isinstance(profile, DocumentProfile)

    # Should detect scanned (or at least not native_digital)
    assert profile.origin_type in (
        "scanned_image",
        "mixed",
    ), f"Expected scanned_image or mixed, got {profile.origin_type}"

    # Stage 2: Extraction
    extracted = extraction_router.extract(profile, str(doc_path))
    assert isinstance(extracted, ExtractedDocument)

    # Check ledger for strategy usage
    ledger_path = ledger_dir / "extraction_ledger.jsonl"
    entries = get_ledger_entries(ledger_path, profile.doc_id)
    assert len(entries) > 0, "Ledger should have entries"

    # For scanned documents, should use vision_augmented
    successful_entries = [e for e in entries if e.get("success")]
    if successful_entries:
        final_strategy = successful_entries[-1].get("strategy_used")
        # Scanned documents should escalate to vision
        if profile.origin_type == "scanned_image":
            assert final_strategy == "vision_augmented", (
                f"Scanned document should use vision_augmented, "
                f"got {final_strategy}"
            )

    # Should have extracted some text (even if scanned)
    assert len(extracted.text_blocks) > 0, "Should extract text from scanned document"


def test_class_c_technical(
    triage_agent: TriageAgent,
    extraction_router: ExtractionRouter,
    ledger_dir: Path,
) -> None:
    """Test on FTA performance survey report (Class C).

    Expected behavior:
    - Mixed layout complexity (narrative + tables)
    - Should use appropriate strategy based on confidence
    """
    doc_path = DATA_DIR / "class_c" / "fta_performance_survey_final_report_2022.pdf"
    if not doc_path.exists():
        pytest.skip(f"Test PDF not found: {doc_path}")

    # Stage 1: Triage
    profile = triage_agent.classify_document(doc_path)
    assert isinstance(profile, DocumentProfile)
    assert profile.layout_complexity in (
        "multi_column",
        "table_heavy",
        "figure_heavy",
        "mixed",
    ), f"Expected complex layout, got {profile.layout_complexity}"

    # Stage 2: Extraction
    extracted = extraction_router.extract(profile, str(doc_path))
    assert isinstance(extracted, ExtractedDocument)
    assert len(extracted.text_blocks) > 0

    # Check ledger for strategy usage
    ledger_path = ledger_dir / "extraction_ledger.jsonl"
    entries = get_ledger_entries(ledger_path, profile.doc_id)
    assert len(entries) > 0, "Ledger should have entries"

    # Should have used a strategy appropriate for mixed layouts
    successful_entries = [e for e in entries if e.get("success")]
    if successful_entries:
        final_strategy = successful_entries[-1].get("strategy_used")
        assert final_strategy in (
            "fast_text",
            "layout_aware",
            "layout_mineru",
            "vision_augmented",
        )

        # Check confidence scores
        confidences = [e.get("confidence_score", 0.0) for e in successful_entries]
        if confidences:
            final_confidence = confidences[-1]
            assert 0.0 <= final_confidence <= 1.0
            # If confidence was low, should have escalated
            if final_confidence < 0.7:
                escalation_paths = [
                    e.get("escalation_path", []) for e in successful_entries
                ]
                assert any(
                    len(path) > 1 for path in escalation_paths
                ), "Should have escalated if confidence was low"


def test_class_d_tax(
    triage_agent: TriageAgent,
    extraction_router: ExtractionRouter,
    ledger_dir: Path,
) -> None:
    """Test on tax expenditure report (Class D).

    Expected behavior:
    - Native digital, table-heavy document
    - Should use fast_text or layout-aware strategy
    - Table extraction should work
    """
    doc_path = DATA_DIR / "class_d" / "tax_expenditure_ethiopia_2021_22.pdf"
    if not doc_path.exists():
        pytest.skip(f"Test PDF not found: {doc_path}")

    # Stage 1: Triage
    profile = triage_agent.classify_document(doc_path)
    assert isinstance(profile, DocumentProfile)

    # Should be native digital or mixed (not scanned)
    assert profile.origin_type in (
        "native_digital",
        "mixed",
    ), f"Expected native_digital or mixed, got {profile.origin_type}"

    # Should recommend fast_text or needs_layout_model
    assert profile.estimated_cost in (
        "fast_text_sufficient",
        "needs_layout_model",
    ), f"Expected fast_text_sufficient or needs_layout_model, got {profile.estimated_cost}"

    # Stage 2: Extraction
    extracted = extraction_router.extract(profile, str(doc_path))
    assert isinstance(extracted, ExtractedDocument)

    # Table extraction should work
    assert len(extracted.tables) > 0, "Should extract tables from tax report"

    # Check ledger for strategy usage
    ledger_path = ledger_dir / "extraction_ledger.jsonl"
    entries = get_ledger_entries(ledger_path, profile.doc_id)
    assert len(entries) > 0, "Ledger should have entries"

    # Should have used fast_text or layout-aware (not vision)
    successful_entries = [e for e in entries if e.get("success")]
    if successful_entries:
        final_strategy = successful_entries[-1].get("strategy_used")
        # For native digital table-heavy docs, should not need vision
        assert final_strategy in (
            "fast_text",
            "layout_aware",
            "layout_mineru",
        ), f"Should use fast_text or layout-aware for native digital, got {final_strategy}"

    # Should have extracted text blocks
    assert len(extracted.text_blocks) > 0, "Should extract text blocks"


def test_escalation_guard(
    triage_agent: TriageAgent,
    extraction_router: ExtractionRouter,
    ledger_dir: Path,
) -> None:
    """Test that escalation guard works correctly.

    If initial strategy has low confidence, should escalate to next strategy.
    """
    # Use a document that might need escalation
    doc_path = DATA_DIR / "class_c" / "fta_performance_survey_final_report_2022.pdf"
    if not doc_path.exists():
        pytest.skip(f"Test PDF not found: {doc_path}")

    profile = triage_agent.classify_document(doc_path)
    extracted = extraction_router.extract(profile, str(doc_path))

    # Check ledger for escalation paths
    ledger_path = ledger_dir / "extraction_ledger.jsonl"
    entries = get_ledger_entries(ledger_path, profile.doc_id)

    if entries:
        # Check that escalation paths are logged
        escalation_paths = [e.get("escalation_path", []) for e in entries]
        assert all(
            len(path) > 0 for path in escalation_paths
        ), "All entries should have escalation paths"

        # If any strategy had low confidence, should see escalation
        low_confidence_entries = [
            e for e in entries if e.get("confidence_score", 1.0) < 0.7
        ]
        if low_confidence_entries:
            # Should have escalated
            assert any(
                len(e.get("escalation_path", [])) > 1 for e in low_confidence_entries
            ), "Should escalate when confidence is low"


def test_ledger_logging(
    triage_agent: TriageAgent,
    extraction_router: ExtractionRouter,
    ledger_dir: Path,
) -> None:
    """Test that extraction ledger logs all required information."""
    doc_path = DATA_DIR / "class_d" / "tax_expenditure_ethiopia_2021_22.pdf"
    if not doc_path.exists():
        pytest.skip(f"Test PDF not found: {doc_path}")

    profile = triage_agent.classify_document(doc_path)
    extracted = extraction_router.extract(profile, str(doc_path))

    # Check ledger entries
    ledger_path = ledger_dir / "extraction_ledger.jsonl"
    entries = get_ledger_entries(ledger_path, profile.doc_id)

    assert len(entries) > 0, "Should have ledger entries"

    # Check that each entry has required fields
    required_fields = [
        "document_id",
        "page_num",
        "strategy_used",
        "confidence_score",
        "cost_estimate",
        "processing_time_seconds",
        "escalation_path",
        "success",
        "timestamp",
    ]

    for entry in entries:
        for field in required_fields:
            assert field in entry, f"Entry missing required field: {field}"

        # Check types
        assert isinstance(entry["document_id"], str)
        assert isinstance(entry["page_num"], int)
        assert isinstance(entry["strategy_used"], str)
        assert isinstance(entry["confidence_score"], (int, float))
        assert isinstance(entry["cost_estimate"], dict)
        assert isinstance(entry["processing_time_seconds"], (int, float))
        assert isinstance(entry["escalation_path"], list)
        assert isinstance(entry["success"], bool)

    # Should have one entry per page
    page_nums = {e["page_num"] for e in entries}
    assert len(page_nums) == profile.metadata.page_count, (
        f"Should have entries for all {profile.metadata.page_count} pages, "
        f"got {len(page_nums)}"
    )


def test_extraction_quality(
    triage_agent: TriageAgent,
    extraction_router: ExtractionRouter,
) -> None:
    """Test that extraction produces reasonable quality results."""
    doc_path = DATA_DIR / "class_d" / "tax_expenditure_ethiopia_2021_22.pdf"
    if not doc_path.exists():
        pytest.skip(f"Test PDF not found: {doc_path}")

    profile = triage_agent.classify_document(doc_path)
    extracted = extraction_router.extract(profile, str(doc_path))

    # Should have extracted content
    assert len(extracted.text_blocks) > 0, "Should extract text blocks"
    assert len(extracted.tables) > 0, "Should extract tables"

    # Text blocks should have content
    non_empty_blocks = [tb for tb in extracted.text_blocks if tb.content.strip()]
    assert len(non_empty_blocks) > 0, "Should have non-empty text blocks"

    # Tables should have structure
    for table in extracted.tables:
        assert len(table.headers) > 0 or len(table.rows) > 0, "Tables should have content"

    # All elements should have valid page numbers
    all_page_nums = (
        [tb.page_num for tb in extracted.text_blocks]
        + [t.page_num for t in extracted.tables]
        + [f.page_num for f in extracted.figures]
    )
    if all_page_nums:
        assert all(
            1 <= p <= profile.metadata.page_count for p in all_page_nums
        ), "All page numbers should be valid"
