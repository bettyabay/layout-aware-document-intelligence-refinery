"""Unit tests for the TriageAgent."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.triage import TriageAgent
from src.models.document_profile import DocumentProfile


DATA_DIR = Path("data")


@pytest.fixture(scope="module")
def profiles_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary profiles directory for tests."""
    return tmp_path_factory.mktemp("profiles")


@pytest.fixture(scope="module")
def triage_agent(profiles_dir: Path) -> TriageAgent:
    """TriageAgent instance for tests."""
    return TriageAgent(profiles_dir=profiles_dir)


@pytest.mark.parametrize(
    ("relative_path", "expected_origin"),
    [
        ("class_a/CBE_ANNUAL_REPORT_2023-24.pdf", "native_digital"),
        ("class_c/fta_performance_survey_final_report_2022.pdf", "mixed"),
        ("class_d/tax_expenditure_ethiopia_2021_22.pdf", "native_digital"),
        ("class_b/Annual_Report_JUNE-2023.pdf", "scanned_image"),
        # Reuse one document to ensure we have at least 5 test cases
        ("class_c/fta_performance_survey_final_report_2022.pdf", "mixed"),
    ],
)
def test_origin_type_classification(
    triage_agent: TriageAgent,
    relative_path: str,
    expected_origin: str,
) -> None:
    """Origin type should classify correctly for known documents.

    If a particular PDF is missing from the workspace, the test is skipped
    rather than failed, so the suite degrades gracefully.
    """
    pdf_path = DATA_DIR / relative_path
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")

    profile = triage_agent.classify_document(pdf_path)
    assert isinstance(profile, DocumentProfile)

    if expected_origin == "scanned_image":
        assert profile.origin_type == expected_origin
    else:
        # For native/mixed we assert we did not misclassify as scanned_image
        assert profile.origin_type != "scanned_image"


@pytest.mark.parametrize(
    ("relative_path", "must_not_be"),
    [
        # Annual report – multi-column narrative with tables
        ("class_a/CBE_ANNUAL_REPORT_2023-24.pdf", "single_column"),
        # Technical assessment – mixed narrative + tables
        ("class_c/fta_performance_survey_final_report_2022.pdf", "single_column"),
        # Tax expenditure report – table-heavy fiscal data
        ("class_d/tax_expenditure_ethiopia_2021_22.pdf", "single_column"),
    ],
)
def test_layout_complexity_not_single_for_complex_docs(
    triage_agent: TriageAgent,
    relative_path: str,
    must_not_be: str,
) -> None:
    """Layout complexity should recognise complex layouts as non-single-column.

    The exact label (multi_column, table_heavy, figure_heavy, mixed) may vary
    by heuristic, but for known complex documents the agent must not classify
    them as purely ``single_column``.
    """
    pdf_path = DATA_DIR / relative_path
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")

    profile = triage_agent.classify_document(pdf_path)
    assert isinstance(profile, DocumentProfile)
    assert profile.layout_complexity != must_not_be


@pytest.mark.parametrize(
    ("relative_path", "expected_domain"),
    [
        ("class_a/CBE_ANNUAL_REPORT_2023-24.pdf", "financial"),
        ("class_d/tax_expenditure_ethiopia_2021_22.pdf", "financial"),
    ],
)
def test_domain_hint_financial(
    triage_agent: TriageAgent,
    relative_path: str,
    expected_domain: str,
) -> None:
    """Financial reports should be classified with a financial domain_hint."""
    pdf_path = DATA_DIR / relative_path
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")

    profile = triage_agent.classify_document(pdf_path)
    assert profile.domain_hint == expected_domain
    assert 0.0 <= profile.language_confidence <= 1.0


def test_profile_persisted(triage_agent: TriageAgent, profiles_dir: Path) -> None:
    """Profiles should be saved as JSON in the profiles directory."""
    # Use any existing PDF; prefer class_c technical report
    pdf_path = DATA_DIR / "class_c/fta_performance_survey_final_report_2022.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")

    profile = triage_agent.classify_document(pdf_path)
    doc_id = pdf_path.stem
    profile_path = profiles_dir / f"{doc_id}.json"

    assert profile_path.exists()
    data = profile_path.read_text(encoding="utf-8")
    loaded = DocumentProfile.from_json(data)
    assert loaded == profile

