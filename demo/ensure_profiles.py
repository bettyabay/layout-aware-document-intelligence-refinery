"""Helper script to ensure we have at least 12 processed documents (3 per class).

This script processes documents multiple times or from multiple sources to meet
the requirement of 12 documents in .refinery/profiles/.
"""

from __future__ import annotations

from pathlib import Path

from src.agents.extractor import ExtractionRouter
from src.agents.triage import TriageAgent

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
REFINERY_DIR = BASE_DIR / ".refinery"
PROFILES_DIR = REFINERY_DIR / "profiles"
LEDGER_PATH = REFINERY_DIR / "extraction_ledger.jsonl"

# Document classes
DOCUMENT_CLASSES = {
    "class_a": "CBE_ANNUAL_REPORT_2023-24.pdf",
    "class_b": "Annual_Report_JUNE-2023.pdf",
    "class_c": "fta_performance_survey_final_report_2022.pdf",
    "class_d": "tax_expenditure_ethiopia_2021_22.pdf",
}

# Target: 3 documents per class = 12 total
TARGET_PER_CLASS = 3


def ensure_profiles() -> None:
    """Ensure we have at least 12 processed documents."""
    REFINERY_DIR.mkdir(exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    triage_agent = TriageAgent(profiles_dir=PROFILES_DIR)
    extraction_router = ExtractionRouter(ledger_path=LEDGER_PATH, confidence_threshold=0.5)

    existing_profiles = {f.stem for f in PROFILES_DIR.glob("*.json")}

    print("Ensuring at least 12 processed documents (3 per class)...\n")

    for class_name, filename in DOCUMENT_CLASSES.items():
        document_path = DATA_DIR / class_name / filename

        if not document_path.exists():
            print(f"  ⚠ Skipping {class_name}: {filename} not found")
            continue

        # Check if profile exists for this document
        doc_id = document_path.stem
        if doc_id in existing_profiles:
            print(f"  ✓ {class_name}: Already processed ({doc_id})")
            continue

        print(f"  Processing {class_name}: {filename}")

        # Process the document (it will create a profile)
        # Note: Since we only have 1 document per class, we'll process it once
        # In a real scenario, you would have multiple different documents per class
        try:
            profile = triage_agent.classify_document(document_path)
            
            # Save profile (triage already saves it, but ensure it exists)
            profile_path = PROFILES_DIR / f"{profile.doc_id}.json"
            if not profile_path.exists():
                profile_path.write_text(profile.to_json(indent=2), encoding="utf-8")

            # Run extraction
            extraction_router.extract(profile, str(document_path))

            print(f"    ✓ Processed: {profile.doc_id}")

        except Exception as e:
            print(f"    ✗ Error: {e}")

    # Final count
    final_count = len(list(PROFILES_DIR.glob("*.json")))
    print(f"\n✓ Total profiles: {final_count}")
    
    if final_count >= 12:
        print("✓ Requirement met: At least 12 profiles created")
    else:
        print(f"⚠ Warning: Only {final_count} profiles created (need 12)")
        print("  Note: To meet the 12-document requirement, you need multiple")
        print("  different documents per class (3 per class). The current")
        print("  data directory has 1 document per class.")


if __name__ == "__main__":
    ensure_profiles()
