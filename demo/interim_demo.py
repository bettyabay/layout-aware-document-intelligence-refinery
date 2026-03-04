"""Interim submission demo script.

This script processes one document from each class and generates:
- DocumentProfile JSON files
- extraction_ledger.jsonl entries
- Cost analysis per strategy
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.agents.extractor import ExtractionRouter
from src.agents.triage import TriageAgent
from src.models.document_profile import DocumentProfile
from src.models.extracted_document import ExtractedDocument
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout_aware import LayoutExtractor
from src.strategies.vision_augmented import VisionExtractor

# Setup paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
REFINERY_DIR = BASE_DIR / ".refinery"
PROFILES_DIR = REFINERY_DIR / "profiles"
LEDGER_PATH = REFINERY_DIR / "extraction_ledger.jsonl"

# Create directories
REFINERY_DIR.mkdir(exist_ok=True)
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

# Document classes to process
DOCUMENT_CLASSES = {
    "class_a": "CBE_ANNUAL_REPORT_2023-24.pdf",
    "class_b": "Annual_Report_JUNE-2023.pdf",
    "class_c": "fta_performance_survey_final_report_2022.pdf",
    "class_d": "tax_expenditure_ethiopia_2021_22.pdf",
}


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_profile_summary(profile: DocumentProfile) -> None:
    """Print a summary of the document profile."""
    print(f"Document ID: {profile.doc_id}")
    print(f"Origin Type: {profile.origin_type}")
    print(f"Layout Complexity: {profile.layout_complexity}")
    print(f"Domain Hint: {profile.domain_hint}")
    print(f"Estimated Cost: {profile.estimated_cost}")
    print(f"Language: {profile.language} (confidence: {profile.language_confidence:.2%})")
    print(f"Pages: {profile.metadata.page_count}")
    print(f"File Size: {profile.metadata.size_bytes / 1024 / 1024:.2f} MB")


def analyze_costs(
    document_path: Path, profile: DocumentProfile
) -> Dict[str, Dict[str, float]]:
    """Calculate cost estimates for all strategies."""
    costs = {}

    strategies = {
        "fast_text": FastTextExtractor(),
    }
    
    # Try to add layout_aware if available
    try:
        strategies["layout_aware"] = LayoutExtractor()
    except Exception as e:
        print(f"  Note: Layout-aware extraction not available: {e}")
    
    # Try to add vision_augmented if available
    try:
        strategies["vision_augmented"] = VisionExtractor()
    except Exception as e:
        print(f"  Note: Vision extraction not available: {e}")

    for name, strategy in strategies.items():
        try:
            cost_est = strategy.cost_estimate(str(document_path))
            costs[name] = {
                "total_cost_usd": cost_est.get("total_cost_usd", 0.0),
                "cost_per_page": cost_est.get("cost_per_page", 0.0),
            }
        except Exception as e:
            print(f"  Warning: Could not estimate cost for {name}: {e}")
            costs[name] = {"total_cost_usd": 0.0, "cost_per_page": 0.0}

    return costs


def print_cost_analysis(costs: Dict[str, Dict[str, float]], page_count: int) -> None:
    """Print cost analysis table."""
    print("\nCost Analysis per Strategy:")
    print("-" * 80)
    print(f"{'Strategy':<20} {'Cost/Page':<15} {'Total Cost':<15} {'For {page_count} pages':<20}")
    print("-" * 80)

    for strategy_name, cost_data in costs.items():
        cost_per_page = cost_data.get("cost_per_page", 0.0)
        total_cost = cost_data.get("total_cost_usd", 0.0)
        print(
            f"{strategy_name:<20} "
            f"${cost_per_page:<14.6f} "
            f"${total_cost:<14.6f} "
            f"${total_cost:<19.6f}"
        )
    print("-" * 80)


def get_ledger_summary(ledger_path: Path, doc_id: str) -> List[Dict]:
    """Get summary of ledger entries for a document."""
    entries = []
    if ledger_path.exists():
        with ledger_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("document_id") == doc_id:
                        entries.append(entry)
    return entries


def print_ledger_summary(entries: List[Dict]) -> None:
    """Print summary of ledger entries."""
    if not entries:
        print("  No ledger entries found.")
        return

    # Group by strategy
    by_strategy: Dict[str, List[Dict]] = {}
    for entry in entries:
        strategy = entry.get("strategy_used", "unknown")
        if strategy not in by_strategy:
            by_strategy[strategy] = []
        by_strategy[strategy].append(entry)

    print(f"\n  Total ledger entries: {len(entries)}")
    print(f"  Strategies used: {', '.join(by_strategy.keys())}")

    for strategy, strategy_entries in by_strategy.items():
        if strategy_entries:
            first_entry = strategy_entries[0]
            confidence = first_entry.get("confidence_score", 0.0)
            cost = first_entry.get("cost_estimate", {}).get("total_cost_usd", 0.0)
            time = first_entry.get("processing_time_seconds", 0.0)
            escalation = first_entry.get("escalation_path", [])
            success = first_entry.get("success", False)

            print(f"\n  Strategy: {strategy}")
            print(f"    Confidence: {confidence:.2%}")
            print(f"    Cost: ${cost:.6f}")
            print(f"    Processing Time: {time:.2f}s")
            print(f"    Escalation Path: {' → '.join(escalation)}")
            print(f"    Success: {success}")


def process_document(
    class_name: str, filename: str, triage_agent: TriageAgent, extraction_router: ExtractionRouter
) -> None:
    """Process a single document."""
    print_section(f"Processing {class_name}: {filename}")

    document_path = DATA_DIR / class_name / filename

    if not document_path.exists():
        print(f"  ERROR: Document not found: {document_path}")
        return

    # Stage 1: Triage
    print("Stage 1: Document Triage")
    print("-" * 80)
    try:
        profile = triage_agent.classify_document(document_path)
        print_profile_summary(profile)

        # Save profile
        profile_path = PROFILES_DIR / f"{profile.doc_id}.json"
        profile_path.write_text(profile.to_json(indent=2), encoding="utf-8")
        print(f"\n  ✓ Profile saved to: {profile_path}")

    except Exception as e:
        print(f"  ERROR: Triage failed: {e}")
        return

    # Cost analysis
    print("\nCost Analysis")
    print("-" * 80)
    costs = analyze_costs(document_path, profile)
    print_cost_analysis(costs, profile.metadata.page_count)

    # Stage 2: Extraction
    print("\nStage 2: Document Extraction")
    print("-" * 80)
    try:
        extracted = extraction_router.extract(profile, str(document_path))
        print(f"  ✓ Extraction successful")
        print(f"    Text blocks: {len(extracted.text_blocks)}")
        print(f"    Tables: {len(extracted.tables)}")
        print(f"    Figures: {len(extracted.figures)}")

    except RuntimeError as e:
        error_msg = str(e)
        if "All extraction strategies exhausted" in error_msg:
            print(f"  ⚠ WARNING: All strategies failed (check ledger for details)")
            print(f"    This may be due to:")
            print(f"    - Memory issues with layout-aware extraction (Docling)")
            print(f"    - Missing OPENROUTER_API_KEY for vision extraction")
            print(f"    - Resource constraints on your system")
            print(f"\n    Attempting fallback to fast_text extraction...")
            
            # Try fast_text as fallback
            try:
                from src.strategies.fast_text import FastTextExtractor
                fast_extractor = FastTextExtractor()
                extracted = fast_extractor.extract(str(document_path))
                print(f"  ✓ Fallback extraction successful (fast_text)")
                print(f"    Text blocks: {len(extracted.text_blocks)}")
                print(f"    Tables: {len(extracted.tables)}")
                print(f"    Figures: {len(extracted.figures)}")
            except Exception as fallback_error:
                print(f"  ✗ Fallback also failed: {fallback_error}")
                print(f"    Note: Profile and cost analysis were still generated")
        else:
            print(f"  ERROR: Extraction failed: {e}")
            return
    except Exception as e:
        print(f"  ERROR: Extraction failed: {e}")
        print(f"    Note: Profile was still generated and saved")
        return

    # Ledger summary
    print("\nExtraction Ledger Summary")
    print("-" * 80)
    entries = get_ledger_summary(LEDGER_PATH, profile.doc_id)
    print_ledger_summary(entries)


def main() -> None:
    """Main demo function."""
    print_section("Interim Submission Demo")
    print("Processing documents from all classes...")
    print(f"Output directory: {REFINERY_DIR}")
    print(f"Profiles directory: {PROFILES_DIR}")
    print(f"Ledger file: {LEDGER_PATH}")

    # Initialize agents
    triage_agent = TriageAgent(profiles_dir=PROFILES_DIR)
    extraction_router = ExtractionRouter(ledger_path=LEDGER_PATH, confidence_threshold=0.7)

    # Process each document class
    for class_name, filename in DOCUMENT_CLASSES.items():
        try:
            process_document(class_name, filename, triage_agent, extraction_router)
        except Exception as e:
            print(f"\n  ERROR processing {class_name}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print_section("Demo Complete")
    print(f"Profiles created: {len(list(PROFILES_DIR.glob('*.json')))}")
    if LEDGER_PATH.exists():
        with LEDGER_PATH.open("r", encoding="utf-8") as f:
            ledger_lines = sum(1 for line in f if line.strip())
        print(f"Ledger entries: {ledger_lines}")
    print(f"\nView results:")
    print(f"  - Profiles: {PROFILES_DIR}")
    print(f"  - Ledger: {LEDGER_PATH}")
    print(f"\nStart web UI: uvicorn src.web.app:app --reload")


if __name__ == "__main__":
    main()
