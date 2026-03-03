"""Manual test script for FastTextExtractor.

This script demonstrates how to use FastTextExtractor and shows the extracted
content. Run it with:

    python notebooks/test_fast_text_manual.py <path_to_pdf>

Or without arguments to test with a sample document from the corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.triage import TriageAgent
from src.strategies.fast_text import FastTextExtractor


def main() -> None:
    """Run manual test of FastTextExtractor."""
    # Determine PDF path
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        # Default to a known test document
        pdf_path = Path("data/class_a/CBE_ANNUAL_REPORT_2023-24.pdf")
        if not pdf_path.exists():
            pdf_path = Path("data/class_d/tax_expenditure_ethiopia_2021_22.pdf")

    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        print("\nUsage: python notebooks/test_fast_text_manual.py <path_to_pdf>")
        sys.exit(1)

    print("=" * 70)
    print(f"Testing FastTextExtractor on: {pdf_path.name}")
    print("=" * 70)

    # Step 1: Triage the document
    print("\n[1] Running Triage Agent...")
    profiles_dir = Path(".refinery/profiles")
    profiles_dir.mkdir(parents=True, exist_ok=True)
    triage = TriageAgent(profiles_dir=profiles_dir)
    profile = triage.classify_document(pdf_path)
    print(f"   Origin Type: {profile.origin_type}")
    print(f"   Layout Complexity: {profile.layout_complexity}")
    print(f"   Domain Hint: {profile.domain_hint}")
    print(f"   Estimated Cost: {profile.estimated_cost}")

    # Step 2: Check if fast text can handle it
    print("\n[2] Checking if FastTextExtractor can handle this document...")
    extractor = FastTextExtractor()
    can_handle = extractor.can_handle(profile)
    print(f"   Can Handle: {can_handle}")

    if not can_handle:
        print("\n   ⚠️  Warning: FastTextExtractor may not be optimal for this document")
        print("   Consider using a layout-aware or vision-augmented strategy instead")

    # Step 3: Calculate confidence score
    print("\n[3] Calculating confidence score...")
    confidence = extractor.confidence_score(str(pdf_path))
    print(f"   Confidence: {confidence:.4f} ({confidence*100:.1f}%)")

    # Step 4: Estimate cost
    print("\n[4] Estimating cost...")
    cost = extractor.cost_estimate(str(pdf_path))
    print(f"   Total Cost: ${cost['total_cost_usd']:.4f}")
    print(f"   Cost per Page: ${cost['cost_per_page']:.4f}")

    # Step 5: Extract content
    print("\n[5] Extracting content...")
    try:
        extracted = extractor.extract(str(pdf_path))
        print(f"   ✓ Extraction successful!")

        # Summary statistics
        print("\n[6] Extraction Summary:")
        print(f"   Text Blocks: {len(extracted.text_blocks):,}")
        print(f"   Tables: {len(extracted.tables)}")
        print(f"   Figures: {len(extracted.figures)}")
        print(f"   Reading Order: {len(extracted.reading_order)} indices")

        # Show sample text blocks
        if extracted.text_blocks:
            print("\n[7] Sample Text Blocks (first 5):")
            for i, block in enumerate(extracted.text_blocks[:5], 1):
                content_preview = block.content[:50] + "..." if len(block.content) > 50 else block.content
                print(f"   {i}. Page {block.page_num}: {content_preview!r}")
                print(f"      BBox: ({block.bbox.x0:.1f}, {block.bbox.y0:.1f}) to ({block.bbox.x1:.1f}, {block.bbox.y1:.1f})")

        # Show sample tables
        if extracted.tables:
            print("\n[8] Sample Tables (first 2):")
            for i, table in enumerate(extracted.tables[:2], 1):
                print(f"   Table {i} (Page {table.page_num}):")
                print(f"      Headers: {table.headers}")
                print(f"      Rows: {len(table.rows)}")
                if table.rows:
                    print(f"      First Row: {table.rows[0]}")

        # Show sample figures
        if extracted.figures:
            print("\n[9] Sample Figures (first 3):")
            for i, figure in enumerate(extracted.figures[:3], 1):
                print(f"   Figure {i} (Page {figure.page_num}):")
                print(f"      Caption: {figure.caption or '(none)'}")
                print(f"      BBox: ({figure.bbox.x0:.1f}, {figure.bbox.y0:.1f}) to ({figure.bbox.x1:.1f}, {figure.bbox.y1:.1f})")

        print("\n" + "=" * 70)
        print("✓ Test completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n   ✗ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
