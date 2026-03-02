"""Example script demonstrating character density analysis.

This script shows how to use the character density analyzer to classify
PDF documents and determine the appropriate extraction strategy.

Usage:
    python notebooks/character_density_example.py <path_to_pdf>
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.character_density_analyzer import (
    analyze_character_density,
    print_analysis_summary,
    save_analysis_results,
)


def main():
    """Run character density analysis on a PDF file."""
    if len(sys.argv) < 2:
        print("Usage: python character_density_example.py <path_to_pdf>")
        print("\nExample:")
        print("  python notebooks/character_density_example.py data/class_a/example.pdf")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    print(f"Analyzing PDF: {pdf_path}")
    print("-" * 60)

    try:
        # Run analysis
        results = analyze_character_density(pdf_path)

        # Print summary
        print_analysis_summary(results)

        # Save results
        output_dir = Path(".refinery/analyses")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{pdf_path.stem}_analysis.json"
        save_analysis_results(results, output_path)

        print(f"\nDetailed results saved to: {output_path}")

        # Show per-page details for first 5 pages
        print("\nPer-Page Details (first 5 pages):")
        print("-" * 60)
        for page in results["pages"][:5]:
            print(
                f"Page {page['page']:3d}: "
                f"{page['char_count']:6,} chars, "
                f"density={page['density']:8.6f}, "
                f"fonts={page['has_fonts']}, "
                f"image_ratio={page['image_ratio']*100:5.2f}%"
            )

        if len(results["pages"]) > 5:
            print(f"... and {len(results['pages']) - 5} more pages")

    except Exception as e:
        print(f"Error analyzing PDF: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
