"""Character density analysis utility for document triage."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pdfplumber

logger = logging.getLogger(__name__)


def analyze_character_density(pdf_path: Path) -> Dict[str, Any]:
    """Analyze character density metrics for a PDF.

    This function performs character density analysis to distinguish
    native digital PDFs from scanned images. It measures:
    - Total characters per page
    - Character density (chars per square point)
    - Font metadata presence
    - Image area ratio

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dictionary containing analysis results with per-page and summary metrics.

    Raises:
        FileNotFoundError: If PDF file does not exist.
        Exception: If PDF cannot be opened or processed.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    results: Dict[str, Any] = {
        "file": str(pdf_path),
        "file_size_bytes": pdf_path.stat().st_size,
        "pages": [],
        "summary": {},
    }

    total_chars = 0
    total_area = 0
    pages_with_fonts = 0
    total_image_area = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

            for i, page in enumerate(pdf.pages, 1):
                # Extract text
                text = page.extract_text() or ""
                char_count = len(text)

                # Get page dimensions
                width = page.width
                height = page.height
                area = width * height

                # Character density (chars per square point)
                density = char_count / area if area > 0 else 0.0

                # Check for font metadata (chars with font info)
                chars = page.chars if hasattr(page, "chars") else []
                has_fonts = len(chars) > 0
                if has_fonts:
                    pages_with_fonts += 1

                # Count unique fonts
                unique_fonts = len(set(char.get("fontname", "") for char in chars)) if chars else 0

                # Image area (approximate)
                images = page.images if hasattr(page, "images") else []
                image_area = sum(img.get("width", 0) * img.get("height", 0) for img in images)
                image_ratio = image_area / area if area > 0 else 0.0

                # Whitespace analysis
                whitespace_chars = text.count(" ") + text.count("\n") + text.count("\t")
                whitespace_ratio = whitespace_chars / char_count if char_count > 0 else 0.0

                page_data = {
                    "page": i,
                    "char_count": char_count,
                    "density": round(density, 6),
                    "has_fonts": has_fonts,
                    "unique_fonts": unique_fonts,
                    "image_area": round(image_area, 2),
                    "image_ratio": round(image_ratio, 4),
                    "whitespace_ratio": round(whitespace_ratio, 4),
                    "width": round(width, 2),
                    "height": round(height, 2),
                    "area": round(area, 2),
                }
                results["pages"].append(page_data)

                total_chars += char_count
                total_area += area
                total_image_area += image_area

            # Summary statistics
            avg_density = total_chars / total_area if total_area > 0 else 0.0
            avg_chars_per_page = total_chars / total_pages if total_pages > 0 else 0.0
            avg_image_ratio = total_image_area / total_area if total_area > 0 else 0.0

            # Determine origin type based on thresholds
            if avg_density > 0.01:
                origin_type = "native_digital"
            elif avg_density < 0.001:
                origin_type = "scanned_image"
            else:
                origin_type = "mixed"

            # Determine recommended strategy
            if origin_type == "scanned_image":
                recommended_strategy = "vision_augmented"
            elif avg_density > 0.01 and avg_image_ratio < 0.5:
                recommended_strategy = "fast_text"
            else:
                recommended_strategy = "layout_aware"

            results["summary"] = {
                "total_pages": total_pages,
                "total_chars": total_chars,
                "avg_chars_per_page": round(avg_chars_per_page, 2),
                "avg_density": round(avg_density, 6),
                "pages_with_fonts": pages_with_fonts,
                "font_ratio": round(pages_with_fonts / total_pages, 4) if total_pages > 0 else 0.0,
                "avg_image_ratio": round(avg_image_ratio, 4),
                "origin_type": origin_type,
                "recommended_strategy": recommended_strategy,
                "confidence": _calculate_confidence(avg_density, pages_with_fonts / total_pages if total_pages > 0 else 0.0, avg_image_ratio),
            }

    except Exception as e:
        logger.error(f"Error analyzing PDF {pdf_path}: {e}")
        raise

    return results


def _calculate_confidence(
    density: float, font_ratio: float, image_ratio: float
) -> float:
    """Calculate confidence score for extraction strategy.

    Args:
        density: Character density (chars/point²).
        font_ratio: Ratio of pages with font metadata.
        image_ratio: Ratio of page area covered by images.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    # Character density score (normalized to 0-1)
    density_score = min(1.0, density / 0.01) if density > 0 else 0.0

    # Font metadata score
    font_score = font_ratio

    # Image area score (lower is better for text extraction)
    image_score = 1.0 if image_ratio < 0.5 else max(0.0, 1.0 - (image_ratio - 0.5) * 2)

    # Weighted combination
    confidence = 0.4 * density_score + 0.3 * font_score + 0.3 * image_score

    return round(confidence, 4)


def save_analysis_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save analysis results to JSON file.

    Args:
        results: Analysis results dictionary.
        output_path: Path to save JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Analysis results saved to {output_path}")


def print_analysis_summary(results: Dict[str, Any]) -> None:
    """Print a human-readable summary of analysis results.

    Args:
        results: Analysis results dictionary.
    """
    summary = results["summary"]
    print(f"\n{'='*60}")
    print(f"Character Density Analysis: {Path(results['file']).name}")
    print(f"{'='*60}")
    print(f"Total Pages: {summary['total_pages']}")
    print(f"Total Characters: {summary['total_chars']:,}")
    print(f"Average Chars per Page: {summary['avg_chars_per_page']:.2f}")
    print(f"Average Density: {summary['avg_density']:.6f} chars/point²")
    print(f"Font Metadata: {summary['pages_with_fonts']}/{summary['total_pages']} pages ({summary['font_ratio']*100:.1f}%)")
    print(f"Image Area Ratio: {summary['avg_image_ratio']*100:.2f}%")
    print(f"\nClassification:")
    print(f"  Origin Type: {summary['origin_type']}")
    print(f"  Recommended Strategy: {summary['recommended_strategy']}")
    print(f"  Confidence: {summary['confidence']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    """CLI interface for character density analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze character density of a PDF")
    parser.add_argument("pdf_path", type=Path, help="Path to PDF file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save JSON results (optional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-page information",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run analysis
    results = analyze_character_density(args.pdf_path)

    # Print summary
    print_analysis_summary(results)

    # Show per-page details if verbose
    if args.verbose:
        print("\nPer-Page Details:")
        print("-" * 60)
        for page in results["pages"]:
            print(
                f"Page {page['page']}: {page['char_count']:,} chars, "
                f"density={page['density']:.6f}, "
                f"fonts={page['has_fonts']}, "
                f"image_ratio={page['image_ratio']*100:.2f}%"
            )

    # Save results if output path provided
    if args.output:
        save_analysis_results(results, args.output)
