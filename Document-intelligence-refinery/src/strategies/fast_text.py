"""Strategy A: Fast Text Extraction using pdfplumber."""

from pathlib import Path

import pdfplumber

from src.models import (
    BBox,
    DocumentProfile,
    ExtractedDocument,
    ExtractedMetadata,
    ExtractedPage,
    StrategyName,
    TextBlock,
    content_hash_for_text,
)
from src.strategies.base import ExtractionStrategy


class FastTextExtractor(ExtractionStrategy):
    """Fast text extraction using pdfplumber."""

    name = "fast_text"

    def extract(
        self, pdf_path: Path, profile: DocumentProfile, rules: dict
    ) -> tuple[ExtractedDocument, float, float]:
        """Extract text using pdfplumber."""
        pages: list[ExtractedPage] = []
        total_chars = 0

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text blocks
                words = page.extract_words() or []
                text_blocks: list[TextBlock] = []
                
                # Group words into blocks (simplified)
                current_block_words = []
                current_block_bbox = None
                
                for word in words:
                    word_text = word.get("text", "")
                    word_bbox = BBox(
                        x0=float(word.get("x0", 0)),
                        y0=float(word.get("top", 0)),
                        x1=float(word.get("x1", 0)),
                        y1=float(word.get("bottom", 0)),
                    )
                    
                    if not current_block_words:
                        current_block_words = [word_text]
                        current_block_bbox = word_bbox
                    else:
                        # Simple grouping: if words are close, add to same block
                        if word_bbox.x0 - current_block_bbox.x1 < 10:  # Close horizontally
                            current_block_words.append(word_text)
                            current_block_bbox.x1 = word_bbox.x1
                            current_block_bbox.y1 = max(current_block_bbox.y1, word_bbox.y1)
                        else:
                            # Create block
                            block_text = " ".join(current_block_words)
                            text_blocks.append(
                                TextBlock(
                                    id=f"p{page_num}-b{len(text_blocks)}",
                                    text=block_text,
                                    bbox=current_block_bbox,
                                    reading_order=len(text_blocks),
                                )
                            )
                            total_chars += len(block_text)
                            current_block_words = [word_text]
                            current_block_bbox = word_bbox
                
                # Add last block
                if current_block_words:
                    block_text = " ".join(current_block_words)
                    text_blocks.append(
                        TextBlock(
                            id=f"p{page_num}-b{len(text_blocks)}",
                            text=block_text,
                            bbox=current_block_bbox,
                            reading_order=len(text_blocks),
                        )
                    )
                    total_chars += len(block_text)

                pages.append(
                    ExtractedPage(
                        page_number=page_num,
                        width=float(page.width),
                        height=float(page.height),
                        text_blocks=text_blocks,
                        tables=[],
                        figures=[],
                        ldu_ids=[],
                    )
                )

        # Calculate confidence
        confidence = self._calculate_confidence(total_chars, len(pages), profile)

        extracted = ExtractedDocument(
            doc_id=profile.doc_id,
            document_name=profile.document_name,
            pages=pages,
            metadata=ExtractedMetadata(
                source_strategy=StrategyName.A,
                confidence_score=confidence,
                strategy_sequence=[StrategyName.A],
            ),
            ldus=[],
            page_index=None,
            provenance_chains=[],
        )

        return extracted, confidence, 0.0  # Free

    def _calculate_confidence(
        self, total_chars: int, num_pages: int, profile: DocumentProfile
    ) -> float:
        """Calculate confidence score for fast text extraction."""
        chars_per_page = total_chars / num_pages if num_pages > 0 else 0
        
        # High confidence if good character density
        if chars_per_page > 500:
            return 0.9
        elif chars_per_page > 200:
            return 0.7
        elif chars_per_page > 50:
            return 0.5
        else:
            return 0.3
