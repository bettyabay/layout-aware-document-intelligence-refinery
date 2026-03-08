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
    TableObject,
    TextBlock,
    content_hash_for_text,
)
from src.strategies.base import ExtractionStrategy


def normalize_bbox(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
    """Normalize bounding box coordinates to ensure x1>=x0 and y1>=y0."""
    # Ensure x1 >= x0
    if x1 < x0:
        x0, x1 = x1, x0
    # Ensure y1 >= y0
    if y1 < y0:
        y0, y1 = y1, y0
    # Ensure minimum size
    if x1 == x0:
        x1 = x0 + 1.0
    if y1 == y0:
        y1 = y0 + 1.0
    return x0, y0, x1, y1


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
                text_blocks: list[TextBlock] = []
                tables: list = []
                
                # Extract tables first (before text to avoid overlap)
                page_tables = page.extract_tables()
                for table_idx, table_data in enumerate(page_tables or []):
                    if not table_data or len(table_data) < 2:
                        continue
                    
                    # Get table bounding box (approximate from first/last cells)
                    # For now, create a placeholder bbox
                    table_bbox = BBox(x0=0, y0=0, x1=page.width, y1=page.height)
                    
                    # Convert to TableObject
                    headers = table_data[0] if table_data else []
                    rows = table_data[1:] if len(table_data) > 1 else []
                    
                    table_obj = TableObject(
                        id=f"p{page_num}-t{table_idx}",
                        headers=[str(h) if h else "" for h in headers],
                        rows=[[str(cell) if cell else "" for cell in row] for row in rows],
                        bbox=table_bbox,
                        page_number=page_num,
                    )
                    tables.append(table_obj)
                
                # Extract text using pdfplumber - use multiple methods for reliability
                # Method 1: Try extract_text() first (simplest and most reliable)
                full_text = page.extract_text()
                if full_text and full_text.strip():
                    # Split into paragraphs/lines
                    lines = [line.strip() for line in full_text.split("\n") if line.strip()]
                    
                    # Group lines into blocks (paragraphs)
                    current_block = []
                    for line in lines:
                        if line:
                            current_block.append(line)
                            # If line ends with period or is short, it might be end of paragraph
                            if len(current_block) >= 3 or (line.endswith('.') and len(current_block) >= 1):
                                block_text = " ".join(current_block)
                                if block_text.strip():
                                    # Get approximate bbox - use page dimensions
                                    para_bbox = BBox(x0=0, y0=0, x1=page.width, y1=page.height)
                                    text_blocks.append(
                                        TextBlock(
                                            id=f"p{page_num}-b{len(text_blocks)}",
                                            text=block_text,
                                            bbox=para_bbox,
                                            reading_order=len(text_blocks),
                                        )
                                    )
                                    total_chars += len(block_text)
                                current_block = []
                    
                    # Add remaining block
                    if current_block:
                        block_text = " ".join(current_block)
                        if block_text.strip():
                            para_bbox = BBox(x0=0, y0=0, x1=page.width, y1=page.height)
                            text_blocks.append(
                                TextBlock(
                                    id=f"p{page_num}-b{len(text_blocks)}",
                                    text=block_text,
                                    bbox=para_bbox,
                                    reading_order=len(text_blocks),
                                )
                            )
                            total_chars += len(block_text)
                
                # Method 2: If extract_text() didn't work, try extract_words() with better grouping
                if not text_blocks:
                    words = page.extract_words()
                    if words:
                        # Group words by proximity into lines, then into blocks
                        sorted_words = sorted(words, key=lambda w: (w.get("top", 0), w.get("x0", 0)))
                        
                        current_line_words = []
                        current_y = None
                        current_block_lines = []
                        
                        for word in sorted_words:
                            word_text = word.get("text", "").strip()
                            if not word_text:
                                continue
                            
                            word_y = word.get("top", 0)
                            
                            # Group into lines (same y-coordinate within threshold)
                            if current_y is None or abs(word_y - current_y) < 5:
                                current_line_words.append(word_text)
                                if current_y is None:
                                    current_y = word_y
                            else:
                                # New line
                                if current_line_words:
                                    current_block_lines.append(" ".join(current_line_words))
                                
                                # If y-coordinate changed significantly, might be new paragraph
                                if current_y and abs(word_y - current_y) > 20:
                                    # Create block from accumulated lines
                                    if current_block_lines:
                                        block_text = " ".join(current_block_lines)
                                        if block_text.strip():
                                            block_bbox = BBox(x0=0, y0=0, x1=page.width, y1=page.height)
                                            text_blocks.append(
                                                TextBlock(
                                                    id=f"p{page_num}-b{len(text_blocks)}",
                                                    text=block_text,
                                                    bbox=block_bbox,
                                                    reading_order=len(text_blocks),
                                                )
                                            )
                                            total_chars += len(block_text)
                                        current_block_lines = []
                                
                                current_line_words = [word_text]
                                current_y = word_y
                        
                        # Add last line
                        if current_line_words:
                            current_block_lines.append(" ".join(current_line_words))
                        
                        # Add last block
                        if current_block_lines:
                            block_text = " ".join(current_block_lines)
                            if block_text.strip():
                                block_bbox = BBox(x0=0, y0=0, x1=page.width, y1=page.height)
                                text_blocks.append(
                                    TextBlock(
                                        id=f"p{page_num}-b{len(text_blocks)}",
                                        text=block_text,
                                        bbox=block_bbox,
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
                        tables=tables,
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
