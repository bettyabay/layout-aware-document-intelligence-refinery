"""Strategy B: Layout-Aware Extraction (simplified - uses pdfplumber with layout awareness)."""

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
)
from src.strategies.base import ExtractionStrategy


class LayoutExtractor(ExtractionStrategy):
    """Layout-aware extraction (simplified implementation)."""

    name = "layout_aware"

    def extract(
        self, pdf_path: Path, profile: DocumentProfile, rules: dict
    ) -> tuple[ExtractedDocument, float, float]:
        """Extract with layout awareness."""
        pages: list[ExtractedPage] = []
        total_chars = 0

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text blocks with better layout awareness
                words = page.extract_words() or []
                text_blocks: list[TextBlock] = []
                tables: list[TableObject] = []

                # Try to extract tables
                pdf_tables = page.find_tables()
                for table_idx, table in enumerate(pdf_tables):
                    try:
                        table_data = table.extract()
                        if table_data and len(table_data) > 1:
                            headers = table_data[0] if table_data else []
                            rows = table_data[1:] if len(table_data) > 1 else []
                            
                            # Get table bbox
                            bbox = table.bbox if hasattr(table, "bbox") else (0, 0, page.width, page.height)
                            
                            tables.append(
                                TableObject(
                                    id=f"p{page_num}-t{table_idx}",
                                    title=None,
                                    headers=[str(h) if h else "" for h in headers],
                                    rows=[[str(cell) if cell else "" for cell in row] for row in rows],
                                    bbox=BBox(
                                        x0=float(bbox[0]),
                                        y0=float(bbox[1]),
                                        x1=float(bbox[2]),
                                        y1=float(bbox[3]),
                                    ),
                                    reading_order=len(tables),
                                )
                            )
                    except Exception:
                        pass  # Skip malformed tables

                # Extract text blocks (group by proximity)
                if words:
                    # Sort by reading order (top to bottom, left to right)
                    sorted_words = sorted(words, key=lambda w: (w.get("top", 0), w.get("x0", 0)))
                    
                    current_line = []
                    current_y = None
                    
                    for word in sorted_words:
                        word_y = word.get("top", 0)
                        word_text = word.get("text", "")
                        
                        if current_y is None or abs(word_y - current_y) < 5:  # Same line
                            current_line.append((word, word_text))
                            current_y = word_y
                        else:
                            # New line - create block from previous line
                            if current_line:
                                block_text = " ".join([w[1] for w in current_line])
                                word_objs = [w[0] for w in current_line]
                                bbox = BBox(
                                    x0=float(min(w.get("x0", 0) for w in word_objs)),
                                    y0=float(min(w.get("top", 0) for w in word_objs)),
                                    x1=float(max(w.get("x1", 0) for w in word_objs)),
                                    y1=float(max(w.get("bottom", 0) for w in word_objs)),
                                )
                                text_blocks.append(
                                    TextBlock(
                                        id=f"p{page_num}-b{len(text_blocks)}",
                                        text=block_text,
                                        bbox=bbox,
                                        reading_order=len(text_blocks),
                                    )
                                )
                                total_chars += len(block_text)
                            
                            current_line = [(word, word_text)]
                            current_y = word_y
                    
                    # Add last line
                    if current_line:
                        block_text = " ".join([w[1] for w in current_line])
                        word_objs = [w[0] for w in current_line]
                        bbox = BBox(
                            x0=float(min(w.get("x0", 0) for w in word_objs)),
                            y0=float(min(w.get("top", 0) for w in word_objs)),
                            x1=float(max(w.get("x1", 0) for w in word_objs)),
                            y1=float(max(w.get("bottom", 0) for w in word_objs)),
                        )
                        text_blocks.append(
                            TextBlock(
                                id=f"p{page_num}-b{len(text_blocks)}",
                                text=block_text,
                                bbox=bbox,
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
        confidence = self._calculate_confidence(total_chars, len(pages), profile, len(tables))

        extracted = ExtractedDocument(
            doc_id=profile.doc_id,
            document_name=profile.document_name,
            pages=pages,
            metadata=ExtractedMetadata(
                source_strategy=StrategyName.B,
                confidence_score=confidence,
                strategy_sequence=[StrategyName.B],
            ),
            ldus=[],
            page_index=None,
            provenance_chains=[],
        )

        return extracted, confidence, 0.0  # Free (local processing)

    def _calculate_confidence(
        self, total_chars: int, num_pages: int, profile: DocumentProfile, num_tables: int
    ) -> float:
        """Calculate confidence score."""
        chars_per_page = total_chars / num_pages if num_pages > 0 else 0
        
        base_confidence = 0.6
        if chars_per_page > 500:
            base_confidence += 0.2
        elif chars_per_page > 200:
            base_confidence += 0.1
        
        # Bonus for table extraction
        if num_tables > 0:
            base_confidence += 0.1
        
        return min(0.95, base_confidence)
