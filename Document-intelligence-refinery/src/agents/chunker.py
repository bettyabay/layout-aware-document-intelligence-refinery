"""Semantic Chunking Engine - Stage 3: LDU generation with 5 rules."""

import re
from typing import Optional

from src.models import (
    BBox,
    ExtractedDocument,
    LDU,
    ProvenanceChain,
    content_hash_for_text,
    estimate_token_count,
)


class ChunkingEngine:
    """Semantic chunking engine with 5 core rules."""

    def __init__(self, rules: dict):
        self.rules = rules
        chunking_cfg = rules.get("chunking", {})
        self.max_tokens = int(chunking_cfg.get("max_tokens_per_chunk", 512))
        self.min_tokens = int(chunking_cfg.get("min_tokens_per_chunk", 50))
        self.split_strategy = chunking_cfg.get("split_strategy", "semantic")
        self.preserve_lists = chunking_cfg.get("preserve_lists", True)
        self.assign_parent_sections = chunking_cfg.get("assign_parent_sections", True)

    def build(self, extracted: ExtractedDocument) -> list[LDU]:
        """Build LDUs from extracted document."""
        ldus: list[LDU] = []
        current_section: Optional[str] = None
        ldu_counter = 0

        for page in extracted.pages:
            page_ldu_ids: list[str] = []

            # Rule 1: Tables - keep as single LDUs
            for table in page.tables:
                ldu_counter += 1
                ldu_id = f"ldu-{extracted.doc_id}-{ldu_counter}"
                
                # Convert table to text representation
                table_text = self._table_to_text(table)
                content_hash = content_hash_for_text(table_text)
                token_count = estimate_token_count(table_text)

                # Create provenance chain
                provenance = ProvenanceChain(
                    document_name=extracted.document_name,
                    page_number=page.page_number,
                    bbox=table.bbox,
                    content_hash=content_hash,
                )

                ldu = LDU(
                    id=ldu_id,
                    text=table_text,
                    content_hash=content_hash,
                    chunk_type="table",
                    bounding_box=table.bbox,
                    token_count=token_count,
                    parent_section=current_section,
                    page_refs=[page.page_number],
                    provenance_chain=[provenance],
                )

                ldus.append(ldu)
                page_ldu_ids.append(ldu_id)

            # Rule 2: Figures with captions
            for figure in page.figures:
                ldu_counter += 1
                ldu_id = f"ldu-{extracted.doc_id}-{ldu_counter}"
                
                figure_text = f"[Figure: {figure.caption or 'Untitled'}]"
                content_hash = content_hash_for_text(figure_text)
                token_count = estimate_token_count(figure_text)

                provenance = ProvenanceChain(
                    document_name=extracted.document_name,
                    page_number=page.page_number,
                    bbox=figure.bbox,
                    content_hash=content_hash,
                )

                ldu = LDU(
                    id=ldu_id,
                    text=figure_text,
                    content_hash=content_hash,
                    chunk_type="figure",
                    bounding_box=figure.bbox,
                    token_count=token_count,
                    parent_section=current_section,
                    page_refs=[page.page_number],
                    provenance_chain=[provenance],
                )

                ldus.append(ldu)
                page_ldu_ids.append(ldu_id)

            # Process text blocks
            current_chunk_text = ""
            current_chunk_bbox: Optional[BBox] = None
            current_chunk_tokens = 0
            current_chunk_blocks: list = []

            for block in page.text_blocks:
                block_text = block.text.strip()
                if not block_text:
                    continue

                # Rule 4: Detect section headers
                if self._is_section_header(block_text):
                    # Save current chunk if exists
                    if current_chunk_text:
                        ldu_counter += 1
                        ldu_id = f"ldu-{extracted.doc_id}-{ldu_counter}"
                        ldu = self._create_text_ldu(
                            ldu_id,
                            current_chunk_text,
                            current_chunk_bbox,
                            current_chunk_tokens,
                            current_chunk_blocks,
                            page.page_number,
                            extracted.document_name,
                            current_section,
                        )
                        ldus.append(ldu)
                        page_ldu_ids.append(ldu_id)
                        current_chunk_text = ""
                        current_chunk_bbox = None
                        current_chunk_tokens = 0
                        current_chunk_blocks = []

                    # Update current section
                    current_section = block_text

                # Rule 3: Detect lists
                is_list = self._is_list(block_text)
                if is_list and self.preserve_lists:
                    # Save current chunk
                    if current_chunk_text:
                        ldu_counter += 1
                        ldu_id = f"ldu-{extracted.doc_id}-{ldu_counter}"
                        ldu = self._create_text_ldu(
                            ldu_id,
                            current_chunk_text,
                            current_chunk_bbox,
                            current_chunk_tokens,
                            current_chunk_blocks,
                            page.page_number,
                            extracted.document_name,
                            current_section,
                        )
                        ldus.append(ldu)
                        page_ldu_ids.append(ldu_id)
                        current_chunk_text = ""
                        current_chunk_bbox = None
                        current_chunk_tokens = 0
                        current_chunk_blocks = []

                    # Create list LDU
                    ldu_counter += 1
                    ldu_id = f"ldu-{extracted.doc_id}-{ldu_counter}"
                    list_text = block_text
                    content_hash = content_hash_for_text(list_text)
                    token_count = estimate_token_count(list_text)

                    provenance = ProvenanceChain(
                        document_name=extracted.document_name,
                        page_number=page.page_number,
                        bbox=block.bbox,
                        content_hash=content_hash,
                    )

                    ldu = LDU(
                        id=ldu_id,
                        text=list_text,
                        content_hash=content_hash,
                        chunk_type="list",
                        bounding_box=block.bbox,
                        token_count=token_count,
                        parent_section=current_section if self.assign_parent_sections else None,
                        page_refs=[page.page_number],
                        provenance_chain=[provenance],
                    )

                    ldus.append(ldu)
                    page_ldu_ids.append(ldu_id)
                    continue

                # Add to current chunk
                block_tokens = estimate_token_count(block_text)
                
                # Check if adding this block would exceed max tokens
                if current_chunk_tokens + block_tokens > self.max_tokens and current_chunk_text:
                    # Split chunk
                    ldu_counter += 1
                    ldu_id = f"ldu-{extracted.doc_id}-{ldu_counter}"
                    ldu = self._create_text_ldu(
                        ldu_id,
                        current_chunk_text,
                        current_chunk_bbox,
                        current_chunk_tokens,
                        current_chunk_blocks,
                        page.page_number,
                        extracted.document_name,
                        current_section,
                    )
                    ldus.append(ldu)
                    page_ldu_ids.append(ldu_id)
                    current_chunk_text = block_text
                    current_chunk_bbox = block.bbox
                    current_chunk_tokens = block_tokens
                    current_chunk_blocks = [block]
                else:
                    # Append to current chunk
                    if current_chunk_text:
                        current_chunk_text += " " + block_text
                    else:
                        current_chunk_text = block_text
                    
                    if current_chunk_bbox:
                        # Merge bboxes
                        current_chunk_bbox = BBox(
                            x0=min(current_chunk_bbox.x0, block.bbox.x0),
                            y0=min(current_chunk_bbox.y0, block.bbox.y0),
                            x1=max(current_chunk_bbox.x1, block.bbox.x1),
                            y1=max(current_chunk_bbox.y1, block.bbox.y1),
                        )
                    else:
                        current_chunk_bbox = block.bbox
                    
                    current_chunk_tokens += block_tokens
                    current_chunk_blocks.append(block)

            # Save last chunk
            if current_chunk_text:
                ldu_counter += 1
                ldu_id = f"ldu-{extracted.doc_id}-{ldu_counter}"
                ldu = self._create_text_ldu(
                    ldu_id,
                    current_chunk_text,
                    current_chunk_bbox,
                    current_chunk_tokens,
                    current_chunk_blocks,
                    page.page_number,
                    extracted.document_name,
                    current_section,
                )
                ldus.append(ldu)
                page_ldu_ids.append(ldu_id)

            # Update page LDU IDs
            page.ldu_ids = page_ldu_ids

        # Rule 5: Cross-reference resolution
        if self.rules.get("chunking", {}).get("resolve_cross_references", True):
            self._resolve_cross_references(ldus)

        # Link adjacent chunks
        for i in range(len(ldus) - 1):
            ldus[i].next_chunk_id = ldus[i + 1].id
            ldus[i + 1].previous_chunk_id = ldus[i].id

        return ldus

    def _table_to_text(self, table) -> str:
        """Convert table to text representation."""
        lines = []
        if table.headers:
            lines.append(" | ".join(table.headers))
            lines.append(" | ".join(["---"] * len(table.headers)))
        for row in table.rows:
            lines.append(" | ".join(str(cell) for cell in row))
        return "\n".join(lines)

    def _is_section_header(self, text: str) -> bool:
        """Detect if text is a section header."""
        text = text.strip()
        if not text or len(text) > 200:
            return False
        
        text_upper = text.upper()
        words = text.split()
        
        # Pattern 1: Numbered sections (e.g., "1. INTRODUCTION", "2.3 Methodology")
        if re.match(r"^\d+[\.\)]\s+[A-Z]", text):
            return True
        
        # Pattern 2: All caps short text (likely headers)
        if len(words) <= 8 and text_upper == text and len(text) > 5:
            return True
        
        # Pattern 3: Common section keywords
        section_keywords = [
            "CHAPTER", "SECTION", "PART", "NOTICE", "AGENDA",
            "CONTENTS", "TABLE OF CONTENTS", "INTRODUCTION",
            "SUMMARY", "CONCLUSION", "APPENDIX", "REFERENCES"
        ]
        for keyword in section_keywords:
            if text_upper.startswith(keyword):
                return True
        
        # Pattern 4: Roman numerals followed by text (e.g., "I. Introduction")
        if re.match(r"^[IVX]+\.\s+[A-Z]", text):
            return True
        
        # Pattern 5: Short text that's mostly uppercase (80%+)
        if len(text) > 5 and len(text) < 100:
            upper_count = sum(1 for c in text if c.isupper())
            if upper_count / len(text) > 0.7:
                return True
        
        return False

    def _is_list(self, text: str) -> bool:
        """Detect if text is a list."""
        lines = text.split("\n")
        if len(lines) < 2:
            return False
        
        # Check for numbered or bulleted lists
        list_patterns = [
            r"^\d+[\.\)]\s+",  # Numbered: 1. or 1)
            r"^[-*•]\s+",  # Bulleted
            r"^[a-z][\.\)]\s+",  # Lettered: a. or a)
        ]
        
        list_count = 0
        for line in lines[:5]:  # Check first 5 lines
            for pattern in list_patterns:
                if re.match(pattern, line.strip()):
                    list_count += 1
                    break
        
        return list_count >= 2  # At least 2 list items

    def _create_text_ldu(
        self,
        ldu_id: str,
        text: str,
        bbox: Optional[BBox],
        token_count: int,
        blocks: list,
        page_num: int,
        doc_name: str,
        parent_section: Optional[str],
    ) -> LDU:
        """Create a text LDU."""
        content_hash = content_hash_for_text(text)
        
        # Create provenance chain from blocks
        provenance_chain = []
        if blocks:
            for block in blocks:
                provenance_chain.append(
                    ProvenanceChain(
                        document_name=doc_name,
                        page_number=page_num,
                        bbox=block.bbox,
                        content_hash=content_hash_for_text(block.text),
                    )
                )

        return LDU(
            id=ldu_id,
            text=text,
            content_hash=content_hash,
            chunk_type="paragraph",
            bounding_box=bbox,
            token_count=token_count,
            parent_section=parent_section if self.assign_parent_sections else None,
            page_refs=[page_num],
            provenance_chain=provenance_chain,
        )

    def _resolve_cross_references(self, ldus: list[LDU]) -> None:
        """Resolve cross-references (Rule 5)."""
        # Simple implementation: find references like "Table 3", "Figure 2", etc.
        reference_patterns = [
            (r"Table\s+(\d+)", "table"),
            (r"Figure\s+(\d+)", "figure"),
            (r"Section\s+(\d+)", "section"),
        ]

        for ldu in ldus:
            for pattern, ref_type in reference_patterns:
                matches = re.finditer(pattern, ldu.text, re.IGNORECASE)
                for match in matches:
                    ref_num = match.group(1)
                    # Find referenced chunk
                    for other_ldu in ldus:
                        if (
                            other_ldu.chunk_type == ref_type
                            and ref_num in other_ldu.text
                            and other_ldu.id not in ldu.reference_ids
                        ):
                            ldu.reference_ids.append(other_ldu.id)
                            break
