"""PageIndex Builder - Stage 4: Hierarchical navigation structure."""

import re
from typing import Optional

from src.models import ExtractedDocument, PageIndexNode


class PageIndexBuilder:
    """Build hierarchical PageIndex tree from extracted document."""

    def __init__(self, rules: dict | None = None):
        self.rules = rules or {}

    def build(self, extracted: ExtractedDocument) -> PageIndexNode:
        """Build PageIndex tree with proper hierarchy."""
        root = PageIndexNode(
            id=f"root-{extracted.doc_id}",
            node_type="document",
            label=extracted.document_name,
            page_number=None,
            bbox=None,
            summary=None,
            children=[],
        )

        # Build section hierarchy from LDUs
        sections: dict[str, PageIndexNode] = {}  # section_name -> node
        current_section: Optional[PageIndexNode] = None
        current_subsection: Optional[PageIndexNode] = None
        
        # Track pages per section
        section_pages: dict[str, set[int]] = {}  # section_name -> set of page numbers

        # First pass: identify sections from LDUs
        for ldu in extracted.ldus:
            if ldu.parent_section:
                section_name = ldu.parent_section
                if section_name not in sections:
                    # Create new section node
                    section_node = PageIndexNode(
                        id=f"section-{len(sections)}",
                        node_type="section",
                        label=section_name,
                        page_number=ldu.page_refs[0] if ldu.page_refs else None,
                        bbox=ldu.bounding_box,
                        summary=self._generate_section_summary(ldu, extracted),
                        children=[],
                    )
                    sections[section_name] = section_node
                    section_pages[section_name] = set()
                
                # Track pages for this section
                if ldu.page_refs:
                    section_pages[section_name].update(ldu.page_refs)

        # Second pass: organize pages into sections
        if sections:
            # Group pages by section
            for section_name, section_node in sections.items():
                pages_in_section = sorted(section_pages[section_name])
                
                # Create page nodes for this section
                for page_num in pages_in_section:
                    page = next((p for p in extracted.pages if p.page_number == page_num), None)
                    if page:
                        page_node = PageIndexNode(
                            id=f"page-{page_num}",
                            node_type="page",
                            label=f"Page {page_num}",
                            page_number=page_num,
                            bbox=None,
                            summary=self._generate_summary(page),
                            children=[],
                        )
                        section_node.children.append(page_node)
                
                root.children.append(section_node)
        else:
            # No sections found - detect sections from text blocks
            sections = self._detect_sections_from_text(extracted)
            if sections:
                for section_node in sections.values():
                    root.children.append(section_node)
            else:
                # Fallback: add pages directly
                for page in extracted.pages[:20]:  # Limit to first 20 pages for performance
                    page_node = PageIndexNode(
                        id=f"page-{page.page_number}",
                        node_type="page",
                        label=f"Page {page.page_number}",
                        page_number=page.page_number,
                        bbox=None,
                        summary=self._generate_summary(page),
                        children=[],
                    )
                    root.children.append(page_node)

        # Add metadata
        root.summary = f"Document: {extracted.document_name} with {len(extracted.pages)} pages"

        return root

    def _detect_sections_from_text(self, extracted: ExtractedDocument) -> dict[str, PageIndexNode]:
        """Detect sections from text blocks (fallback method)."""
        sections: dict[str, PageIndexNode] = {}
        section_patterns = [
            r"^(\d+\.?\s+[A-Z][A-Z\s]{5,})",  # Numbered sections: "1. INTRODUCTION"
            r"^(CHAPTER\s+\d+)",  # Chapters
            r"^(SECTION\s+\d+)",  # Sections
            r"^([A-Z][A-Z\s]{10,})$",  # All caps headers
        ]
        
        current_section = None
        current_section_pages = set()
        
        for page in extracted.pages:
            for block in page.text_blocks[:3]:  # Check first 3 blocks per page
                text = block.text.strip()
                if len(text) < 5 or len(text) > 100:
                    continue
                
                # Check if this looks like a section header
                is_header = False
                for pattern in section_patterns:
                    match = re.match(pattern, text)
                    if match:
                        section_title = match.group(1).strip()
                        if section_title not in sections:
                            section_node = PageIndexNode(
                                id=f"section-{len(sections)}",
                                node_type="section",
                                label=section_title,
                                page_number=page.page_number,
                                bbox=block.bbox,
                                summary=text[:200],
                                children=[],
                            )
                            sections[section_title] = section_node
                            current_section = section_title
                            current_section_pages = {page.page_number}
                        is_header = True
                        break
                
                if is_header:
                    break
            
            # Add page to current section
            if current_section and page.page_number not in current_section_pages:
                page_node = PageIndexNode(
                    id=f"page-{page.page_number}",
                    node_type="page",
                    label=f"Page {page.page_number}",
                    page_number=page.page_number,
                    bbox=None,
                    summary=self._generate_summary(page),
                    children=[],
                )
                sections[current_section].children.append(page_node)
                current_section_pages.add(page.page_number)
        
        return sections

    def _generate_section_summary(self, ldu, extracted: ExtractedDocument) -> str:
        """Generate summary for a section based on its LDUs."""
        # Find all LDUs in this section
        section_ldus = [l for l in extracted.ldus if l.parent_section == ldu.parent_section]
        if section_ldus:
            # Use first few LDUs for summary
            summary_text = " ".join([l.text[:100] for l in section_ldus[:3]])
            if len(summary_text) > 300:
                summary_text = summary_text[:300] + "..."
            return summary_text
        return ldu.text[:200] if ldu.text else ""

    def _generate_summary(self, page) -> str:
        """Generate summary for page (simplified)."""
        # In production, use LLM for summarization
        text_blocks = page.text_blocks[:5]  # First 5 blocks
        summary_text = " ".join([b.text[:100] for b in text_blocks])
        if len(summary_text) > 200:
            summary_text = summary_text[:200] + "..."
        return summary_text or f"Page {page.page_number} content"
