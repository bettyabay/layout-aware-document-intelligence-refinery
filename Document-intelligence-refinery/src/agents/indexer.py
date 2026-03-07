"""PageIndex Builder - Stage 4: Hierarchical navigation structure."""

from src.models import ExtractedDocument, PageIndexNode


class PageIndexBuilder:
    """Build hierarchical PageIndex tree from extracted document."""

    def __init__(self, rules: dict | None = None):
        self.rules = rules or {}

    def build(self, extracted: ExtractedDocument) -> PageIndexNode:
        """Build PageIndex tree."""
        root = PageIndexNode(
            id=f"root-{extracted.doc_id}",
            node_type="document",
            label=extracted.document_name,
            page_number=None,
            bbox=None,
            summary=None,
            children=[],
        )

        current_section: PageIndexNode | None = None
        current_subsection: PageIndexNode | None = None

        for page in extracted.pages:
            # Create page node
            page_node = PageIndexNode(
                id=f"page-{page.page_number}",
                node_type="page",
                label=f"Page {page.page_number}",
                page_number=page.page_number,
                bbox=None,
                summary=self._generate_summary(page),
                children=[],
            )

            # Group by sections if available
            for ldu_id in page.ldu_ids:
                # Find LDU
                ldu = next((l for l in extracted.ldus if l.id == ldu_id), None)
                if not ldu:
                    continue

                # Check if LDU is a section header
                if ldu.parent_section and ldu.parent_section != current_section:
                    # New section
                    current_section = PageIndexNode(
                        id=f"section-{ldu.parent_section}",
                        node_type="section",
                        label=ldu.parent_section,
                        page_number=page.page_number,
                        bbox=ldu.bounding_box,
                        summary=None,
                        children=[],
                    )
                    root.children.append(current_section)
                    current_subsection = None

                # Add to appropriate parent
                if current_section:
                    current_section.children.append(page_node)
                else:
                    root.children.append(page_node)
                    break  # Only add page once

        # If no sections found, add pages directly
        if not root.children:
            for page in extracted.pages:
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

        return root

    def _generate_summary(self, page) -> str:
        """Generate summary for page (simplified)."""
        # In production, use LLM for summarization
        text_blocks = page.text_blocks[:5]  # First 5 blocks
        summary_text = " ".join([b.text[:100] for b in text_blocks])
        if len(summary_text) > 200:
            summary_text = summary_text[:200] + "..."
        return summary_text or f"Page {page.page_number} content"
