"""Section Chunker for Rule 4: Section Header Inheritance.

This module implements specialized section chunking logic to ensure that section
headers are stored as parent metadata on all child chunks within that section,
with support for nested sections and full path inheritance.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from src.models.ldu import LDU

logger = logging.getLogger(__name__)


class SectionNode:
    """Node in a section hierarchy tree.

    Attributes:
        title: Section title/header text.
        level: Section level (1 = top-level, 2 = subsection, etc.).
        page_start: Starting page number.
        page_end: Ending page number (updated as we process).
        parent: Parent section node.
        children: List of child section nodes.
        chunk_ids: List of chunk content hashes that belong to this section.
        full_path: Full section path (e.g., "1. Introduction > 1.1 Overview").
    """

    def __init__(
        self,
        title: str,
        level: int,
        page_start: int,
        parent: Optional["SectionNode"] = None,
    ):
        """Initialize a section node.

        Args:
            title: Section title.
            level: Section level (1-based).
            page_start: Starting page number.
            parent: Parent section node.
        """
        self.title = title
        self.level = level
        self.page_start = page_start
        self.page_end = page_start
        self.parent = parent
        self.children: List["SectionNode"] = []
        self.chunk_ids: List[str] = []
        self.full_path = self._compute_full_path()

    def _compute_full_path(self) -> str:
        """Compute the full section path from root to this node.

        Returns:
            Full path string (e.g., "1. Introduction > 1.1 Overview").
        """
        if self.parent:
            return f"{self.parent.full_path} > {self.title}"
        return self.title

    def add_child(self, child: "SectionNode") -> None:
        """Add a child section node.

        Args:
            child: Child section node to add.
        """
        self.children.append(child)
        child.parent = self

    def update_page_end(self, page: int) -> None:
        """Update the ending page number for this section.

        Args:
            page: Page number to set as end (if greater than current).
        """
        self.page_end = max(self.page_end, page)
        if self.parent:
            self.parent.update_page_end(page)


class SectionChunker:
    """Specialized chunker for section hierarchy management.

    Builds a section tree from headers and assigns section context to all
    child chunks, with support for nested sections and full path inheritance.
    """

    def __init__(self):
        """Initialize the SectionChunker."""
        pass

    def build_section_hierarchy(self, chunks: List[LDU]) -> SectionNode:
        """Build a section hierarchy tree from header chunks.

        Args:
            chunks: List of all chunks, including headers.

        Returns:
            Root section node (with title "Document Root").
        """
        # Identify header chunks
        header_chunks = [c for c in chunks if c.chunk_type == "header"]

        if not header_chunks:
            # No headers found, return root node
            root = SectionNode("Document Root", 0, 1)
            return root

        # Sort headers by page and position
        sorted_headers = sorted(
            header_chunks,
            key=lambda c: (
                min(c.page_refs),
                c.bounding_box.get("y0", 0),
            ),
        )

        # Build hierarchy
        root = SectionNode("Document Root", 0, min(c.page_refs[0] for c in sorted_headers))
        stack: List[SectionNode] = [root]  # Stack to track current section path

        for header_chunk in sorted_headers:
            # Determine section level from header content
            level = self._detect_header_level(header_chunk.content, header_chunk)

            # Pop stack until we find the appropriate parent
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()

            parent = stack[-1]
            page_num = min(header_chunk.page_refs)

            # Create new section node
            section = SectionNode(
                title=header_chunk.content.strip(),
                level=level,
                page_start=page_num,
                parent=parent,
            )

            parent.add_child(section)
            stack.append(section)

            # Update page ranges
            parent.update_page_end(page_num)

        logger.info(
            f"Built section hierarchy with {len(root.children)} top-level sections"
        )
        return root

    def assign_section_to_chunks(
        self, chunks: List[LDU], section_tree: SectionNode
    ) -> List[LDU]:
        """Assign section context to all chunks based on the section hierarchy.

        Args:
            chunks: List of chunks to enrich.
            section_tree: Root section node of the hierarchy.

        Returns:
            Updated chunks with parent_section and section_path populated.
        """
        # Build a map of page -> sections active on that page
        page_sections: Dict[int, List[SectionNode]] = {}
        self._build_page_section_map(section_tree, page_sections)

        # Sort chunks by page and position
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (
                min(c.page_refs),
                c.bounding_box.get("y0", 0),
            ),
        )

        # Track current section for each page
        current_sections: Dict[int, SectionNode] = {}

        for chunk in sorted_chunks:
            # Skip header chunks themselves (they don't have parent sections)
            if chunk.chunk_type == "header":
                chunk.parent_section = None
                continue

            # Find the appropriate section for this chunk
            page_num = min(chunk.page_refs)
            section = self._find_section_for_chunk(
                chunk, page_num, page_sections, current_sections
            )

            if section:
                # Assign section information
                chunk.parent_section = section.title
                chunk.metadata["section_path"] = section.full_path
                chunk.metadata["section_level"] = section.level
                chunk.metadata["section_page_start"] = section.page_start
                chunk.metadata["section_page_end"] = section.page_end

                # Update current section for this page
                current_sections[page_num] = section
            else:
                # No section found, use root
                chunk.parent_section = None
                chunk.metadata["section_path"] = "Document Root"

        logger.info(
            f"Assigned sections to {len([c for c in chunks if c.parent_section])} chunks"
        )

        return sorted_chunks

    def _detect_header_level(self, content: str, chunk: LDU) -> int:
        """Detect the level of a header.

        Uses multiple heuristics:
        1. Numbering patterns (1., 1.1., 1.1.1., etc.)
        2. Markdown-style headers (#, ##, ###)
        3. Font size (if available in metadata)
        4. Position and formatting

        Args:
            content: Header content.
            chunk: Header chunk (may contain metadata).

        Returns:
            Header level (1 = top-level, 2 = subsection, etc.).
        """
        content_stripped = content.strip()

        # Check for numbered sections (1., 1.1., 1.1.1., etc.)
        numbered_pattern = r"^(\d+(?:\.\d+)*)[.)]\s+"
        match = re.match(numbered_pattern, content_stripped)
        if match:
            number_parts = match.group(1).split(".")
            return len(number_parts)

        # Check for markdown-style headers
        if content_stripped.startswith("#"):
            level = len(content_stripped) - len(content_stripped.lstrip("#"))
            return min(level, 6)  # Cap at level 6

        # Check for Roman numerals or letters (I., II., A., B., etc.)
        roman_pattern = r"^([IVXLCDM]+|([A-Z]))[.)]\s+"
        match = re.match(roman_pattern, content_stripped, re.IGNORECASE)
        if match:
            # Top-level sections often use Roman numerals or single letters
            return 1

        # Check metadata for header level
        if "header_level" in chunk.metadata:
            return chunk.metadata["header_level"]

        # Default: assume it's a top-level header if it's short and all caps
        if len(content_stripped) < 100 and content_stripped.isupper():
            return 1

        # Default to level 2 (subsection)
        return 2

    def _build_page_section_map(
        self, section: SectionNode, page_sections: Dict[int, List[SectionNode]]
    ) -> None:
        """Build a map of page numbers to active sections.

        Recursively traverses the section tree and adds each section to
        all pages it spans.

        Args:
            section: Current section node.
            page_sections: Dictionary to populate.
        """
        # Add this section to all pages it spans
        for page in range(section.page_start, section.page_end + 1):
            if page not in page_sections:
                page_sections[page] = []
            page_sections[page].append(section)

        # Recursively process children
        for child in section.children:
            self._build_page_section_map(child, page_sections)

    def _find_section_for_chunk(
        self,
        chunk: LDU,
        page_num: int,
        page_sections: Dict[int, List[SectionNode]],
        current_sections: Dict[int, SectionNode],
    ) -> Optional[SectionNode]:
        """Find the most appropriate section for a chunk.

        Args:
            chunk: The chunk to find a section for.
            page_num: Page number of the chunk.
            page_sections: Map of page numbers to active sections.
            current_sections: Map of page numbers to current section.

        Returns:
            The most appropriate section node, or None.
        """
        # Get active sections for this page
        active_sections = page_sections.get(page_num, [])

        if not active_sections:
            # No sections on this page, check if we have a current section
            return current_sections.get(page_num)

        # Sort by level (deeper = more specific)
        active_sections.sort(key=lambda s: s.level, reverse=True)

        # Return the most specific (deepest) section
        return active_sections[0] if active_sections else None

    def find_chunks_in_section(
        self, chunks: List[LDU], section_path: str
    ) -> List[LDU]:
        """Find all chunks that belong to a specific section.

        Supports queries like "find all chunks in Section 3.2" by matching
        section paths.

        Args:
            chunks: List of chunks to search.
            section_path: Section path to match (can be partial).

        Returns:
            List of chunks in the specified section.
        """
        matching_chunks = []

        for chunk in chunks:
            chunk_path = chunk.metadata.get("section_path", "")
            if section_path in chunk_path or chunk_path.endswith(section_path):
                matching_chunks.append(chunk)

        return matching_chunks


def build_section_hierarchy(chunks: List[LDU]) -> SectionNode:
    """Build a section hierarchy from chunks (convenience function).

    Args:
        chunks: List of chunks including headers.

    Returns:
        Root section node.
    """
    chunker = SectionChunker()
    return chunker.build_section_hierarchy(chunks)
