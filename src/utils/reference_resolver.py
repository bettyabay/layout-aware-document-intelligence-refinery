"""Reference Resolver for Rule 5: Cross-Reference Resolution.

This module implements specialized cross-reference resolution logic to ensure
that references like "see Table 3", "as shown in Figure Y", and "refer to
Section Z" are resolved and stored as chunk relationships.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from src.models.ldu import CrossReference, LDU

logger = logging.getLogger(__name__)

# Reference patterns
REFERENCE_PATTERNS = {
    "table": [
        r"(?:see\s+)?(?:table|Table)\s+(\d+)",
        r"(?:refer\s+to\s+)?(?:table|Table)\s+(\d+)",
        r"(?:as\s+shown\s+in\s+)?(?:table|Table)\s+(\d+)",
        r"(?:see\s+)?(?:table|Table)\s+(\d+)\s+above",
        r"(?:see\s+)?(?:table|Table)\s+(\d+)\s+below",
        r"Tbl\.\s*(\d+)",
        r"Tbl\s+(\d+)",
    ],
    "figure": [
        r"(?:see\s+)?(?:figure|Figure|Fig\.?)\s+(\d+)",
        r"(?:as\s+shown\s+in\s+)?(?:figure|Figure|Fig\.?)\s+(\d+)",
        r"(?:refer\s+to\s+)?(?:figure|Figure|Fig\.?)\s+(\d+)",
        r"(?:see\s+)?(?:figure|Figure|Fig\.?)\s+(\d+)\s+above",
        r"(?:see\s+)?(?:figure|Figure|Fig\.?)\s+(\d+)\s+below",
    ],
    "section": [
        r"(?:see\s+)?(?:section|Section|Sec\.?)\s+([\d.]+)",
        r"(?:refer\s+to\s+)?(?:section|Section|Sec\.?)\s+([\d.]+)",
        r"(?:as\s+discussed\s+in\s+)?(?:section|Section|Sec\.?)\s+([\d.]+)",
        r"(?:see\s+)?(?:section|Section|Sec\.?)\s+([\d.]+)\s+above",
        r"(?:see\s+)?(?:section|Section|Sec\.?)\s+([\d.]+)\s+below",
        r"§\s*([\d.]+)",  # Section symbol
    ],
    "equation": [
        r"(?:see\s+)?(?:equation|Equation|Eq\.?)\s+(\d+)",
        r"(?:refer\s+to\s+)?(?:equation|Equation|Eq\.?)\s+(\d+)",
        r"Eq\.\s*(\d+)",
    ],
}


class Reference:
    """Represents a detected reference before resolution.

    Attributes:
        anchor_text: The text that triggered the reference.
        reference_type: Type of reference (table, figure, section, etc.).
        target_number: The number/identifier extracted from the text.
        position: Character position in the source text.
    """

    def __init__(
        self,
        anchor_text: str,
        reference_type: str,
        target_number: str,
        position: int = 0,
    ):
        """Initialize a reference.

        Args:
            anchor_text: The text that triggered the reference.
            reference_type: Type of reference.
            target_number: The number/identifier extracted.
            position: Character position in source text.
        """
        self.anchor_text = anchor_text
        self.reference_type = reference_type
        self.target_number = target_number
        self.position = position


class ReferenceResolver:
    """Specialized resolver for cross-references in document chunks.

    Detects references, resolves them to actual chunk IDs, and stores them
    as CrossReference objects in chunk metadata.
    """

    def __init__(self):
        """Initialize the ReferenceResolver."""
        self.compiled_patterns = {}
        for ref_type, patterns in REFERENCE_PATTERNS.items():
            self.compiled_patterns[ref_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def find_references(self, text: str) -> List[Reference]:
        """Find all references in a text string.

        Args:
            text: Text content to search for references.

        Returns:
            List of Reference objects found in the text.
        """
        references = []

        for ref_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    target_number = match.group(1)
                    anchor_text = match.group(0)
                    position = match.start()

                    # Avoid duplicates (same anchor text at same position)
                    if not any(
                        r.anchor_text == anchor_text and r.position == position
                        for r in references
                    ):
                        references.append(
                            Reference(
                                anchor_text=anchor_text,
                                reference_type=ref_type,
                                target_number=target_number,
                                position=position,
                            )
                        )

        return references

    def resolve_reference(
        self, reference: Reference, chunks: List[LDU]
    ) -> Optional[CrossReference]:
        """Resolve a reference to an actual chunk ID.

        Args:
            reference: The reference to resolve.
            chunks: List of all chunks to search.

        Returns:
            CrossReference object if resolved, None if unresolved.
        """
        target_chunk = self._find_target_chunk(reference, chunks)

        if target_chunk:
            return CrossReference(
                target_id=target_chunk.content_hash,
                reference_type=reference.reference_type,
                anchor_text=reference.anchor_text,
            )

        return None

    def resolve_all_references(
        self, chunks: List[LDU]
    ) -> Tuple[List[LDU], Dict[str, List[Reference]]]:
        """Resolve all cross-references in a list of chunks.

        Args:
            chunks: List of chunks to process.

        Returns:
            Tuple of (updated chunks, unresolved references dict).
            The unresolved dict maps chunk content_hash to list of unresolved references.
        """
        unresolved: Dict[str, List[Reference]] = {}

        # Build indices for faster lookup
        table_chunks = self._build_type_index(chunks, "table")
        figure_chunks = self._build_type_index(chunks, "figure")
        section_chunks = self._build_section_index(chunks)

        for chunk in chunks:
            # Skip tables and figures themselves (they don't reference themselves)
            if chunk.chunk_type in ("table", "figure"):
                continue

            # Find all references in this chunk
            references = self.find_references(chunk.content)

            chunk_unresolved = []

            for ref in references:
                # Resolve based on type
                if ref.reference_type == "table":
                    target_chunk = self._resolve_table_reference(
                        ref, table_chunks
                    )
                elif ref.reference_type == "figure":
                    target_chunk = self._resolve_figure_reference(
                        ref, figure_chunks
                    )
                elif ref.reference_type == "section":
                    target_chunk = self._resolve_section_reference(
                        ref, section_chunks
                    )
                elif ref.reference_type == "equation":
                    # Equations might not be chunks yet - skip for now
                    target_chunk = None
                else:
                    target_chunk = None

                if target_chunk:
                    # Create CrossReference
                    cross_ref = CrossReference(
                        target_id=target_chunk.content_hash,
                        reference_type=ref.reference_type,
                        anchor_text=ref.anchor_text,
                    )
                    chunk.cross_references.append(cross_ref)
                    logger.debug(
                        f"Resolved {ref.reference_type} reference '{ref.anchor_text}' "
                        f"to chunk {target_chunk.content_hash[:8]}"
                    )
                else:
                    # Store unresolved reference
                    chunk_unresolved.append(ref)
                    logger.debug(
                        f"Could not resolve {ref.reference_type} reference "
                        f"'{ref.anchor_text}' in chunk {chunk.content_hash[:8]}"
                    )

            if chunk_unresolved:
                unresolved[chunk.content_hash] = chunk_unresolved

        # Store unresolved references in metadata for audit
        for chunk in chunks:
            if chunk.content_hash in unresolved:
                chunk.metadata["unresolved_references"] = [
                    {
                        "anchor_text": r.anchor_text,
                        "reference_type": r.reference_type,
                        "target_number": r.target_number,
                    }
                    for r in unresolved[chunk.content_hash]
                ]

        logger.info(
            f"Resolved references in {len(chunks)} chunks. "
            f"{len(unresolved)} chunks have unresolved references."
        )

        return chunks, unresolved

    def _find_target_chunk(
        self, reference: Reference, chunks: List[LDU]
    ) -> Optional[LDU]:
        """Find the target chunk for a reference.

        Args:
            reference: The reference to resolve.
            chunks: List of chunks to search.

        Returns:
            Target chunk if found, None otherwise.
        """
        if reference.reference_type == "table":
            table_chunks = self._build_type_index(chunks, "table")
            return self._resolve_table_reference(reference, table_chunks)
        elif reference.reference_type == "figure":
            figure_chunks = self._build_type_index(chunks, "figure")
            return self._resolve_figure_reference(reference, figure_chunks)
        elif reference.reference_type == "section":
            section_chunks = self._build_section_index(chunks)
            return self._resolve_section_reference(reference, section_chunks)
        else:
            return None

    def _build_type_index(
        self, chunks: List[LDU], chunk_type: str
    ) -> Dict[int, LDU]:
        """Build an index of chunks by type and order.

        Args:
            chunks: List of chunks.
            chunk_type: Type to index (e.g., "table", "figure").

        Returns:
            Dictionary mapping order number (1-based) to chunk.
        """
        typed_chunks = [
            (i + 1, c) for i, c in enumerate(chunks) if c.chunk_type == chunk_type
        ]
        return dict(typed_chunks)

    def _build_section_index(self, chunks: List[LDU]) -> Dict[str, LDU]:
        """Build an index of section chunks by section number/path.

        Args:
            chunks: List of chunks.

        Returns:
            Dictionary mapping section identifier to chunk.
        """
        section_index = {}

        # Index by section number in title
        for chunk in chunks:
            if chunk.chunk_type == "header":
                # Extract section number from title
                section_num = self._extract_section_number(chunk.content)
                if section_num:
                    section_index[section_num] = chunk

                # Also index by section path if available
                section_path = chunk.metadata.get("section_path", "")
                if section_path:
                    section_index[section_path] = chunk

        return section_index

    def _extract_section_number(self, text: str) -> Optional[str]:
        """Extract section number from text.

        Args:
            text: Section header text.

        Returns:
            Section number (e.g., "1.1", "3.2.1") or None.
        """
        # Pattern for numbered sections
        pattern = r"^(\d+(?:\.\d+)*)[.)]\s+"
        match = re.match(pattern, text.strip())
        if match:
            return match.group(1)
        return None

    def _resolve_table_reference(
        self, reference: Reference, table_index: Dict[int, LDU]
    ) -> Optional[LDU]:
        """Resolve a table reference to a chunk.

        Args:
            reference: The table reference.
            table_index: Index of table chunks by order.

        Returns:
            Target table chunk if found, None otherwise.
        """
        try:
            table_num = int(reference.target_number)
            return table_index.get(table_num)
        except ValueError:
            return None

    def _resolve_figure_reference(
        self, reference: Reference, figure_index: Dict[int, LDU]
    ) -> Optional[LDU]:
        """Resolve a figure reference to a chunk.

        Args:
            reference: The figure reference.
            figure_index: Index of figure chunks by order.

        Returns:
            Target figure chunk if found, None otherwise.
        """
        try:
            figure_num = int(reference.target_number)
            return figure_index.get(figure_num)
        except ValueError:
            return None

    def _resolve_section_reference(
        self, reference: Reference, section_index: Dict[str, LDU]
    ) -> Optional[LDU]:
        """Resolve a section reference to a chunk.

        Args:
            reference: The section reference.
            section_index: Index of section chunks by identifier.

        Returns:
            Target section chunk if found, None otherwise.
        """
        # Try exact match first
        target_num = reference.target_number
        if target_num in section_index:
            return section_index[target_num]

        # Try partial match (e.g., "3.2" matches "3.2.1")
        for section_id, chunk in section_index.items():
            if section_id.startswith(target_num) or target_num.startswith(
                section_id
            ):
                return chunk

        return None


def resolve_cross_references(chunks: List[LDU]) -> Tuple[List[LDU], Dict[str, List]]:
    """Resolve all cross-references in chunks (convenience function).

    Args:
        chunks: List of chunks to process.

    Returns:
        Tuple of (updated chunks, unresolved references dict).
    """
    resolver = ReferenceResolver()
    return resolver.resolve_all_references(chunks)
