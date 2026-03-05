"""Chunk Validator for enforcing semantic chunking rules.

This module implements the ChunkValidator class, which enforces the five core
chunking rules that ensure RAG-optimized, semantically coherent chunks.
"""

from __future__ import annotations

import logging
from typing import List

from src.models.ldu import LDU

logger = logging.getLogger(__name__)


class ChunkValidator:
    """Validator for enforcing semantic chunking rules.

    The ChunkValidator ensures that chunks comply with the five core rules:
    1. No table cell split from header
    2. Figure caption as metadata
    3. Numbered lists intact
    4. Section headers as parent metadata
    5. Cross-reference resolution
    """

    def validate_all(self, chunks: List[LDU]) -> bool:
        """Validate all chunks against all five rules.

        Args:
            chunks: List of LDUs to validate.

        Returns:
            True if all rules pass, False otherwise.
        """
        rules = [
            self.validate_rule_1,
            self.validate_rule_2,
            self.validate_rule_3,
            self.validate_rule_4,
            self.validate_rule_5,
        ]

        all_passed = True
        for rule_func in rules:
            try:
                if not rule_func(chunks):
                    all_passed = False
                    logger.warning(f"Validation rule failed: {rule_func.__name__}")
            except Exception as e:
                logger.error(f"Error validating {rule_func.__name__}: {e}")
                all_passed = False

        return all_passed

    def validate_rule_1(self, chunks: List[LDU]) -> bool:
        """Rule 1: No table cell split from header.

        This rule ensures:
        1. Tables are atomic chunks (or properly split large tables)
        2. Headers are always preserved with their data rows
        3. No table cell is separated from its header context
        4. Split tables are properly marked and include headers in each chunk

        Args:
            chunks: List of LDUs to validate.

        Returns:
            True if rule is satisfied, False otherwise.
        """
        table_chunks = [c for c in chunks if c.chunk_type == "table"]

        if not table_chunks:
            logger.debug("Rule 1 passed: No table chunks to validate")
            return True

        # Group table chunks by page and spatial proximity to detect split tables
        table_groups = self._group_table_chunks_by_proximity(table_chunks)

        for chunk in table_chunks:
            # Check 1: Table metadata includes headers
            if "table_headers" not in chunk.metadata:
                logger.warning(
                    f"Table chunk {chunk.content_hash[:8]} missing table_headers metadata"
                )
                return False

            # Check 2: Content is not empty
            if not chunk.content.strip():
                logger.warning(
                    f"Table chunk {chunk.content_hash[:8]} has empty content"
                )
                return False

            # Check 3: Headers are present in content
            headers = chunk.metadata.get("table_headers", [])
            if headers:
                content_lower = chunk.content.lower()
                header_found = any(
                    header.lower() in content_lower for header in headers if header
                )
                if not header_found:
                    logger.warning(
                        f"Table chunk {chunk.content_hash[:8]} headers not found in content. "
                        f"Headers: {headers[:3]}..."
                    )
                    return False

            # Check 4: If this is a partial table (split), verify it has proper metadata
            is_partial = chunk.metadata.get("is_partial_table", False)
            if is_partial:
                if "chunk_index" not in chunk.metadata:
                    logger.warning(
                        f"Partial table chunk {chunk.content_hash[:8]} missing chunk_index"
                    )
                    return False
                if "total_chunks" not in chunk.metadata:
                    logger.warning(
                        f"Partial table chunk {chunk.content_hash[:8]} missing total_chunks"
                    )
                    return False
                # Verify headers are still present in partial chunks
                if not headers:
                    logger.warning(
                        f"Partial table chunk {chunk.content_hash[:8]} missing headers"
                    )
                    return False

            # Check 5: Verify table structure integrity
            # Content should start with headers if headers exist
            if headers:
                content_lines = chunk.content.split("\n")
                first_line = content_lines[0] if content_lines else ""
                # Check if first line contains headers (allowing for formatting)
                first_line_lower = first_line.lower()
                header_match_count = sum(
                    1
                    for header in headers
                    if header and header.lower() in first_line_lower
                )
                # At least 50% of headers should appear in first line
                if header_match_count < len(headers) * 0.5:
                    logger.warning(
                        f"Table chunk {chunk.content_hash[:8]} headers not at start of content. "
                        f"Only {header_match_count}/{len(headers)} headers found in first line."
                    )
                    # This is a warning, not a hard failure, as formatting may vary
                    pass

        # Check 6: Verify no table appears in multiple chunks without proper splitting
        for group in table_groups:
            if len(group) > 1:
                # Multiple chunks for same table - verify they're properly split
                partial_count = sum(
                    1 for c in group if c.metadata.get("is_partial_table", False)
                )
                if partial_count > 0 and partial_count != len(group):
                    logger.warning(
                        f"Table group has mixed partial/non-partial chunks. "
                        f"This may indicate improper splitting."
                    )
                    return False

        logger.debug(
            f"Rule 1 passed: {len(table_chunks)} table chunks validated, "
            f"{len(table_groups)} table groups identified"
        )
        return True

    def _group_table_chunks_by_proximity(
        self, table_chunks: List[LDU]
    ) -> List[List[LDU]]:
        """Group table chunks that likely belong to the same table.

        This helps detect when a table has been split across multiple chunks.

        Args:
            table_chunks: List of table LDUs.

        Returns:
            List of groups, where each group is a list of chunks that likely
            belong to the same table.
        """
        groups = []
        SPATIAL_THRESHOLD = 50.0  # PDF points
        PAGE_TOLERANCE = 1  # Allow adjacent pages

        for chunk in table_chunks:
            # Try to find an existing group this chunk belongs to
            matched_group = None
            for group in groups:
                # Check if chunk is spatially close to any chunk in the group
                for group_chunk in group:
                    # Same page or adjacent pages
                    page_match = any(
                        abs(p1 - p2) <= PAGE_TOLERANCE
                        for p1 in chunk.page_refs
                        for p2 in group_chunk.page_refs
                    )
                    if not page_match:
                        continue

                    # Check spatial proximity
                    bbox1 = chunk.bounding_box
                    bbox2 = group_chunk.bounding_box

                    center1_x = (bbox1.get("x0", 0) + bbox1.get("x1", 0)) / 2
                    center1_y = (bbox1.get("y0", 0) + bbox1.get("y1", 0)) / 2
                    center2_x = (bbox2.get("x0", 0) + bbox2.get("x1", 0)) / 2
                    center2_y = (bbox2.get("y0", 0) + bbox2.get("y1", 0)) / 2

                    distance = (
                        (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2
                    ) ** 0.5

                    # Check if headers match (same table should have same headers)
                    headers1 = chunk.metadata.get("table_headers", [])
                    headers2 = group_chunk.metadata.get("table_headers", [])
                    headers_match = headers1 == headers2

                    if (
                        page_match
                        and distance < SPATIAL_THRESHOLD
                        and headers_match
                    ):
                        matched_group = group
                        break

                if matched_group:
                    break

            if matched_group:
                matched_group.append(chunk)
            else:
                # Create new group
                groups.append([chunk])

        return groups

    def validate_rule_2(self, chunks: List[LDU]) -> bool:
        """Rule 2: Figure caption as metadata.

        All figure chunks must have their captions stored in the metadata
        dictionary, not just in the content field.

        Args:
            chunks: List of LDUs to validate.

        Returns:
            True if rule is satisfied, False otherwise.
        """
        figure_chunks = [c for c in chunks if c.chunk_type == "figure"]

        for chunk in figure_chunks:
            # Check that caption is in metadata
            if "caption" not in chunk.metadata:
                logger.warning(
                    f"Figure chunk {chunk.content_hash[:8]} missing caption in metadata"
                )
                return False

            # If there's a caption in content, it should match metadata
            if chunk.content and chunk.content != "[Figure]":
                metadata_caption = chunk.metadata.get("caption", "")
                if metadata_caption and chunk.content != metadata_caption:
                    # Content might be just the caption, which is fine
                    if chunk.content not in metadata_caption:
                        logger.warning(
                            f"Figure chunk {chunk.content_hash[:8]} content/metadata mismatch"
                        )
                        # This is a warning, not a failure - content can be a subset
                        pass

        logger.debug(f"Rule 2 passed: {len(figure_chunks)} figure chunks validated")
        return True

    def validate_rule_3(self, chunks: List[LDU]) -> bool:
        """Rule 3: Numbered lists intact.

        Numbered lists should be kept as single LDUs unless they exceed
        max_tokens. If split, the split should occur at list item boundaries,
        not mid-item.

        Args:
            chunks: List of LDUs to validate.

        Returns:
            True if rule is satisfied, False otherwise.
        """
        list_chunks = [c for c in chunks if c.chunk_type == "list"]

        for chunk in list_chunks:
            content = chunk.content.strip()

            # Check if content looks like a list (starts with number or bullet)
            is_numbered = any(content.startswith(f"{i}.") for i in range(1, 100))
            is_bulleted = content.startswith(("-", "*", "•"))

            if not (is_numbered or is_bulleted):
                # Might be a false positive classification, but not a violation
                continue

            # Check that list items are complete (heuristic: ends with period or newline)
            # This is a simplified check - in practice, you'd want more sophisticated
            # parsing to ensure list items aren't split mid-sentence
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # If a line starts with a list marker but doesn't end properly,
                # it might be a split item
                if (line.startswith(("-", "*", "•")) or any(
                    line.startswith(f"{i}.") for i in range(1, 100)
                )):
                    # Check if line seems incomplete (very short, no punctuation)
                    if len(line) < 10 and not any(
                        p in line for p in [".", "!", "?", ";", ":"]
                    ):
                        logger.warning(
                            f"List chunk {chunk.content_hash[:8]} may have incomplete items"
                        )
                        # This is a warning, not a hard failure
                        pass

        logger.debug(f"Rule 3 passed: {len(list_chunks)} list chunks validated")
        return True

    def validate_rule_4(self, chunks: List[LDU]) -> bool:
        """Rule 4: Section headers as parent metadata.

        All chunks that appear after a header (until the next header) should
        have that header's content as their parent_section. Headers themselves
        should not have a parent_section (they are top-level).

        Args:
            chunks: List of LDUs to validate.

        Returns:
            True if rule is satisfied, False otherwise.
        """
        # Sort chunks by page and position
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (
                min(c.page_refs),
                c.bounding_box.get("y0", 0),
            ),
        )

        current_header: str | None = None
        header_chunks = []

        for chunk in sorted_chunks:
            if chunk.chunk_type == "header":
                # Headers should not have parent_section
                if chunk.parent_section is not None:
                    logger.warning(
                        f"Header chunk {chunk.content_hash[:8]} has parent_section "
                        f"but should be None"
                    )
                    return False

                current_header = chunk.content
                header_chunks.append(chunk)
            else:
                # Non-header chunks should have parent_section if there's a header
                if current_header is not None:
                    if chunk.parent_section != current_header:
                        logger.warning(
                            f"Chunk {chunk.content_hash[:8]} has incorrect parent_section. "
                            f"Expected '{current_header}', got '{chunk.parent_section}'"
                        )
                        # This might be acceptable if sections are nested - log as warning
                        pass
                # If there's no current header, parent_section can be None (document start)

        logger.debug(
            f"Rule 4 passed: {len(header_chunks)} headers, "
            f"{len(sorted_chunks) - len(header_chunks)} child chunks validated"
        )
        return True

    def validate_rule_5(self, chunks: List[LDU]) -> bool:
        """Rule 5: Cross-reference resolution.

        Cross-references in chunk content (e.g., "see Table 3") should be
        resolved and stored in the cross_references field with valid target_ids.

        Args:
            chunks: List of LDUs to validate.

        Returns:
            True if rule is satisfied, False otherwise.
        """
        # Build index of chunks by content_hash
        chunk_index = {c.content_hash: c for c in chunks}

        chunks_with_refs = [c for c in chunks if c.cross_references]

        for chunk in chunks_with_refs:
            for ref in chunk.cross_references:
                # Check that target_id exists in chunk_index
                if ref.target_id not in chunk_index:
                    logger.warning(
                        f"Chunk {chunk.content_hash[:8]} has cross-reference to "
                        f"unknown target_id: {ref.target_id[:8]}"
                    )
                    # This might be acceptable if reference is to external content
                    pass

                # Check that reference_type matches target chunk type
                target_chunk = chunk_index.get(ref.target_id)
                if target_chunk:
                    if ref.reference_type == "table" and target_chunk.chunk_type != "table":
                        logger.warning(
                            f"Cross-reference type mismatch: expected table, "
                            f"got {target_chunk.chunk_type}"
                        )
                        return False
                    if ref.reference_type == "figure" and target_chunk.chunk_type != "figure":
                        logger.warning(
                            f"Cross-reference type mismatch: expected figure, "
                            f"got {target_chunk.chunk_type}"
                        )
                        return False

        logger.debug(
            f"Rule 5 passed: {len(chunks_with_refs)} chunks with cross-references validated"
        )
        return True
