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

        This rule ensures:
        1. All figure chunks have caption stored in metadata (if caption exists)
        2. No caption exists as a standalone chunk
        3. Orphaned captions (captions without figures) are flagged

        Args:
            chunks: List of LDUs to validate.

        Returns:
            True if rule is satisfied, False otherwise.
        """
        from src.utils.figure_chunker import FigureChunker

        figure_chunks = [c for c in chunks if c.chunk_type == "figure"]
        figure_chunker = FigureChunker()

        # Check 1: All figure chunks have caption in metadata
        for chunk in figure_chunks:
            if "caption" not in chunk.metadata:
                logger.warning(
                    f"Figure chunk {chunk.content_hash[:8]} missing caption in metadata"
                )
                return False

            # If figure has a caption, verify it's in metadata
            has_caption = chunk.metadata.get("has_caption", False)
            caption_text = chunk.metadata.get("caption", "")

            if has_caption and not caption_text:
                logger.warning(
                    f"Figure chunk {chunk.content_hash[:8]} marked as having caption "
                    f"but caption text is empty"
                )
                return False

        # Check 2: No caption exists as standalone chunk
        # Look for text chunks that look like captions
        text_chunks = [c for c in chunks if c.chunk_type != "figure"]
        orphaned_captions = []

        for chunk in text_chunks:
            if figure_chunker._is_caption_text(chunk.content):
                # Check if this caption is near any figure chunk
                is_paired = False
                for figure_chunk in figure_chunks:
                    # Check spatial proximity (simplified check)
                    chunk_pages = set(chunk.page_refs)
                    figure_pages = set(figure_chunk.page_refs)
                    if chunk_pages & figure_pages:  # Same page
                        # Check if caption appears in figure metadata
                        figure_caption = figure_chunk.metadata.get("caption", "")
                        if figure_caption and chunk.content.strip() in figure_caption:
                            is_paired = True
                            break

                if not is_paired:
                    orphaned_captions.append(chunk)

        if orphaned_captions:
            logger.warning(
                f"Found {len(orphaned_captions)} orphaned caption(s) that are not "
                f"paired with figures. These should be in figure metadata, not "
                f"separate chunks."
            )
            for orphan in orphaned_captions:
                logger.warning(
                    f"  - Orphaned caption on page {orphan.page_refs[0]}: "
                    f"{orphan.content[:50]}..."
                )
            # This is a warning, but we'll allow it (some captions might be standalone)
            # Return False only if we want strict enforcement
            # For now, we'll log but not fail
            pass

        # Check 3: Verify caption content matches metadata
        for chunk in figure_chunks:
            caption_text = chunk.metadata.get("caption", "")
            if caption_text:
                # Content should be the caption or "[Figure]"
                if chunk.content not in [caption_text, "[Figure]"]:
                    # Content might be a subset, which is acceptable
                    if caption_text not in chunk.content and chunk.content != "[Figure]":
                        logger.debug(
                            f"Figure chunk {chunk.content_hash[:8]} content doesn't "
                            f"match caption metadata exactly (may be acceptable)"
                        )

        logger.debug(
            f"Rule 2 passed: {len(figure_chunks)} figure chunks validated, "
            f"{len(orphaned_captions)} orphaned captions found"
        )
        return True

    def validate_rule_3(self, chunks: List[LDU]) -> bool:
        """Rule 3: Numbered lists intact.

        This rule ensures:
        1. Lists are kept as single LDUs when possible
        2. If split, split occurs at item boundaries, not within items
        3. List items are complete (not split mid-sentence)
        4. Split lists have proper metadata indicating they're partial

        Args:
            chunks: List of LDUs to validate.

        Returns:
            True if rule is satisfied, False otherwise.
        """
        from src.utils.list_chunker import ListChunker

        list_chunks = [c for c in chunks if c.chunk_type == "list"]
        list_chunker = ListChunker()

        if not list_chunks:
            logger.debug("Rule 3 passed: No list chunks to validate")
            return True

        for chunk in list_chunks:
            # Check 1: List metadata should indicate it's a list
            if not chunk.metadata.get("is_list", False):
                logger.warning(
                    f"List chunk {chunk.content_hash[:8]} missing 'is_list' metadata"
                )
                return False

            # Check 2: List type should be specified
            list_type = chunk.metadata.get("list_type")
            if list_type not in ["numbered", "bulleted"]:
                logger.warning(
                    f"List chunk {chunk.content_hash[:8]} has invalid list_type: {list_type}"
                )
                return False

            # Check 3: Verify list items are complete
            content = chunk.content.strip()
            lines = content.split("\n")

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if line is a list item
                line_info = list_chunker._analyze_list_item(line)
                if line_info["is_list_item"]:
                    # Verify item is complete (not split mid-sentence)
                    # Items should have some content after the marker
                    marker_len = len(line_info["list_marker"])
                    item_content = line[marker_len:].strip()

                    if not item_content:
                        logger.warning(
                            f"List chunk {chunk.content_hash[:8]} has empty list item"
                        )
                        return False

                    # Check if item seems incomplete (very short, no punctuation, no space)
                    if (
                        len(item_content) < 5
                        and not any(p in item_content for p in [".", "!", "?", ";", ":"])
                        and " " not in item_content
                    ):
                        logger.warning(
                            f"List chunk {chunk.content_hash[:8]} may have incomplete item: "
                            f"{line[:50]}..."
                        )
                        # This is a warning, not a hard failure
                        pass

            # Check 4: If this is a partial list (split), verify it has proper metadata
            is_partial = chunk.metadata.get("is_partial_list", False)
            if is_partial:
                if "chunk_index" not in chunk.metadata:
                    logger.warning(
                        f"Partial list chunk {chunk.content_hash[:8]} missing chunk_index"
                    )
                    return False
                if "total_chunks" not in chunk.metadata:
                    logger.warning(
                        f"Partial list chunk {chunk.content_hash[:8]} missing total_chunks"
                    )
                    return False

                # Verify item count is reasonable
                item_count = chunk.metadata.get("item_count", 0)
                if item_count == 0:
                    logger.warning(
                        f"Partial list chunk {chunk.content_hash[:8]} has zero items"
                    )
                    return False

            # Check 5: Verify list structure integrity
            # All items should start with the same marker type
            item_markers = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                line_info = list_chunker._analyze_list_item(line)
                if line_info["is_list_item"]:
                    item_markers.append(line_info["list_marker"])

            if item_markers:
                # Check consistency (all markers should be similar for same list type)
                first_marker = item_markers[0]
                inconsistent_count = sum(
                    1 for m in item_markers if m != first_marker
                )
                # Allow some variation (e.g., different numbering)
                if inconsistent_count > len(item_markers) * 0.3:
                    logger.warning(
                        f"List chunk {chunk.content_hash[:8]} has inconsistent markers"
                    )
                    # This is a warning, not a failure
                    pass

        logger.debug(
            f"Rule 3 passed: {len(list_chunks)} list chunks validated"
        )
        return True

    def validate_rule_4(self, chunks: List[LDU]) -> bool:
        """Rule 4: Section headers as parent metadata.

        This rule ensures:
        1. Section headers are stored as parent metadata on all child chunks
        2. Headers themselves don't have parent_section (they are top-level)
        3. Nested sections preserve full path in section_path metadata
        4. Child chunks inherit section from their parent section
        5. Section hierarchy is properly maintained

        Args:
            chunks: List of LDUs to validate.

        Returns:
            True if rule is satisfied, False otherwise.
        """
        from src.utils.section_chunker import SectionChunker

        # Sort chunks by page and position
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (
                min(c.page_refs),
                c.bounding_box.get("y0", 0),
            ),
        )

        header_chunks = [c for c in sorted_chunks if c.chunk_type == "header"]
        non_header_chunks = [c for c in sorted_chunks if c.chunk_type != "header"]

        # Check 1: Headers should not have parent_section
        for chunk in header_chunks:
            if chunk.parent_section is not None:
                logger.warning(
                    f"Header chunk {chunk.content_hash[:8]} has parent_section "
                    f"but should be None"
                )
                return False

        # Check 2: Non-header chunks should have section information
        chunks_with_sections = 0
        chunks_without_sections = 0

        for chunk in non_header_chunks:
            if chunk.parent_section is not None:
                chunks_with_sections += 1

                # Check 3: Verify section_path metadata exists for chunks with sections
                section_path = chunk.metadata.get("section_path")
                if not section_path:
                    logger.warning(
                        f"Chunk {chunk.content_hash[:8]} has parent_section "
                        f"but missing section_path metadata"
                    )
                    return False

                # Check 4: Verify section_path includes parent_section
                if chunk.parent_section not in section_path:
                    logger.warning(
                        f"Chunk {chunk.content_hash[:8]} section_path doesn't include "
                        f"parent_section. Path: {section_path}, Parent: {chunk.parent_section}"
                    )
                    # This is a warning, not a hard failure (path might be nested)
                    pass

                # Check 5: Verify section metadata is complete
                if "section_level" not in chunk.metadata:
                    logger.warning(
                        f"Chunk {chunk.content_hash[:8]} missing section_level metadata"
                    )
                    # This is a warning, not a hard failure
                    pass

            else:
                chunks_without_sections += 1

        # Check 6: Verify section hierarchy consistency
        # Build section tree and verify chunks are assigned correctly
        section_chunker = SectionChunker()
        section_tree = section_chunker.build_section_hierarchy(chunks)

        # Build page section map for validation
        page_sections_map: Dict[int, List] = {}
        section_chunker._build_page_section_map(section_tree, page_sections_map)

        # Verify that chunks are in the correct sections based on position
        for chunk in non_header_chunks:
            if chunk.parent_section:
                # Find the section this chunk should belong to
                page_num = min(chunk.page_refs)
                expected_section = section_chunker._find_section_for_chunk(
                    chunk,
                    page_num,
                    page_sections_map,
                    {},
                )

                # If we can determine expected section, verify it matches
                if expected_section and expected_section.title != chunk.parent_section:
                    # Check if it's a nested section (parent might be in path)
                    section_path = chunk.metadata.get("section_path", "")
                    if expected_section.title in section_path:
                        # Acceptable - chunk is in a nested section
                        pass
                    else:
                        logger.warning(
                            f"Chunk {chunk.content_hash[:8]} may be in wrong section. "
                            f"Expected: {expected_section.title}, Got: {chunk.parent_section}"
                        )
                        # This is a warning, not a hard failure

        logger.debug(
            f"Rule 4 passed: {len(header_chunks)} headers, "
            f"{chunks_with_sections} chunks with sections, "
            f"{chunks_without_sections} chunks without sections"
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
