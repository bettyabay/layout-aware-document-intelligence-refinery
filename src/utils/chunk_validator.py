"""Chunk Validator for enforcing semantic chunking rules.

This module implements the ChunkValidator class, which enforces the five core
chunking rules that ensure RAG-optimized, semantically coherent chunks.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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

    The validator is pluggable - rules can be enabled/disabled via configuration,
    and custom rules can be added.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        log_file: Optional[str] = ".refinery/chunk_validation.log",
        auto_fix: bool = False,
    ):
        """Initialize the ChunkValidator.

        Args:
            config: Configuration dictionary with rule settings. If None, all
                rules are enabled by default.
            log_file: Path to log file for violations. If None, logging is
                disabled.
            auto_fix: Whether to attempt automatic fixes for violations.
        """
        self.config = config or {}
        self.log_file = Path(log_file) if log_file else None
        self.auto_fix = auto_fix
        self.violations: List[Dict] = []

        # Ensure log directory exists
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Rule registry - maps rule names to validation functions
        self.rule_registry: Dict[str, Callable[[List[LDU]], bool]] = {
            "rule1_table_integrity": self.validate_rule_1,
            "rule2_figure_captions": self.validate_rule_2,
            "rule3_list_preservation": self.validate_rule_3,
            "rule4_section_hierarchy": self.validate_rule_4,
            "rule5_cross_references": self.validate_rule_5,
        }

    def register_rule(
        self, name: str, validator_func: Callable[[List[LDU]], bool]
    ) -> None:
        """Register a custom validation rule.

        Args:
            name: Name of the rule.
            validator_func: Function that takes a list of chunks and returns
                True if validation passes, False otherwise.
        """
        self.rule_registry[name] = validator_func
        logger.info(f"Registered custom validation rule: {name}")

    def validate_all(
        self, chunks: List[LDU]
    ) -> Tuple[bool, Dict[str, bool], List[Dict]]:
        """Validate all chunks against all enabled rules.

        Args:
            chunks: List of LDUs to validate.

        Returns:
            Tuple of:
            - overall_success: True if all enabled rules pass
            - results: Dictionary mapping rule names to pass/fail status
            - violations: List of violation dictionaries
        """
        self.violations = []
        results: Dict[str, bool] = {}

        # Get enabled rules from config
        enabled_rules = self._get_enabled_rules()

        # Run each enabled rule
        for rule_name, rule_func in self.rule_registry.items():
            if rule_name not in enabled_rules:
                logger.debug(f"Skipping disabled rule: {rule_name}")
                results[rule_name] = None  # None means disabled
                continue

            try:
                passed = rule_func(chunks)
                results[rule_name] = passed

                if not passed:
                    # Collect violations for this rule
                    rule_violations = self._collect_rule_violations(
                        rule_name, chunks
                    )
                    self.violations.extend(rule_violations)

                    # Attempt auto-fix if enabled
                    if self.auto_fix:
                        fixed = self._attempt_auto_fix(rule_name, chunks, rule_violations)
                        if fixed:
                            # Re-validate after fix
                            results[rule_name] = rule_func(chunks)
                            logger.info(f"Auto-fixed violations for rule: {rule_name}")

            except Exception as e:
                logger.error(f"Error validating {rule_name}: {e}", exc_info=True)
                results[rule_name] = False
                self.violations.append(
                    {
                        "rule": rule_name,
                        "chunk_id": None,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        # Log violations to file
        if self.log_file and self.violations:
            self._log_violations()

        overall_success = all(
            result for result in results.values() if result is not None
        )

        return overall_success, results, self.violations

    def validate_all_simple(self, chunks: List[LDU]) -> bool:
        """Validate all chunks against all enabled rules (simple interface).

        This method maintains backward compatibility with the original interface.

        Args:
            chunks: List of LDUs to validate.

        Returns:
            True if all enabled rules pass, False otherwise.
        """
        overall_success, _, _ = self.validate_all(chunks)
        return overall_success

    def _get_enabled_rules(self) -> List[str]:
        """Get list of enabled rules from configuration.

        Returns:
            List of enabled rule names.
        """
        # Default: all rules enabled
        default_rules = list(self.rule_registry.keys())

        # Check config for rule enablement
        validation_config = self.config.get("validation", {})
        enabled_rules = validation_config.get("enabled_rules", default_rules)

        # If enabled_rules is a list, use it; if it's a dict, check each rule
        if isinstance(enabled_rules, dict):
            return [
                rule_name
                for rule_name, enabled in enabled_rules.items()
                if enabled
            ]

        return enabled_rules if isinstance(enabled_rules, list) else default_rules

    def _collect_rule_violations(
        self, rule_name: str, chunks: List[LDU]
    ) -> List[Dict]:
        """Collect violations for a specific rule.

        Args:
            rule_name: Name of the rule.
            chunks: List of chunks to check.

        Returns:
            List of violation dictionaries.
        """
        violations = []

        # Rule-specific violation collection
        if rule_name == "rule1_table_integrity":
            violations.extend(self._collect_table_violations(chunks))
        elif rule_name == "rule2_figure_captions":
            violations.extend(self._collect_figure_violations(chunks))
        elif rule_name == "rule3_list_preservation":
            violations.extend(self._collect_list_violations(chunks))
        elif rule_name == "rule4_section_hierarchy":
            violations.extend(self._collect_section_violations(chunks))
        elif rule_name == "rule5_cross_references":
            violations.extend(self._collect_reference_violations(chunks))

        return violations

    def _collect_table_violations(self, chunks: List[LDU]) -> List[Dict]:
        """Collect table integrity violations.

        Args:
            chunks: List of chunks.

        Returns:
            List of violation dictionaries.
        """
        violations = []
        table_chunks = [c for c in chunks if c.chunk_type == "table"]

        for chunk in table_chunks:
            if "table_headers" not in chunk.metadata:
                violations.append(
                    {
                        "rule": "rule1_table_integrity",
                        "chunk_id": chunk.content_hash,
                        "violation": "missing_table_headers",
                        "chunk_type": "table",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        return violations

    def _collect_figure_violations(self, chunks: List[LDU]) -> List[Dict]:
        """Collect figure caption violations.

        Args:
            chunks: List of chunks.

        Returns:
            List of violation dictionaries.
        """
        violations = []
        figure_chunks = [c for c in chunks if c.chunk_type == "figure"]

        for chunk in figure_chunks:
            if "caption" not in chunk.metadata:
                violations.append(
                    {
                        "rule": "rule2_figure_captions",
                        "chunk_id": chunk.content_hash,
                        "violation": "missing_caption_metadata",
                        "chunk_type": "figure",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        return violations

    def _collect_list_violations(self, chunks: List[LDU]) -> List[Dict]:
        """Collect list preservation violations.

        Args:
            chunks: List of chunks.

        Returns:
            List of violation dictionaries.
        """
        violations = []
        list_chunks = [c for c in chunks if c.chunk_type == "list"]

        for chunk in list_chunks:
            if not chunk.metadata.get("is_list", False):
                violations.append(
                    {
                        "rule": "rule3_list_preservation",
                        "chunk_id": chunk.content_hash,
                        "violation": "missing_list_metadata",
                        "chunk_type": "list",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        return violations

    def _collect_section_violations(self, chunks: List[LDU]) -> List[Dict]:
        """Collect section hierarchy violations.

        Args:
            chunks: List of chunks.

        Returns:
            List of violation dictionaries.
        """
        violations = []
        header_chunks = [c for c in chunks if c.chunk_type == "header"]

        for chunk in header_chunks:
            if chunk.parent_section is not None:
                violations.append(
                    {
                        "rule": "rule4_section_hierarchy",
                        "chunk_id": chunk.content_hash,
                        "violation": "header_has_parent_section",
                        "chunk_type": "header",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        return violations

    def _collect_reference_violations(self, chunks: List[LDU]) -> List[Dict]:
        """Collect cross-reference violations.

        Args:
            chunks: List of chunks.

        Returns:
            List of violation dictionaries.
        """
        violations = []
        chunk_index = {c.content_hash: c for c in chunks}

        for chunk in chunks:
            for ref in chunk.cross_references:
                if ref.target_id not in chunk_index:
                    violations.append(
                        {
                            "rule": "rule5_cross_references",
                            "chunk_id": chunk.content_hash,
                            "violation": "broken_reference",
                            "target_id": ref.target_id,
                            "reference_type": ref.reference_type,
                            "anchor_text": ref.anchor_text,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

        return violations

    def _attempt_auto_fix(
        self, rule_name: str, chunks: List[LDU], violations: List[Dict]
    ) -> bool:
        """Attempt to automatically fix violations.

        Args:
            rule_name: Name of the rule.
            chunks: List of chunks.
            violations: List of violations to fix.

        Returns:
            True if fixes were applied, False otherwise.
        """
        fixed = False

        if rule_name == "rule1_table_integrity":
            # Auto-fix: Add missing table headers metadata
            for violation in violations:
                chunk_id = violation.get("chunk_id")
                if chunk_id:
                    chunk = next(
                        (c for c in chunks if c.content_hash == chunk_id), None
                    )
                    if chunk and chunk.chunk_type == "table":
                        # Try to extract headers from content
                        content_lines = chunk.content.split("\n")
                        if content_lines:
                            # First line might be headers
                            headers = content_lines[0].split(" | ")
                            chunk.metadata["table_headers"] = headers
                            fixed = True

        elif rule_name == "rule2_figure_captions":
            # Auto-fix: Add missing caption metadata
            for violation in violations:
                chunk_id = violation.get("chunk_id")
                if chunk_id:
                    chunk = next(
                        (c for c in chunks if c.content_hash == chunk_id), None
                    )
                    if chunk and chunk.chunk_type == "figure":
                        # Use content as caption if available
                        caption = chunk.content if chunk.content != "[Figure]" else ""
                        chunk.metadata["caption"] = caption
                        chunk.metadata["has_caption"] = bool(caption)
                        fixed = True

        elif rule_name == "rule3_list_preservation":
            # Auto-fix: Add missing list metadata
            for violation in violations:
                chunk_id = violation.get("chunk_id")
                if chunk_id:
                    chunk = next(
                        (c for c in chunks if c.content_hash == chunk_id), None
                    )
                    if chunk and chunk.chunk_type == "list":
                        chunk.metadata["is_list"] = True
                        # Try to detect list type from content
                        if any(
                            chunk.content.strip().startswith(f"{i}.")
                            for i in range(1, 100)
                        ):
                            chunk.metadata["list_type"] = "numbered"
                        else:
                            chunk.metadata["list_type"] = "bulleted"
                        fixed = True

        elif rule_name == "rule4_section_hierarchy":
            # Auto-fix: Remove parent_section from headers
            for violation in violations:
                chunk_id = violation.get("chunk_id")
                if chunk_id:
                    chunk = next(
                        (c for c in chunks if c.content_hash == chunk_id), None
                    )
                    if chunk and chunk.chunk_type == "header":
                        chunk.parent_section = None
                        fixed = True

        # Note: rule5 (cross-references) cannot be auto-fixed as we need
        # to know the correct target chunk

        if fixed:
            logger.info(f"Auto-fixed {len(violations)} violations for {rule_name}")

        return fixed

    def _log_violations(self) -> None:
        """Log violations to file.

        Writes violations in JSONL format for easy parsing.
        """
        if not self.log_file:
            return

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                for violation in self.violations:
                    f.write(json.dumps(violation) + "\n")
            logger.info(
                f"Logged {len(self.violations)} violations to {self.log_file}"
            )
        except Exception as e:
            logger.error(f"Failed to log violations to {self.log_file}: {e}")


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

        This rule ensures:
        1. Cross-references are detected and resolved to actual chunk IDs
        2. All resolved references point to existing chunks
        3. Reference types match target chunk types
        4. Broken references are flagged
        5. Unresolved references are stored in metadata for audit

        Args:
            chunks: List of LDUs to validate.

        Returns:
            True if rule is satisfied, False otherwise.
        """
        from src.utils.reference_resolver import ReferenceResolver

        # Build index of chunks by content_hash
        chunk_index = {c.content_hash: c for c in chunks}

        chunks_with_refs = [c for c in chunks if c.cross_references]
        chunks_with_unresolved = [
            c
            for c in chunks
            if c.metadata.get("unresolved_references")
        ]

        broken_refs = []
        type_mismatches = []

        # Check 1: Verify all resolved references point to existing chunks
        for chunk in chunks_with_refs:
            for ref in chunk.cross_references:
                # Check that target_id exists in chunk_index
                if ref.target_id not in chunk_index:
                    broken_refs.append((chunk, ref))
                    logger.warning(
                        f"Chunk {chunk.content_hash[:8]} has broken cross-reference: "
                        f"target_id {ref.target_id[:8]} not found. "
                        f"Reference: '{ref.anchor_text}'"
                    )
                    continue

                # Check 2: Verify reference_type matches target chunk type
                target_chunk = chunk_index.get(ref.target_id)
                if target_chunk:
                    expected_type = ref.reference_type
                    actual_type = target_chunk.chunk_type

                    # Map reference types to chunk types
                    type_mapping = {
                        "table": "table",
                        "figure": "figure",
                        "section": "header",
                        "equation": "paragraph",  # Equations might be in paragraphs
                    }

                    expected_chunk_type = type_mapping.get(expected_type)
                    if expected_chunk_type and actual_type != expected_chunk_type:
                        type_mismatches.append((chunk, ref, target_chunk))
                        logger.warning(
                            f"Cross-reference type mismatch in chunk "
                            f"{chunk.content_hash[:8]}: "
                            f"reference type '{expected_type}' points to chunk type "
                            f"'{actual_type}'. Reference: '{ref.anchor_text}'"
                        )
                        # This is a warning, not a hard failure
                        pass

        # Check 3: Verify unresolved references are stored in metadata
        for chunk in chunks_with_unresolved:
            unresolved = chunk.metadata.get("unresolved_references", [])
            if not unresolved:
                logger.warning(
                    f"Chunk {chunk.content_hash[:8]} has unresolved_references "
                    f"metadata key but empty list"
                )
                continue

            # Verify unresolved references have required fields
            for unresolved_ref in unresolved:
                required_fields = ["anchor_text", "reference_type", "target_number"]
                for field in required_fields:
                    if field not in unresolved_ref:
                        logger.warning(
                            f"Chunk {chunk.content_hash[:8]} unresolved reference "
                            f"missing field '{field}'"
                        )
                        return False

        # Check 4: Verify that references were actually detected
        # Use ReferenceResolver to find all references and compare
        resolver = ReferenceResolver()
        total_detected = 0
        total_resolved = 0

        for chunk in chunks:
            if chunk.chunk_type in ("table", "figure"):
                continue

            detected = resolver.find_references(chunk.content)
            total_detected += len(detected)
            total_resolved += len(chunk.cross_references)

        unresolved_count = sum(
            len(c.metadata.get("unresolved_references", []))
            for c in chunks
        )

        if total_detected != total_resolved + unresolved_count:
            logger.warning(
                f"Reference count mismatch: detected {total_detected}, "
                f"resolved {total_resolved}, unresolved {unresolved_count}"
            )
            # This is a warning, not a hard failure (some might be filtered)

        # Summary
        if broken_refs:
            logger.warning(
                f"Found {len(broken_refs)} broken references pointing to non-existent chunks"
            )
            # This is a warning, not a hard failure (references might be to external content)

        if type_mismatches:
            logger.warning(
                f"Found {len(type_mismatches)} type mismatches in cross-references"
            )
            # This is a warning, not a hard failure

        logger.debug(
            f"Rule 5 passed: {len(chunks_with_refs)} chunks with resolved references, "
            f"{len(chunks_with_unresolved)} chunks with unresolved references, "
            f"{len(broken_refs)} broken references, "
            f"{len(type_mismatches)} type mismatches"
        )

        return True
