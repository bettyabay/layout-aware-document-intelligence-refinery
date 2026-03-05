"""List Chunker for Rule 3: Numbered List Integrity.

This module implements specialized list chunking logic to ensure that numbered
and bulleted lists are kept as single LDUs unless they exceed max_tokens, and
when split, they are split at list item boundaries, never within items.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from src.models.extracted_document import TextBlock
from src.models.ldu import LDU

logger = logging.getLogger(__name__)

# Common list patterns
NUMBERED_LIST_PATTERNS = [
    r"^\d+[.)]\s+",  # "1. " or "1) "
    r"^[ivxlcdm]+[.)]\s+",  # Roman numerals: "i. ", "ii. ", "iv. "
    r"^[IVXLCDM]+[.)]\s+",  # Uppercase Roman: "I. ", "II. ", "IV. "
    r"^[a-z][.)]\s+",  # Lowercase letters: "a. ", "b. "
    r"^[A-Z][.)]\s+",  # Uppercase letters: "A. ", "B. "
]

BULLET_LIST_PATTERNS = [
    r"^[•·▪▫]\s+",  # Bullet characters
    r"^[-*+]\s+",  # Dash, asterisk, plus
    r"^○\s+",  # Circle
    r"^■\s+",  # Square
]


class ListChunker:
    """Specialized chunker for list structures.

    Ensures that lists are kept as single LDUs when possible, and when split,
    they are split at item boundaries, never within items.
    """

    def __init__(
        self,
        max_tokens_per_chunk: int = 512,
        preserve_lists: bool = True,
        list_split_strategy: str = "by_item",
    ):
        """Initialize the ListChunker.

        Args:
            max_tokens_per_chunk: Maximum tokens per chunk before splitting.
            preserve_lists: Whether to preserve lists as single units.
            list_split_strategy: Strategy for splitting large lists.
                Options: "by_item" (split between items) or "by_sublist" (split by sublists).
        """
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.preserve_lists = preserve_lists
        self.list_split_strategy = list_split_strategy

    def identify_list_items(
        self, text_blocks: List[TextBlock]
    ) -> List[Tuple[TextBlock, Dict]]:
        """Identify and group list items from text blocks.

        Args:
            text_blocks: List of text blocks to analyze.

        Returns:
            List of tuples (text_block, list_info_dict) where list_info_dict contains:
            - 'is_list_item': bool
            - 'list_type': 'numbered' | 'bulleted' | None
            - 'list_marker': str (the marker pattern found)
            - 'item_number': Optional[int] (for numbered lists)
        """
        list_items = []

        for block in text_blocks:
            content = block.content.strip()
            if not content:
                continue

            list_info = self._analyze_list_item(content)
            if list_info["is_list_item"]:
                list_items.append((block, list_info))

        return list_items

    def group_list_items(
        self, list_items: List[Tuple[TextBlock, Dict]]
    ) -> List[List[Tuple[TextBlock, Dict]]]:
        """Group consecutive list items that belong to the same list.

        Args:
            list_items: List of (text_block, list_info) tuples.

        Returns:
            List of groups, where each group is a list of items belonging to
            the same list.
        """
        if not list_items:
            return []

        groups = []
        current_group = [list_items[0]]

        for i in range(1, len(list_items)):
            prev_block, prev_info = list_items[i - 1]
            curr_block, curr_info = list_items[i]

            # Check if items belong to the same list
            if self._belongs_to_same_list(prev_block, prev_info, curr_block, curr_info):
                current_group.append(list_items[i])
            else:
                # Start a new group
                groups.append(current_group)
                current_group = [list_items[i]]

        # Add the last group
        if current_group:
            groups.append(current_group)

        return groups

    def merge_list_items(
        self, items: List[Tuple[TextBlock, Dict]], list_type: str
    ) -> LDU:
        """Merge a group of list items into a single LDU.

        Args:
            items: List of (text_block, list_info) tuples belonging to the same list.
            list_type: Type of list ('numbered' or 'bulleted').

        Returns:
            A single LDU representing the entire list.
        """
        # Combine all item content
        content_lines = []
        page_refs = set()
        bbox_coords = {"x0": float("inf"), "y0": float("inf"), "x1": 0.0, "y1": 0.0}

        for block, list_info in items:
            content_lines.append(block.content)
            page_refs.add(block.page_num)

            # Expand bounding box
            bbox_coords["x0"] = min(bbox_coords["x0"], block.bbox.x0)
            bbox_coords["y0"] = min(bbox_coords["y0"], block.bbox.y0)
            bbox_coords["x1"] = max(bbox_coords["x1"], block.bbox.x1)
            bbox_coords["y1"] = max(bbox_coords["y1"], block.bbox.y1)

        content = "\n".join(content_lines)
        token_count = len(content) // 4

        # Create metadata
        metadata = {
            "list_type": list_type,
            "item_count": len(items),
            "list_marker": items[0][1].get("list_marker", ""),
            "is_list": True,
        }

        # For numbered lists, include numbering info
        if list_type == "numbered":
            first_item_info = items[0][1]
            if "item_number" in first_item_info:
                metadata["starting_number"] = first_item_info["item_number"]

        ldu = LDU(
            content=content,
            chunk_type="list",
            page_refs=sorted(list(page_refs)),
            bounding_box=bbox_coords,
            token_count=token_count,
            metadata=metadata,
        )

        return ldu

    def split_large_list(
        self, items: List[Tuple[TextBlock, Dict]], list_type: str
    ) -> List[LDU]:
        """Split a large list at item boundaries.

        Args:
            items: List of (text_block, list_info) tuples.
            list_type: Type of list ('numbered' or 'bulleted').

        Returns:
            List of LDUs, each representing a portion of the list.
            Each LDU includes all items in that portion.
        """
        chunks = []

        if self.list_split_strategy == "by_item":
            # Split between items
            current_chunk_items = []
            current_tokens = 0

            for block, list_info in items:
                item_content = block.content
                item_tokens = len(item_content) // 4

                # Check if adding this item would exceed max_tokens
                if (
                    current_tokens + item_tokens > self.max_tokens_per_chunk
                    and current_chunk_items
                ):
                    # Create chunk from accumulated items
                    chunk = self.merge_list_items(current_chunk_items, list_type)
                    chunk.metadata["is_partial_list"] = True
                    chunk.metadata["chunk_index"] = len(chunks)
                    chunks.append(chunk)

                    # Start new chunk
                    current_chunk_items = [block, list_info]
                    current_tokens = item_tokens
                else:
                    current_chunk_items.append((block, list_info))
                    current_tokens += item_tokens

            # Add remaining items as final chunk
            if current_chunk_items:
                chunk = self.merge_list_items(current_chunk_items, list_type)
                if len(chunks) > 0:
                    chunk.metadata["is_partial_list"] = True
                    chunk.metadata["chunk_index"] = len(chunks)
                    chunk.metadata["total_chunks"] = len(chunks) + 1
                    # Update previous chunks' total_chunks
                    for prev_chunk in chunks:
                        prev_chunk.metadata["total_chunks"] = len(chunks) + 1
                chunks.append(chunk)

        else:  # by_sublist
            # Split by sublists (groups of related items)
            # This is a simplified implementation - could be enhanced
            # to detect sublist boundaries based on indentation or numbering
            chunks = self.split_large_list(items, list_type)  # Fallback to by_item

        logger.info(
            f"Split large {list_type} list into {len(chunks)} chunks at item boundaries"
        )

        return chunks

    def _analyze_list_item(self, content: str) -> Dict:
        """Analyze if a text block is a list item and extract list information.

        Args:
            content: Text content to analyze.

        Returns:
            Dictionary with list information:
            - 'is_list_item': bool
            - 'list_type': 'numbered' | 'bulleted' | None
            - 'list_marker': str (the marker pattern found)
            - 'item_number': Optional[int] (for numbered lists)
        """
        result = {
            "is_list_item": False,
            "list_type": None,
            "list_marker": "",
            "item_number": None,
        }

        # Check numbered list patterns
        for pattern in NUMBERED_LIST_PATTERNS:
            match = re.match(pattern, content, re.IGNORECASE)
            if match:
                result["is_list_item"] = True
                result["list_type"] = "numbered"
                result["list_marker"] = match.group(0)

                # Try to extract item number
                marker_text = match.group(0).strip()
                # Remove punctuation
                marker_clean = re.sub(r"[.)]\s*$", "", marker_text)
                try:
                    result["item_number"] = int(marker_clean)
                except ValueError:
                    # Might be Roman numeral or letter - skip for now
                    pass

                return result

        # Check bullet list patterns
        for pattern in BULLET_LIST_PATTERNS:
            match = re.match(pattern, content)
            if match:
                result["is_list_item"] = True
                result["list_type"] = "bulleted"
                result["list_marker"] = match.group(0)
                return result

        return result

    def _belongs_to_same_list(
        self,
        block1: TextBlock,
        info1: Dict,
        block2: TextBlock,
        info2: Dict,
    ) -> bool:
        """Check if two list items belong to the same list.

        Args:
            block1: First text block.
            info1: List info for first block.
            block2: Second text block.
            info2: List info for second block.

        Returns:
            True if items belong to the same list.
        """
        # Must be same list type
        if info1["list_type"] != info2["list_type"]:
            return False

        # Must be on same page or adjacent pages
        if abs(block1.page_num - block2.page_num) > 1:
            return False

        # For numbered lists, check if numbering is consecutive
        if info1["list_type"] == "numbered":
            num1 = info1.get("item_number")
            num2 = info2.get("item_number")

            if num1 is not None and num2 is not None:
                # Check if numbers are consecutive (within reasonable range)
                if abs(num2 - num1) > 10:  # Allow some gap for sub-items
                    return False

            # Check if markers are similar
            if info1["list_marker"] != info2["list_marker"]:
                # Might be different numbering style, but still same list
                # Check spatial proximity
                vertical_distance = abs(block1.bbox.y0 - block2.bbox.y0)
                if vertical_distance > 200:  # Too far apart
                    return False

        # For bulleted lists, check if markers match
        if info1["list_type"] == "bulleted":
            if info1["list_marker"] != info2["list_marker"]:
                # Check spatial proximity
                vertical_distance = abs(block1.bbox.y0 - block2.bbox.y0)
                if vertical_distance > 200:  # Too far apart
                    return False

        # Check spatial proximity (items should be close vertically)
        vertical_distance = abs(block1.bbox.y0 - block2.bbox.y0)
        if vertical_distance > 300:  # Too far apart
            return False

        return True


def identify_lists(text_blocks: List[TextBlock]) -> List[List[TextBlock]]:
    """Identify all lists in a collection of text blocks.

    Args:
        text_blocks: List of text blocks to analyze.

    Returns:
        List of lists, where each inner list contains text blocks belonging
        to the same list.
    """
    chunker = ListChunker()
    list_items = chunker.identify_list_items(text_blocks)
    groups = chunker.group_list_items(list_items)

    # Convert back to just text blocks
    result = []
    for group in groups:
        result.append([block for block, _ in group])

    return result
