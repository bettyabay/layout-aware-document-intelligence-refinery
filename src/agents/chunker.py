"""Semantic Chunking Engine.

This module implements the ChunkingEngine, which transforms raw extracted
documents (ExtractedDocument) into Logical Document Units (LDUs) that are
RAG-optimized and preserve structural context.

The chunking engine enforces five core rules:
1. No table cell split from header
2. Figure caption as metadata
3. Numbered lists intact
4. Section headers as parent metadata
5. Cross-reference resolution
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from src.models.extracted_document import ExtractedDocument
from src.models.ldu import CrossReference, LDU
from src.utils.chunk_validator import ChunkValidator
from src.utils.figure_chunker import FigureChunker
from src.utils.list_chunker import ListChunker
from src.utils.reference_resolver import ReferenceResolver
from src.utils.section_chunker import SectionChunker
from src.utils.table_chunker import TableChunker
from src.utils.token_counter import TokenCounter, count_tokens

logger = logging.getLogger(__name__)


class ChunkingEngine:
    """Semantic chunking engine that converts ExtractedDocument to LDUs.

    The ChunkingEngine applies semantic chunking rules to ensure that extracted
    content is split into coherent, RAG-friendly units while preserving structural
    relationships and spatial provenance.

    Attributes:
        config: Loaded chunking configuration from YAML.
        validator: ChunkValidator instance for enforcing chunking rules.
    """

    def __init__(self, config_path: str = "rubric/extraction_rules.yaml"):
        """Initialize the ChunkingEngine.

        Args:
            config_path: Path to the YAML configuration file containing chunking
                rules and thresholds.
        """
        self.config_path = Path(config_path)
        self.config = self.load_config()
        # Initialize validator with config and logging
        validation_config = self.config.get("validation", {})
        log_file = validation_config.get(
            "log_file", ".refinery/chunk_validation.log"
        )
        auto_fix = validation_config.get("auto_fix", False)
        self.validator = ChunkValidator(
            config=self.config, log_file=log_file, auto_fix=auto_fix
        )
        max_tokens = self.config.get("max_tokens_per_chunk", 512)
        preserve_lists = self.config.get("preserve_lists", True)
        list_split_strategy = self.config.get("list_split_strategy", "by_item")
        self.table_chunker = TableChunker(max_tokens_per_chunk=max_tokens)
        self.figure_chunker = FigureChunker()
        self.list_chunker = ListChunker(
            max_tokens_per_chunk=max_tokens,
            preserve_lists=preserve_lists,
            list_split_strategy=list_split_strategy,
        )
        self.section_chunker = SectionChunker()
        self.reference_resolver = ReferenceResolver()
        # Initialize token counter
        token_model = self.config.get("token_model", "gpt-4")
        self.token_counter = TokenCounter(model=token_model)

    def load_config(self) -> dict:
        """Load chunking rules from the configuration file.

        Returns:
            Dictionary containing chunking configuration. If the file doesn't
            exist or lacks a 'chunking' section, returns an empty dict.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            yaml.YAMLError: If the YAML is malformed.
        """
        if not self.config_path.exists():
            logger.warning(
                f"Config file not found at {self.config_path}. "
                "Using default chunking rules."
            )
            return {}

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f) or {}
                return full_config.get("chunking", {})
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
            raise

    def chunk(self, extracted_document: ExtractedDocument) -> List[LDU]:
        """Convert an ExtractedDocument into a list of LDUs.

        This is the main entry point for the chunking engine. It processes
        the extracted document according to the configured chunking rules and
        returns a list of semantically coherent chunks.

        Args:
            extracted_document: The normalized extracted document to chunk.

        Returns:
            List of LDU instances, each representing a semantically coherent
            chunk with spatial provenance and structural metadata.

        Raises:
            ValueError: If chunk validation fails after chunking.
        """
        logger.info(
            f"Chunking document with {len(extracted_document.text_blocks)} text blocks, "
            f"{len(extracted_document.tables)} tables, and {len(extracted_document.figures)} figures"
        )

        chunks: List[LDU] = []

        # Process figures first (to identify and pair captions)
        # This returns both figure chunks and a set of used caption block IDs
        figure_chunks, used_caption_blocks = self._chunk_figures(extracted_document)
        chunks.extend(figure_chunks)

        # Process text blocks (skipping those used as captions)
        chunks.extend(
            self._chunk_text_blocks(extracted_document, used_caption_blocks)
        )

        # Process tables as atomic chunks
        chunks.extend(self._chunk_tables(extracted_document))

        # Resolve cross-references using ReferenceResolver
        chunks, unresolved = self.reference_resolver.resolve_all_references(chunks)
        if unresolved:
            logger.warning(
                f"Found {sum(len(refs) for refs in unresolved.values())} "
                f"unresolved references in {len(unresolved)} chunks"
            )

        # Build section hierarchy and assign parent sections
        section_tree = self.section_chunker.build_section_hierarchy(chunks)
        chunks = self.section_chunker.assign_section_to_chunks(chunks, section_tree)

        # Validate all chunks against the enabled rules
        overall_success, results, violations = self.validate_chunks_detailed(chunks)

        # Check enforcement mode
        enforcement_mode = self.config.get("enforcement_mode", "strict")
        if not overall_success:
            if enforcement_mode == "strict":
                raise ValueError(
                    f"Chunk validation failed. Violations: {len(violations)}. "
                    f"Results: {results}"
                )
            elif enforcement_mode == "warn":
                logger.warning(
                    f"Chunk validation found {len(violations)} violations. "
                    f"Results: {results}"
                )
            # relaxed mode: just log, don't raise

        logger.info(f"Generated {len(chunks)} LDUs from document")
        return chunks

    def _chunk_text_blocks(
        self,
        extracted_document: ExtractedDocument,
        used_caption_blocks: set = None,
    ) -> List[LDU]:
        """Chunk text blocks into paragraphs, headers, and lists.

        Skips text blocks that are captions and have been paired with figures.
        Groups list items together before chunking.

        Args:
            extracted_document: The extracted document.
            used_caption_blocks: Set of text block IDs that are used as captions
                and should be skipped.

        Returns:
            List of LDUs from text blocks.
        """
        chunks = []
        max_tokens = self.config.get("max_tokens_per_chunk", 512)
        used_caption_blocks = used_caption_blocks or set()
        preserve_lists = self.config.get("preserve_lists", True)

        # Filter out used caption blocks
        available_blocks = [
            block
            for block in extracted_document.text_blocks
            if id(block) not in used_caption_blocks
        ]

        # Also filter out caption blocks that are paired with figures
        filtered_blocks = []
        for block in available_blocks:
            if self.figure_chunker._is_caption_text(block.content):
                # Check if there's a nearby figure that might use this as caption
                is_paired = False
                for figure in extracted_document.figures:
                    if self.figure_chunker._is_spatially_proximate(figure, block):
                        logger.debug(
                            f"Skipping caption text block on page {block.page_num} - "
                            f"likely paired with figure"
                        )
                        is_paired = True
                        break
                if not is_paired:
                    filtered_blocks.append(block)
            else:
                filtered_blocks.append(block)

        # Identify and group lists if preserve_lists is enabled
        if preserve_lists:
            list_items = self.list_chunker.identify_list_items(filtered_blocks)
            list_groups = self.list_chunker.group_list_items(list_items)

            # Track which blocks are part of lists
            list_block_ids = set()
            for group in list_groups:
                for block, _ in group:
                    list_block_ids.add(id(block))

            # Process lists
            for group in list_groups:
                if not group:
                    continue

                # Determine list type from first item
                _, first_info = group[0]
                list_type = first_info["list_type"]

                # Estimate total tokens for the list
                total_content = "\n".join(block.content for block, _ in group)
                total_tokens = self.token_counter.count(total_content)

                if total_tokens <= max_tokens:
                    # Create single LDU for entire list
                    ldu = self.list_chunker.merge_list_items(group, list_type)
                    chunks.append(ldu)
                else:
                    # Split large list at item boundaries
                    split_chunks = self.list_chunker.split_large_list(group, list_type)
                    chunks.extend(split_chunks)

            # Process non-list blocks
            non_list_blocks = [
                block for block in filtered_blocks if id(block) not in list_block_ids
            ]
        else:
            non_list_blocks = filtered_blocks

        # Process remaining text blocks (non-lists)
        for block in non_list_blocks:
            # Determine chunk type based on content heuristics
            chunk_type = self._classify_text_block(block.content)

            # Count tokens using token counter
            token_count = self.token_counter.count(block.content)

            # If content exceeds max_tokens, split intelligently
            if token_count > max_tokens:
                sub_chunks = self._split_large_block(
                    block, chunk_type, max_tokens, extracted_document
                )
                chunks.extend(sub_chunks)
            else:
                ldu = LDU(
                    content=block.content,
                    chunk_type=chunk_type,
                    page_refs=[block.page_num],
                    bounding_box={
                        "x0": block.bbox.x0,
                        "y0": block.bbox.y0,
                        "x1": block.bbox.x1,
                        "y1": block.bbox.y1,
                    },
                    token_count=token_count,
                )
                chunks.append(ldu)

        return chunks

    def _chunk_tables(self, extracted_document: ExtractedDocument) -> List[LDU]:
        """Chunk tables as atomic units using TableChunker.

        Tables are processed using specialized table chunking logic that ensures:
        - Headers are always preserved with data rows
        - Tables are kept as single LDUs when possible
        - Large tables are only split at logical boundaries (row boundaries),
          never between cells or between headers and data

        Args:
            extracted_document: The extracted document.

        Returns:
            List of table LDUs.
        """
        chunks = []
        max_tokens = self.config.get("max_tokens_per_chunk", 512)

        for table in extracted_document.tables:
            # Parse table structure
            structure = self.table_chunker.parse_table_structure(table)

            # Estimate token count for the entire table
            grouped_cells = self.table_chunker.group_table_cells(table)
            content = self.table_chunker._table_to_structured_text(
                table, grouped_cells
            )
            token_count = len(content) // 4

            # If table fits in one chunk, create single LDU
            if token_count <= max_tokens:
                ldu = self.table_chunker.create_table_chunk(table, structure)
                chunks.append(ldu)
            else:
                # Table is too large - split at logical boundaries
                logger.info(
                    f"Table on page {table.page_num} exceeds max_tokens "
                    f"({token_count} > {max_tokens}). Splitting at logical boundaries."
                )
                split_chunks = self.table_chunker.split_large_table(
                    table, structure
                )
                chunks.extend(split_chunks)

        return chunks

    def _chunk_figures(
        self, extracted_document: ExtractedDocument
    ) -> Tuple[List[LDU], set]:
        """Chunk figures with captions stored as metadata using FigureChunker.

        This method ensures that:
        - Captions are found using spatial proximity and pattern matching
        - Captions are stored as metadata, never as separate chunks
        - Figures are paired with their captions before chunking

        Args:
            extracted_document: The extracted document.

        Returns:
            Tuple of (list of figure LDUs, set of used caption block IDs).
        """
        chunks = []
        used_caption_blocks = set()  # Track which text blocks are used as captions

        for figure in extracted_document.figures:
            # Find caption for this figure
            caption_block = self.figure_chunker.find_caption_for_figure(
                figure, extracted_document
            )

            # Create LDU with caption in metadata
            ldu = self.figure_chunker.pair_figure_with_caption(figure, caption_block)

            # Mark caption block as used (so it won't be chunked separately)
            if caption_block:
                used_caption_blocks.add(id(caption_block))

            chunks.append(ldu)

        return chunks, used_caption_blocks

    def _classify_text_block(self, content: str) -> str:
        """Classify a text block as paragraph, header, list, or footnote.

        Args:
            content: The text content to classify.

        Returns:
            Chunk type string.
        """
        content_stripped = content.strip()

        # Check for headers (typically short, all caps, or numbered)
        if len(content_stripped) < 100 and (
            content_stripped.isupper()
            or content_stripped.startswith(("#", "##", "###"))
            or any(
                content_stripped.startswith(f"{i}.")
                for i in range(1, 10)
            )
        ):
            return "header"

        # Check for lists (numbered or bulleted)
        if content_stripped.startswith(("-", "*", "•")) or any(
            content_stripped.startswith(f"{i}.") for i in range(1, 100)
        ):
            return "list"

        # Check for footnotes (typically at bottom, small text, or numbered)
        if content_stripped.startswith(("Footnote", "Note:", "Note ")):
            return "footnote"

        return "paragraph"

    def _split_large_block(
        self,
        block,
        chunk_type: str,
        max_tokens: int,
        extracted_document: ExtractedDocument = None,
    ) -> List[LDU]:
        """Split a large text block into smaller chunks with intelligent boundaries.

        Priority for splitting:
        1. Between sections (if section markers detected)
        2. Between paragraphs (double newlines)
        3. Between sentences (period, exclamation, question mark)

        NEVER splits within tables, figures, or lists (these should be handled
        by their respective chunkers).

        Args:
            block: The text block to split.
            chunk_type: The chunk type for the block.
            max_tokens: Maximum tokens per chunk.
            extracted_document: Optional extracted document for context.

        Returns:
            List of LDUs from the split block.
        """
        import re

        content = block.content
        split_strategy = self.config.get("split_strategy", "semantic")

        # Strategy: semantic (preferred) - tries sections > paragraphs > sentences
        if split_strategy == "semantic":
            chunks = self._split_semantic(
                content, block, chunk_type, max_tokens
            )
        elif split_strategy == "greedy":
            # Greedy: just fill up to max_tokens, split at any sentence boundary
            chunks = self._split_greedy(content, block, chunk_type, max_tokens)
        elif split_strategy == "balanced":
            # Balanced: prefer paragraphs, fall back to sentences
            chunks = self._split_balanced(content, block, chunk_type, max_tokens)
        else:
            # Default to semantic
            chunks = self._split_semantic(content, block, chunk_type, max_tokens)

        return chunks

    def _split_semantic(
        self, content: str, block, chunk_type: str, max_tokens: int
    ) -> List[LDU]:
        """Split content using semantic boundaries (sections > paragraphs > sentences).

        Args:
            content: Text content to split.
            block: Original text block.
            chunk_type: Chunk type.
            max_tokens: Maximum tokens per chunk.

        Returns:
            List of LDUs.
        """
        chunks = []

        # First, try splitting by sections (double newlines with potential headers)
        # Pattern: \n\n followed by potential section markers
        section_pattern = r"\n\n+(?=\d+\.\s+[A-Z]|##+|Chapter|Section|Part\s+\d+)"
        sections = re.split(section_pattern, content)

        if len(sections) > 1:
            # We have sections, split at section boundaries
            for section in sections:
                section = section.strip()
                if not section:
                    continue

                section_tokens = self.token_counter.count(section)
                if section_tokens <= max_tokens:
                    chunks.append(
                        self._create_chunk_from_block(
                            section, block, chunk_type, section_tokens
                        )
                    )
                else:
                    # Section too large, split by paragraphs
                    sub_chunks = self._split_by_paragraphs(
                        section, block, chunk_type, max_tokens
                    )
                    chunks.extend(sub_chunks)
        else:
            # No clear sections, try paragraphs
            chunks = self._split_by_paragraphs(content, block, chunk_type, max_tokens)

        return chunks

    def _split_by_paragraphs(
        self, content: str, block, chunk_type: str, max_tokens: int
    ) -> List[LDU]:
        """Split content by paragraphs (double newlines).

        Args:
            content: Text content to split.
            block: Original text block.
            chunk_type: Chunk type.
            max_tokens: Maximum tokens per chunk.

        Returns:
            List of LDUs.
        """
        chunks = []
        paragraphs = re.split(r"\n\n+", content)

        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.token_counter.count(para)

            if para_tokens > max_tokens:
                # Paragraph itself is too large, split by sentences
                if current_chunk:
                    chunks.append(
                        self._create_chunk_from_block(
                            current_chunk.strip(), block, chunk_type, current_tokens
                        )
                    )
                    current_chunk = ""
                    current_tokens = 0

                sub_chunks = self._split_by_sentences(
                    para, block, chunk_type, max_tokens
                )
                chunks.extend(sub_chunks)
            elif current_tokens + para_tokens > max_tokens and current_chunk:
                # Adding this paragraph would exceed limit, finalize current chunk
                chunks.append(
                    self._create_chunk_from_block(
                        current_chunk.strip(), block, chunk_type, current_tokens
                    )
                )
                current_chunk = para
                current_tokens = para_tokens
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_tokens += para_tokens

        # Add remaining content
        if current_chunk.strip():
            chunks.append(
                self._create_chunk_from_block(
                    current_chunk.strip(), block, chunk_type, current_tokens
                )
            )

        return chunks

    def _split_by_sentences(
        self, content: str, block, chunk_type: str, max_tokens: int
    ) -> List[LDU]:
        """Split content by sentences (period, exclamation, question mark).

        Args:
            content: Text content to split.
            block: Original text block.
            chunk_type: Chunk type.
            max_tokens: Maximum tokens per chunk.

        Returns:
            List of LDUs.
        """
        chunks = []
        # Split on sentence boundaries, preserving the punctuation
        sentences = re.split(r"([.!?]\s+)", content)

        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            if not sentence.strip():
                continue

            sentence_tokens = self.token_counter.count(sentence)

            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Create chunk from accumulated content
                chunks.append(
                    self._create_chunk_from_block(
                        current_chunk.strip(), block, chunk_type, current_tokens
                    )
                )
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += sentence
                current_tokens += sentence_tokens

        # Add remaining content
        if current_chunk.strip():
            chunks.append(
                self._create_chunk_from_block(
                    current_chunk.strip(), block, chunk_type, current_tokens
                )
            )

        return chunks

    def _split_greedy(
        self, content: str, block, chunk_type: str, max_tokens: int
    ) -> List[LDU]:
        """Greedy splitting: fill up to max_tokens, split at sentence boundaries.

        Args:
            content: Text content to split.
            block: Original text block.
            chunk_type: Chunk type.
            max_tokens: Maximum tokens per chunk.

        Returns:
            List of LDUs.
        """
        return self._split_by_sentences(content, block, chunk_type, max_tokens)

    def _split_balanced(
        self, content: str, block, chunk_type: str, max_tokens: int
    ) -> List[LDU]:
        """Balanced splitting: prefer paragraphs, fall back to sentences.

        Args:
            content: Text content to split.
            block: Original text block.
            chunk_type: Chunk type.
            max_tokens: Maximum tokens per chunk.

        Returns:
            List of LDUs.
        """
        return self._split_by_paragraphs(content, block, chunk_type, max_tokens)

    def _create_chunk_from_block(
        self, content: str, block, chunk_type: str, token_count: int
    ) -> LDU:
        """Create an LDU from a text block.

        Args:
            content: Chunk content.
            block: Original text block.
            chunk_type: Chunk type.
            token_count: Token count for the chunk.

        Returns:
            LDU instance.
        """
        return LDU(
            content=content,
            chunk_type=chunk_type,
            page_refs=[block.page_num],
            bounding_box={
                "x0": block.bbox.x0,
                "y0": block.bbox.y0,
                "x1": block.bbox.x1,
                "y1": block.bbox.y1,
            },
            token_count=token_count,
        )

    def _table_to_text(self, table) -> str:
        """Convert a table structure to a readable text representation.

        Args:
            table: The table object.

        Returns:
            Text representation of the table.
        """
        lines = []
        # Header row
        lines.append(" | ".join(table.headers))
        lines.append("-" * (sum(len(h) for h in table.headers) + len(table.headers) * 3))
        # Data rows
        for row in table.rows:
            lines.append(" | ".join(str(cell) for cell in row))
        return "\n".join(lines)



    def validate_chunks(self, chunks: List[LDU]) -> bool:
        """Validate chunks against all enabled rules (simple interface).

        Args:
            chunks: List of LDUs to validate.

        Returns:
            True if all validation rules pass, False otherwise.
        """
        return self.validator.validate_all_simple(chunks)

    def validate_chunks_detailed(
        self, chunks: List[LDU]
    ) -> Tuple[bool, Dict[str, bool], List[Dict]]:
        """Validate chunks against all enabled rules with detailed results.

        Args:
            chunks: List of LDUs to validate.

        Returns:
            Tuple of:
            - overall_success: True if all enabled rules pass
            - results: Dictionary mapping rule names to pass/fail status
            - violations: List of violation dictionaries
        """
        return self.validator.validate_all(chunks)
