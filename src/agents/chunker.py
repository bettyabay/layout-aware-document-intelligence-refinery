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
from typing import List

import yaml

from src.models.extracted_document import ExtractedDocument
from src.models.ldu import CrossReference, LDU
from src.utils.chunk_validator import ChunkValidator
from src.utils.table_chunker import TableChunker

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
        self.validator = ChunkValidator()
        max_tokens = self.config.get("max_tokens_per_chunk", 512)
        self.table_chunker = TableChunker(max_tokens_per_chunk=max_tokens)

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

        # Process text blocks into paragraph/header/list chunks
        chunks.extend(self._chunk_text_blocks(extracted_document))

        # Process tables as atomic chunks
        chunks.extend(self._chunk_tables(extracted_document))

        # Process figures with captions
        chunks.extend(self._chunk_figures(extracted_document))

        # Resolve cross-references
        chunks = self._resolve_cross_references(chunks)

        # Assign parent sections
        chunks = self._assign_parent_sections(chunks)

        # Validate all chunks against the five core rules
        if not self.validate_chunks(chunks):
            raise ValueError(
                "Chunk validation failed. One or more chunking rules were violated."
            )

        logger.info(f"Generated {len(chunks)} LDUs from document")
        return chunks

    def _chunk_text_blocks(
        self, extracted_document: ExtractedDocument
    ) -> List[LDU]:
        """Chunk text blocks into paragraphs, headers, and lists.

        Args:
            extracted_document: The extracted document.

        Returns:
            List of LDUs from text blocks.
        """
        chunks = []
        max_tokens = self.config.get("max_tokens_per_chunk", 512)

        for block in extracted_document.text_blocks:
            # Determine chunk type based on content heuristics
            chunk_type = self._classify_text_block(block.content)

            # Estimate token count (rough: ~4 chars per token)
            token_count = len(block.content) // 4

            # If content exceeds max_tokens, split intelligently
            if token_count > max_tokens:
                sub_chunks = self._split_large_block(block, chunk_type, max_tokens)
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

    def _chunk_figures(self, extracted_document: ExtractedDocument) -> List[LDU]:
        """Chunk figures with captions stored as metadata.

        Args:
            extracted_document: The extracted document.

        Returns:
            List of figure LDUs.
        """
        chunks = []

        for figure in extracted_document.figures:
            # Figures have minimal text content (just the caption if available)
            content = figure.caption or "[Figure]"

            ldu = LDU(
                content=content,
                chunk_type="figure",
                page_refs=[figure.page_num],
                bounding_box={
                    "x0": figure.bbox.x0,
                    "y0": figure.bbox.y0,
                    "x1": figure.bbox.x1,
                    "y1": figure.bbox.y1,
                },
                token_count=len(content) // 4,
                metadata={
                    "caption": figure.caption,
                    "has_caption": bool(figure.caption),
                },
            )
            chunks.append(ldu)

        return chunks

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
        self, block, chunk_type: str, max_tokens: int
    ) -> List[LDU]:
        """Split a large text block into smaller chunks.

        This method attempts to split on sentence boundaries to preserve
        semantic coherence.

        Args:
            block: The text block to split.
            chunk_type: The chunk type for the block.
            max_tokens: Maximum tokens per chunk.

        Returns:
            List of LDUs from the split block.
        """
        import re

        # Split on sentence boundaries (period, exclamation, question mark)
        sentences = re.split(r"([.!?]\s+)", block.content)
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(sentence) // 4
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Create chunk from accumulated content
                ldu = LDU(
                    content=current_chunk.strip(),
                    chunk_type=chunk_type,
                    page_refs=[block.page_num],
                    bounding_box={
                        "x0": block.bbox.x0,
                        "y0": block.bbox.y0,
                        "x1": block.bbox.x1,
                        "y1": block.bbox.y1,
                    },
                    token_count=current_tokens,
                )
                chunks.append(ldu)
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += sentence
                current_tokens += sentence_tokens

        # Add remaining content
        if current_chunk.strip():
            ldu = LDU(
                content=current_chunk.strip(),
                chunk_type=chunk_type,
                page_refs=[block.page_num],
                bounding_box={
                    "x0": block.bbox.x0,
                    "y0": block.bbox.y0,
                    "x1": block.bbox.x1,
                    "y1": block.bbox.y1,
                },
                token_count=current_tokens,
            )
            chunks.append(ldu)

        return chunks

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

    def _resolve_cross_references(self, chunks: List[LDU]) -> List[LDU]:
        """Resolve cross-references in chunk content.

        Looks for patterns like "see Table 3" or "Figure 2" and attempts to
        link them to the corresponding chunks.

        Args:
            chunks: List of chunks to process.

        Returns:
            Updated chunks with cross_references populated.
        """
        import re

        # Build index of chunks by type and content
        table_chunks = {i: c for i, c in enumerate(chunks) if c.chunk_type == "table"}
        figure_chunks = {
            i: c for i, c in enumerate(chunks) if c.chunk_type == "figure"
        }

        # Pattern to match references like "Table 3", "Figure 2", "see Table 3"
        table_pattern = re.compile(
            r"(?:see\s+)?(?:table|Table)\s+(\d+)", re.IGNORECASE
        )
        figure_pattern = re.compile(
            r"(?:see\s+)?(?:figure|Figure|Fig\.?)\s+(\d+)", re.IGNORECASE
        )

        for chunk in chunks:
            if chunk.chunk_type in ("table", "figure"):
                continue  # Skip tables and figures themselves

            # Find table references
            for match in table_pattern.finditer(chunk.content):
                table_num = int(match.group(1))
                # Try to find matching table (simplified: by order)
                if table_num <= len(table_chunks):
                    table_idx = list(table_chunks.keys())[table_num - 1]
                    target_chunk = chunks[table_idx]
                    chunk.cross_references.append(
                        CrossReference(
                            target_id=target_chunk.content_hash,
                            reference_type="table",
                            anchor_text=match.group(0),
                        )
                    )

            # Find figure references
            for match in figure_pattern.finditer(chunk.content):
                figure_num = int(match.group(1))
                if figure_num <= len(figure_chunks):
                    figure_idx = list(figure_chunks.keys())[figure_num - 1]
                    target_chunk = chunks[figure_idx]
                    chunk.cross_references.append(
                        CrossReference(
                            target_id=target_chunk.content_hash,
                            reference_type="figure",
                            anchor_text=match.group(0),
                        )
                    )

        return chunks

    def _assign_parent_sections(self, chunks: List[LDU]) -> List[LDU]:
        """Assign parent section titles to chunks based on spatial proximity.

        Headers are identified, and all subsequent chunks until the next header
        are assigned that header as their parent_section.

        Args:
            chunks: List of chunks to process.

        Returns:
            Updated chunks with parent_section populated.
        """
        current_section: str | None = None

        # Sort chunks by page and then by vertical position (y0)
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (
                min(c.page_refs),
                c.bounding_box.get("y0", 0),
            ),
        )

        for chunk in sorted_chunks:
            if chunk.chunk_type == "header":
                current_section = chunk.content
                chunk.parent_section = None  # Headers don't have parent sections
            else:
                chunk.parent_section = current_section

        return sorted_chunks

    def validate_chunks(self, chunks: List[LDU]) -> bool:
        """Validate chunks against all five core chunking rules.

        Args:
            chunks: List of LDUs to validate.

        Returns:
            True if all validation rules pass, False otherwise.
        """
        return self.validator.validate_all(chunks)
