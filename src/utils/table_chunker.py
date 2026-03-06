"""Table Chunker for Rule 1: Table Integrity.

This module implements specialized table chunking logic to ensure that tables
are never split in ways that separate headers from data rows or cells from
their context.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from src.models.extracted_document import Table
from src.models.ldu import LDU
from src.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)


class TableChunker:
    """Specialized chunker for table structures.

    Ensures that tables are chunked as atomic units, with headers always
    preserved with their data rows. If a table is too large, it can be split
    at logical boundaries (e.g., section breaks) but never between individual
    cells.
    """

    def __init__(self, max_tokens_per_chunk: int = 512):
        """Initialize the TableChunker.

        Args:
            max_tokens_per_chunk: Maximum tokens per chunk before considering
                splitting. Even if exceeded, tables are only split at logical
                boundaries.
        """
        self.max_tokens_per_chunk = max_tokens_per_chunk

    def parse_table_structure(self, table: Table) -> Dict:
        """Parse and identify the structure of a table.

        Identifies:
        - Header rows (typically the first row)
        - Data rows
        - Any special rows (totals, subtotals, etc.)

        Args:
            table: The table to parse.

        Returns:
            Dictionary with structure information:
            - 'header_rows': List of header row indices
            - 'data_rows': List of data row indices
            - 'special_rows': List of special row indices (totals, etc.)
            - 'column_count': Number of columns
        """
        structure = {
            "header_rows": [0] if table.headers else [],
            "data_rows": list(range(len(table.rows))),
            "special_rows": [],
            "column_count": len(table.headers) if table.headers else 0,
        }

        # Identify special rows (rows that look like totals or subtotals)
        for i, row in enumerate(table.rows):
            # Check if row contains summary indicators
            row_text = " ".join(str(cell) for cell in row).lower()
            if any(
                indicator in row_text
                for indicator in ["total", "subtotal", "sum", "grand total"]
            ):
                structure["special_rows"].append(i)

        return structure

    def group_table_cells(self, table: Table) -> List[List[str]]:
        """Group all cells of a table together, preserving structure.

        This ensures that headers and data rows are kept together as a single
        logical unit.

        Args:
            table: The table to group.

        Returns:
            List of grouped rows, where each row is a list of cell values.
            Headers are included as the first row(s).
        """
        grouped = []

        # Add headers as first row if present
        if table.headers:
            grouped.append(table.headers)

        # Add all data rows
        for row in table.rows:
            grouped.append(row)

        return grouped

    def create_table_chunk(
        self, table: Table, structure: Optional[Dict] = None
    ) -> LDU:
        """Create a single LDU for an entire table.

        This is the primary method for creating table chunks. It ensures that
        the entire table (headers + all rows) is kept as a single atomic unit.

        Args:
            table: The table to chunk.
            structure: Optional pre-parsed table structure. If not provided,
                it will be parsed automatically.

        Returns:
            A single LDU representing the entire table.
        """
        if structure is None:
            structure = self.parse_table_structure(table)

        # Group all cells together
        grouped_cells = self.group_table_cells(table)

        # Convert to structured text representation
        content = self._table_to_structured_text(table, grouped_cells)

        # Estimate token count using token counter
        token_count = count_tokens(content)

        # Create metadata with table structure information
        metadata = {
            "table_headers": table.headers,
            "row_count": len(table.rows),
            "column_count": structure["column_count"],
            "has_headers": bool(table.headers),
            "header_row_count": len(structure["header_rows"]),
            "special_row_count": len(structure["special_rows"]),
            "table_structure": {
                "header_rows": structure["header_rows"],
                "special_rows": structure["special_rows"],
            },
        }

        ldu = LDU(
            content=content,
            chunk_type="table",
            page_refs=[table.page_num],
            bounding_box={
                "x0": table.bbox.x0,
                "y0": table.bbox.y0,
                "x1": table.bbox.x1,
                "y1": table.bbox.y1,
            },
            token_count=token_count,
            metadata=metadata,
        )

        return ldu

    def split_large_table(
        self, table: Table, structure: Optional[Dict] = None
    ) -> List[LDU]:
        """Split a large table at logical boundaries.

        This method should only be called if a table exceeds max_tokens_per_chunk.
        It splits the table at logical boundaries (e.g., section breaks, group
        boundaries) but NEVER between individual cells or between headers and
        their data rows.

        Args:
            table: The table to split.
            structure: Optional pre-parsed table structure.

        Returns:
            List of LDUs, each representing a logical section of the table.
            Each LDU includes headers for context.
        """
        if structure is None:
            structure = self.parse_table_structure(table)

        # Group all cells
        grouped_cells = self.group_table_cells(table)
        headers = table.headers if table.headers else []

        # Estimate tokens per row (rough estimate)
        if grouped_cells:
            sample_row_text = " | ".join(str(cell) for cell in grouped_cells[0])
            tokens_per_row = count_tokens(sample_row_text)
        else:
            tokens_per_row = 10  # Default estimate

        # Calculate how many rows we can fit per chunk (including headers)
        header_tokens = count_tokens(" | ".join(headers)) if headers else 0
        available_tokens = self.max_tokens_per_chunk - header_tokens
        rows_per_chunk = max(1, available_tokens // max(tokens_per_row, 1))

        chunks = []
        current_chunk_rows = []

        # Always start with headers
        if headers:
            current_chunk_rows.append(headers)

        # Split data rows into chunks
        for i, row in enumerate(table.rows):
            current_chunk_rows.append(row)

            # Check if we've reached the chunk size limit
            # Only split at row boundaries, never mid-row
            if len(current_chunk_rows) > rows_per_chunk + len(
                headers
            ):  # +1 for header
                # Create chunk with accumulated rows
                chunk_content = self._rows_to_text(current_chunk_rows[:-1])
                chunk_metadata = {
                    "table_headers": headers,
                    "row_count": len(current_chunk_rows) - len(headers),
                    "column_count": structure["column_count"],
                    "has_headers": bool(headers),
                    "is_partial_table": True,
                    "chunk_index": len(chunks),
                    "total_chunks": None,  # Will be updated later
                }

                chunk_ldu = LDU(
                    content=chunk_content,
                    chunk_type="table",
                    page_refs=[table.page_num],
                    bounding_box={
                        "x0": table.bbox.x0,
                        "y0": table.bbox.y0,
                        "x1": table.bbox.x1,
                        "y1": table.bbox.y1,
                    },
                    token_count=count_tokens(chunk_content),
                    metadata=chunk_metadata,
                )
                chunks.append(chunk_ldu)

                # Start new chunk with headers again
                current_chunk_rows = [headers] if headers else []
                current_chunk_rows.append(row)

        # Add remaining rows as final chunk
        if current_chunk_rows:
            chunk_content = self._rows_to_text(current_chunk_rows)
            chunk_metadata = {
                "table_headers": headers,
                "row_count": len(current_chunk_rows) - len(headers),
                "column_count": structure["column_count"],
                "has_headers": bool(headers),
                "is_partial_table": len(chunks) > 0,
                "chunk_index": len(chunks),
                "total_chunks": len(chunks) + 1,
            }

            # Update previous chunks' total_chunks
            for prev_chunk in chunks:
                if prev_chunk.metadata.get("is_partial_table"):
                    prev_chunk.metadata["total_chunks"] = len(chunks) + 1

            chunk_ldu = LDU(
                content=chunk_content,
                chunk_type="table",
                page_refs=[table.page_num],
                bounding_box={
                    "x0": table.bbox.x0,
                    "y0": table.bbox.y0,
                    "x1": table.bbox.x1,
                    "y1": table.bbox.y1,
                },
                token_count=count_tokens(chunk_content),
                metadata=chunk_metadata,
            )
            chunks.append(chunk_ldu)

        logger.info(
            f"Split large table into {len(chunks)} chunks at logical boundaries"
        )
        return chunks

    def _table_to_structured_text(
        self, table: Table, grouped_cells: List[List[str]]
    ) -> str:
        """Convert a table to structured text representation.

        Args:
            table: The table object.
            grouped_cells: Pre-grouped cells (headers + rows).

        Returns:
            Structured text representation of the table.
        """
        lines = []
        for row in grouped_cells:
            lines.append(" | ".join(str(cell) for cell in row))
        return "\n".join(lines)

    def _rows_to_text(self, rows: List[List[str]]) -> str:
        """Convert a list of rows to text representation.

        Args:
            rows: List of rows, where each row is a list of cell values.

        Returns:
            Text representation of the rows.
        """
        lines = []
        for row in rows:
            lines.append(" | ".join(str(cell) for cell in row))
        return "\n".join(lines)


def identify_table_structures(
    extracted_document,
) -> List[Tuple[Table, Dict]]:
    """Identify all table structures in an extracted document.

    This is a helper function that parses all tables and returns them with
    their structure information.

    Args:
        extracted_document: The ExtractedDocument to analyze.

    Returns:
        List of tuples (table, structure_dict) for each table.
    """
    chunker = TableChunker()
    structures = []
    for table in extracted_document.tables:
        structure = chunker.parse_table_structure(table)
        structures.append((table, structure))
    return structures
