"""FactTable extractor for structured data queries.

This module extracts key-value facts from documents (especially financial/numerical)
and stores them in a SQLite database for precise querying.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.models.ldu import LDU
from src.models.provenance import ProvenanceChain

logger = logging.getLogger(__name__)


class FactTable:
    """SQLite-based fact table for structured queries.

    Extracts and stores key-value facts from documents, especially financial
    and numerical data, for precise SQL queries.
    """

    def __init__(self, db_path: Path):
        """Initialize the fact table.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database schema
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    doc_name TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    fact_type TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    bbox TEXT,
                    content_hash TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for faster queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON facts(doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fact_key ON facts(fact_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fact_type ON facts(fact_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_page_number ON facts(page_number)")

            conn.commit()

    def extract_facts_from_ldus(
        self,
        doc_id: str,
        doc_name: str,
        ldus: List[LDU],
    ) -> int:
        """Extract facts from LDUs and store them in the database.

        Args:
            doc_id: Document identifier.
            doc_name: Document name.
            ldus: List of LDUs to extract facts from.

        Returns:
            Number of facts extracted.
        """
        facts = []

        for ldu in ldus:
            # Extract facts based on chunk type
            if ldu.chunk_type == "table":
                table_facts = self._extract_table_facts(ldu, doc_id, doc_name)
                facts.extend(table_facts)
            elif ldu.chunk_type == "paragraph":
                paragraph_facts = self._extract_paragraph_facts(ldu, doc_id, doc_name)
                facts.extend(paragraph_facts)

        # Store facts in database
        if facts:
            self._store_facts(facts)

        logger.info(f"Extracted {len(facts)} facts from {len(ldus)} LDUs for document {doc_id}")
        return len(facts)

    def _extract_table_facts(
        self, ldu: LDU, doc_id: str, doc_name: str
    ) -> List[Dict[str, Any]]:
        """Extract facts from a table chunk.

        Args:
            ldu: Table LDU.
            doc_id: Document identifier.
            doc_name: Document name.

        Returns:
            List of fact dictionaries.
        """
        facts = []

        # Try to parse table structure from metadata
        table_headers = ldu.metadata.get("table_headers", [])
        table_rows = ldu.metadata.get("table_rows", [])

        if not table_headers or not table_rows:
            # Try to parse from content (markdown table format)
            lines = ldu.content.split("\n")
            if len(lines) > 2 and "|" in lines[0]:
                # Parse markdown table
                headers = [h.strip() for h in lines[0].split("|") if h.strip()]
                for row_line in lines[2:]:  # Skip header and separator
                    if "|" in row_line:
                        values = [v.strip() for v in row_line.split("|") if v.strip()]
                        for header, value in zip(headers, values):
                            if value and value != "-":
                                facts.append({
                                    "doc_id": doc_id,
                                    "doc_name": doc_name,
                                    "fact_key": header,
                                    "fact_value": value,
                                    "fact_type": "table_cell",
                                    "page_number": min(ldu.page_refs),
                                    "bbox": json.dumps(ldu.bounding_box),
                                    "content_hash": ldu.content_hash,
                                    "metadata": json.dumps({"row_index": len(facts)}),
                                })
        else:
            # Use structured table data
            for row_idx, row in enumerate(table_rows):
                for col_idx, (header, value) in enumerate(zip(table_headers, row)):
                    if value and str(value).strip():
                        facts.append({
                            "doc_id": doc_id,
                            "doc_name": doc_name,
                            "fact_key": header,
                            "fact_value": str(value),
                            "fact_type": "table_cell",
                            "page_number": min(ldu.page_refs),
                            "bbox": json.dumps(ldu.bounding_box),
                            "content_hash": ldu.content_hash,
                            "metadata": json.dumps({
                                "row_index": row_idx,
                                "col_index": col_idx,
                            }),
                        })

        return facts

    def _extract_paragraph_facts(
        self, ldu: LDU, doc_id: str, doc_name: str
    ) -> List[Dict[str, Any]]:
        """Extract facts from a paragraph chunk.

        Looks for key-value patterns like "Revenue: $4.2B" or "Date: Q3 2024".

        Args:
            ldu: Paragraph LDU.
            doc_id: Document identifier.
            doc_name: Document name.

        Returns:
            List of fact dictionaries.
        """
        facts = []

        # Pattern for key-value pairs
        patterns = [
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*[:=]\s*([^\n]+)",
            r"([A-Z][A-Z\s]+)\s*[:=]\s*([^\n]+)",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, ldu.content)
            for match in matches:
                key = match.group(1).strip()
                value = match.group(2).strip()

                # Filter out common false positives
                if len(key) > 2 and len(value) > 1 and key.lower() not in ["the", "a", "an"]:
                    facts.append({
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "fact_key": key,
                        "fact_value": value,
                        "fact_type": "key_value",
                        "page_number": min(ldu.page_refs),
                        "bbox": json.dumps(ldu.bounding_box),
                        "content_hash": ldu.content_hash,
                        "metadata": json.dumps({}),
                    })

        # Extract numerical facts (currency, percentages, dates)
        currency_pattern = r"(\$|USD|EUR|GBP)\s*([\d,]+\.?\d*)\s*(?:million|billion|thousand|M|B|K)?"
        for match in re.finditer(currency_pattern, ldu.content, re.IGNORECASE):
            currency = match.group(1)
            amount = match.group(2)
            facts.append({
                "doc_id": doc_id,
                "doc_name": doc_name,
                "fact_key": "Amount",
                "fact_value": f"{currency}{amount}",
                "fact_type": "currency",
                "page_number": min(ldu.page_refs),
                "bbox": json.dumps(ldu.bounding_box),
                "content_hash": ldu.content_hash,
                "metadata": json.dumps({}),
            })

        return facts

    def _store_facts(self, facts: List[Dict[str, Any]]) -> None:
        """Store facts in the database.

        Args:
            facts: List of fact dictionaries.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT INTO facts (
                    doc_id, doc_name, fact_key, fact_value, fact_type,
                    page_number, bbox, content_hash, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    fact["doc_id"],
                    fact["doc_name"],
                    fact["fact_key"],
                    fact["fact_value"],
                    fact["fact_type"],
                    fact["page_number"],
                    fact.get("bbox", "{}"),
                    fact["content_hash"],
                    fact.get("metadata", "{}"),
                )
                for fact in facts
            ])
            conn.commit()

    def query(
        self,
        sql_query: str,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a SQL query on the fact table.

        Args:
            sql_query: SQL query string.
            doc_id: Optional document ID to filter results.

        Returns:
            List of result dictionaries.

        Raises:
            sqlite3.Error: If the SQL query is invalid.
        """
        # Add WHERE clause if doc_id is specified
        if doc_id and "WHERE" not in sql_query.upper():
            sql_query = f"{sql_query} WHERE doc_id = '{doc_id}'"
        elif doc_id:
            sql_query = f"{sql_query} AND doc_id = '{doc_id}'"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql_query)
            results = [dict(row) for row in cursor.fetchall()]

        return results

    def get_provenance_for_fact(
        self, fact_id: int
    ) -> Optional[ProvenanceChain]:
        """Get provenance information for a fact.

        Args:
            fact_id: Fact ID.

        Returns:
            ProvenanceChain if found, None otherwise.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM facts WHERE id = ?", (fact_id,)
            )
            row = cursor.fetchone()

            if row:
                bbox = json.loads(row["bbox"]) if row["bbox"] else {}
                return ProvenanceChain(
                    document_name=row["doc_name"],
                    page_number=row["page_number"],
                    bbox=bbox,
                    content_hash=row["content_hash"],
                    verification_status=False,
                )

        return None

    def delete_document(self, doc_id: str) -> None:
        """Delete all facts for a document.

        Args:
            doc_id: Document identifier.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM facts WHERE doc_id = ?", (doc_id,))
            conn.commit()
        logger.info(f"Deleted all facts for document {doc_id}")


__all__ = ["FactTable"]
