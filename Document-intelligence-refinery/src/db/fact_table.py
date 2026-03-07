"""Fact table extractor with SQLite backend."""

import sqlite3
from pathlib import Path


def init_fact_table(db_path: str | Path) -> None:
    """Initialize fact table in SQLite database."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            fact_key TEXT NOT NULL,
            fact_value TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            content_hash TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def upsert_fact(
    db_path: str | Path,
    doc_id: str,
    fact_key: str,
    fact_value: str,
    page_number: int,
    content_hash: str,
) -> None:
    """Upsert a fact into the fact table."""
    init_fact_table(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO facts (doc_id, fact_key, fact_value, page_number, content_hash) VALUES (?, ?, ?, ?, ?)",
        (doc_id, fact_key, fact_value, page_number, content_hash),
    )
    conn.commit()
    conn.close()


def structured_query(db_path: str | Path, doc_ids: list[str], key: str) -> list[dict]:
    """Query facts by key."""
    init_fact_table(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    if doc_ids:
        placeholders = ",".join(["?"] * len(doc_ids))
        query = f"SELECT * FROM facts WHERE doc_id IN ({placeholders}) AND fact_key = ?"
        params = tuple(doc_ids) + (key,)
    else:
        query = "SELECT * FROM facts WHERE fact_key = ?"
        params = (key,)
    
    cursor = conn.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]
