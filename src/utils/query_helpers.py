"""Helper functions for initializing and using the Query Agent."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from src.agents.query_agent import QueryAgent
from src.utils.fact_table import FactTable
from src.utils.vector_store import VectorStore

logger = logging.getLogger(__name__)


def create_query_agent(
    vector_store_dir: Optional[Path] = None,
    fact_table_path: Optional[Path] = None,
    pageindex_dir: Optional[Path] = None,
    llm_api_key: Optional[str] = None,
    llm_model: str = "mistralai/mistral-7b-instruct",
) -> QueryAgent:
    """Create and initialize a QueryAgent with all dependencies.

    Args:
        vector_store_dir: Directory for ChromaDB persistence.
        fact_table_path: Path to SQLite fact table database.
        pageindex_dir: Directory containing PageIndex JSON files.
        llm_api_key: API key for LLM (OpenRouter). If None, reads from OPENROUTER_API_KEY env var.
        llm_model: LLM model to use.

    Returns:
        Initialized QueryAgent instance.
    """
    # Default paths
    if vector_store_dir is None:
        vector_store_dir = Path(".refinery/vector_store")
    if fact_table_path is None:
        fact_table_path = Path(".refinery/facts.db")
    if pageindex_dir is None:
        pageindex_dir = Path(".refinery/pageindex")

    # Get LLM API key from environment if not provided
    if llm_api_key is None:
        llm_api_key = os.getenv("OPENROUTER_API_KEY")

    # Initialize vector store
    logger.info(f"Initializing vector store at {vector_store_dir}")
    vector_store = VectorStore(
        persist_directory=vector_store_dir,
        embedding_model="all-MiniLM-L6-v2",
    )

    # Initialize fact table
    logger.info(f"Initializing fact table at {fact_table_path}")
    fact_table = FactTable(db_path=fact_table_path)

    # Create query agent
    logger.info("Creating QueryAgent")
    query_agent = QueryAgent(
        vector_store=vector_store,
        fact_table=fact_table,
        pageindex_dir=pageindex_dir,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )

    return query_agent


__all__ = ["create_query_agent"]
