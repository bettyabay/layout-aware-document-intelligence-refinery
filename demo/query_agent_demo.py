"""Demo script for the Query Interface Agent (Stage 5).

This script demonstrates how to:
1. Load processed documents (chunks, PageIndex)
2. Initialize vector store and fact table
3. Query documents using the Query Agent
4. Display answers with provenance
"""

import json
import logging
from pathlib import Path
from typing import Optional

from src.agents.query_agent import QueryAgent
from src.models.ldu import LDU
from src.utils.fact_table import FactTable
from src.utils.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_chunks_from_json(chunks_path: Path) -> list[LDU]:
    """Load LDUs from JSON file.

    Args:
        chunks_path: Path to chunks JSON file.

    Returns:
        List of LDU objects.
    """
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    return [LDU(**chunk) for chunk in chunks_data]


def setup_query_agent_for_document(
    doc_id: str,
    doc_name: str,
    chunks: list[LDU],
    pageindex_path: Optional[Path] = None,
) -> QueryAgent:
    """Set up query agent for a document.

    Args:
        doc_id: Document identifier.
        doc_name: Document name.
        chunks: List of LDUs.
        pageindex_path: Optional path to PageIndex JSON.

    Returns:
        Initialized QueryAgent.
    """
    # Initialize components
    vector_store_dir = Path(".refinery/vector_store")
    fact_table_path = Path(".refinery/facts.db")
    pageindex_dir = Path(".refinery/pageindex")

    # Create vector store and add chunks
    logger.info(f"Adding {len(chunks)} chunks to vector store")
    vector_store = VectorStore(persist_directory=vector_store_dir)
    vector_store.add_ldus(doc_id=doc_id, doc_name=doc_name, ldus=chunks)

    # Create fact table and extract facts
    logger.info("Extracting facts from chunks")
    fact_table = FactTable(db_path=fact_table_path)
    fact_count = fact_table.extract_facts_from_ldus(
        doc_id=doc_id, doc_name=doc_name, ldus=chunks
    )
    logger.info(f"Extracted {fact_count} facts")

    # Create query agent
    query_agent = QueryAgent(
        vector_store=vector_store,
        fact_table=fact_table,
        pageindex_dir=pageindex_dir,
        llm_api_key=None,  # Will read from env
    )

    return query_agent


def demo_query(
    query_agent: QueryAgent,
    query: str,
    doc_id: str,
    doc_name: str,
):
    """Execute a query and display results.

    Args:
        query_agent: QueryAgent instance.
        query: Query string.
        doc_id: Document identifier.
        doc_name: Document name.
    """
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")

    # Execute query
    result = query_agent.query(query=query, doc_id=doc_id, doc_name=doc_name)

    # Display answer
    print("Answer:")
    print(f"  {result['answer']}\n")

    # Display provenance
    if result.get("provenance_chain"):
        print("Provenance:")
        for i, prov in enumerate(result["provenance_chain"], 1):
            print(f"  [{i}] {prov.get('document_name', 'Unknown')}")
            print(f"      Page: {prov.get('page_number', 'N/A')}")
            bbox = prov.get("bbox", {})
            if bbox:
                print(
                    f"      Bounding Box: ({bbox.get('x0', 0):.1f}, {bbox.get('y0', 0):.1f}) "
                    f"to ({bbox.get('x1', 0):.1f}, {bbox.get('y1', 0):.1f})"
                )
            print(f"      Content Hash: {prov.get('content_hash', 'N/A')[:16]}...")
    else:
        print("No provenance information available.\n")


def main():
    """Main demo function."""
    # Example: Use a document that has been processed
    # You should replace these with actual paths from your processed documents
    doc_id = "2018_Audited_Financial_Statement_Report"
    doc_name = "2018_Audited_Financial_Statement_Report.pdf"
    chunks_path = Path("outputs/2018_Audited_Financial_Statement_Report_chunks.json")

    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        logger.info("Please run semantic chunking first to generate chunks.")
        return

    # Load chunks
    logger.info(f"Loading chunks from {chunks_path}")
    chunks = load_chunks_from_json(chunks_path)

    # Set up query agent
    query_agent = setup_query_agent_for_document(
        doc_id=doc_id, doc_name=doc_name, chunks=chunks
    )

    # Example queries
    queries = [
        "What is the total revenue?",
        "Find sections about financial statements",
        "SELECT * FROM facts WHERE fact_key LIKE '%revenue%'",
    ]

    for query in queries:
        demo_query(query_agent, query, doc_id, doc_name)

    print(f"\n{'='*80}")
    print("Demo complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
