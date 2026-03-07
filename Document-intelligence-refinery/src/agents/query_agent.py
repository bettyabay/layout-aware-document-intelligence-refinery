"""Query Interface Agent - Stage 5: LangGraph agent with provenance tracking."""

import json
from pathlib import Path
from typing import Optional

from src.db.fact_table import structured_query
from src.db.vector_store import VectorStore
from src.models import ExtractedDocument, PageIndexNode, ProvenanceChain
from src.utils.rules import load_rules


class QueryAgent:
    """Query agent with three tools: PageIndex navigation, semantic search, structured query."""

    def __init__(self, doc_id: str, rules: dict):
        self.doc_id = doc_id
        self.rules = rules
        self.vector_store = VectorStore()
        self.fact_db_path = Path(".refinery/facts.db")

    def query(
        self,
        query: str,
        use_pageindex: bool = True,
        use_semantic_search: bool = True,
        use_structured_query: bool = False,
        audit_mode: bool = True,
    ) -> dict:
        """Execute query with provenance tracking."""
        results = {
            "answer": "",
            "provenance": [],
            "trace": {},
        }

        # Tool 1: Semantic search
        if use_semantic_search:
            search_results = self.vector_store.search(query, doc_ids=[self.doc_id], k=5)
            results["trace"]["semantic_search"] = search_results
            
            if search_results:
                # Build answer from search results
                answer_parts = []
                for result in search_results[:3]:  # Top 3 results
                    answer_parts.append(result["text"])
                    
                    if audit_mode:
                        # Extract provenance from metadata
                        metadata = result.get("metadata", {})
                        page_refs = metadata.get("page_refs", "[]")
                        try:
                            pages = json.loads(page_refs) if isinstance(page_refs, str) else page_refs
                            if pages:
                                results["provenance"].append(
                                    ProvenanceChain(
                                        document_name=self.doc_id,
                                        page_number=pages[0] if pages else 1,
                                        bbox=None,  # Would need to fetch from LDU
                                        content_hash="",
                                        text_excerpt=result["text"][:200],
                                    )
                                )
                        except Exception:
                            pass
                
                results["answer"] = " ".join(answer_parts)

        # Tool 2: Structured query (for fact extraction)
        if use_structured_query:
            # Extract key from query (simplified)
            fact_key = self._extract_fact_key(query)
            if fact_key:
                fact_results = structured_query(self.fact_db_path, [self.doc_id], fact_key)
                results["trace"]["structured_query"] = fact_results
                
                if fact_results:
                    fact_values = [r["fact_value"] for r in fact_results]
                    results["answer"] += f"\n\nFact: {fact_key} = {', '.join(fact_values)}"

        # Tool 3: PageIndex navigation (simplified)
        if use_pageindex:
            pageindex_path = Path(f".refinery/pageindex/{self.doc_id}_pageindex.json")
            if pageindex_path.exists():
                with open(pageindex_path) as f:
                    pageindex_data = json.load(f)
                results["trace"]["pageindex"] = pageindex_data

        # If no answer generated, provide default
        if not results["answer"]:
            results["answer"] = "No relevant information found. Try rephrasing your query."

        return results

    def _extract_fact_key(self, query: str) -> Optional[str]:
        """Extract fact key from query (simplified)."""
        query_lower = query.lower()
        
        # Common financial fact keys
        financial_keys = ["revenue", "profit", "loss", "assets", "liabilities", "equity"]
        for key in financial_keys:
            if key in query_lower:
                return key
        
        return None
