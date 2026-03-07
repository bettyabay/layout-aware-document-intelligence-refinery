"""Query page - Natural language query interface with provenance."""
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from streamlit import session_state as ss

from src.agents.query_agent import QueryAgent
from src.models import ProvenanceChain


def show():
    """Display the Query page."""
    st.header("Stage 4: Query Interface Agent")
    st.markdown(
        """
        Natural language query interface with full provenance tracking.
        Uses LangGraph agent with three tools:
        - **PageIndex Navigation**: Tree traversal for section-specific queries
        - **Semantic Search**: Vector retrieval for content-based queries
        - **Structured Query**: SQL queries over fact tables
        """
    )

    # Check if document is selected
    if not ss.current_doc_id or ss.current_doc_id not in ss.documents:
        st.warning("⚠️ Please select a document from the sidebar or upload a new one.")
        return

    doc_id = ss.current_doc_id
    doc_info = ss.documents[doc_id]

    # Query input
    st.subheader("Ask a Question")
    query_text = st.text_area(
        "Enter your question about the document:",
        placeholder="e.g., What is the revenue for Q3 2024?",
        height=100
    )

    # Query options
    col1, col2 = st.columns(2)
    with col1:
        use_pageindex = st.checkbox("Use PageIndex Navigation", value=True)
        use_semantic_search = st.checkbox("Use Semantic Search", value=True)
    with col2:
        use_structured_query = st.checkbox("Use Structured Query", value=False)
        audit_mode = st.checkbox("Audit Mode (Show Provenance)", value=True)

    # Query button
    if st.button("🔍 Query", type="primary") and query_text:
        execute_query(doc_id, doc_info, query_text, {
            "use_pageindex": use_pageindex,
            "use_semantic_search": use_semantic_search,
            "use_structured_query": use_structured_query,
            "audit_mode": audit_mode,
        })

    # Display previous queries
    if "query_history" in ss and doc_id in ss.query_history:
        st.markdown("---")
        st.subheader("Query History")
        
        for i, query_result in enumerate(reversed(ss.query_history[doc_id][-5:]), 1):
            with st.expander(f"Query {i}: {query_result.get('query', 'N/A')[:50]}..."):
                st.markdown(f"**Answer:** {query_result.get('answer', 'N/A')}")
                
                if audit_mode and query_result.get("provenance"):
                    st.markdown("**Provenance:**")
                    display_provenance(query_result["provenance"])


def execute_query(doc_id: str, doc_info: dict, query: str, options: dict):
    """Execute a query using the Query Agent."""
    with st.spinner("Processing query..."):
        try:
            from src.utils.rules import load_rules
            
            # Load rules
            rules_path = Path("config/extraction_rules.yaml")
            if not rules_path.exists():
                rules_path = Path("rubric/extraction_rules.yaml")
            rules = load_rules(str(rules_path))
            
            # Initialize Query Agent
            query_agent = QueryAgent(doc_id=doc_id, rules=rules)
            
            # Execute query
            result = query_agent.query(
                query=query,
                use_pageindex=options["use_pageindex"],
                use_semantic_search=options["use_semantic_search"],
                use_structured_query=options["use_structured_query"],
                audit_mode=options["audit_mode"],
            )
            
            # Display results
            st.markdown("### Answer")
            st.markdown(result.get("answer", "No answer generated."))
            
            # Display provenance if available
            if options["audit_mode"] and result.get("provenance"):
                st.markdown("---")
                st.markdown("### Provenance Chain")
                display_provenance(result["provenance"])
            
            # Display query trace if available
            if result.get("trace"):
                with st.expander("🔍 Query Trace"):
                    st.json(result["trace"])
            
            # Store in history
            if "query_history" not in ss:
                ss.query_history = {}
            if doc_id not in ss.query_history:
                ss.query_history[doc_id] = []
            
            ss.query_history[doc_id].append({
                "query": query,
                "answer": result.get("answer", ""),
                "provenance": result.get("provenance", []),
                "trace": result.get("trace", {}),
            })
            
        except Exception as e:
            st.error(f"❌ Error executing query: {str(e)}")
            st.exception(e)


def display_provenance(provenance: list[dict]):
    """Display provenance chain."""
    for i, prov in enumerate(provenance, 1):
        with st.container():
            st.markdown(f"**Citation {i}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"- **Document:** {prov.get('document_name', 'N/A')}")
                st.markdown(f"- **Page:** {prov.get('page_number', 'N/A')}")
            with col2:
                bbox = prov.get('bbox', {})
                if bbox:
                    st.markdown(f"- **BBox:** ({bbox.get('x0', 0):.1f}, {bbox.get('y0', 0):.1f}, "
                              f"{bbox.get('x1', 0):.1f}, {bbox.get('y1', 0):.1f})")
                st.markdown(f"- **Hash:** `{prov.get('content_hash', 'N/A')[:16]}...`")
            
            if prov.get('text_excerpt'):
                st.markdown(f"**Excerpt:** {prov.get('text_excerpt')}")
            
            if prov.get('confidence'):
                st.markdown(f"**Confidence:** {prov.get('confidence'):.2f}")
            
            st.markdown("---")
