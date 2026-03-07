"""
Document Intelligence Refinery - Streamlit Web UI

Main entry point for the Streamlit application.
Provides document upload, processing pipeline, and query interface.
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from streamlit import session_state as ss

# Configure page
st.set_page_config(
    page_title="Document Intelligence Refinery",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "documents" not in ss:
    ss.documents = {}  # doc_id -> document info
if "current_doc_id" not in ss:
    ss.current_doc_id = None
if "processing_status" not in ss:
    ss.processing_status = {}  # doc_id -> status dict


def main():
    """Main Streamlit application."""
    st.title("📄 Document Intelligence Refinery")
    st.markdown(
        """
        A production-grade, multi-stage agentic pipeline for document extraction 
        with provenance tracking. Transform unstructured PDFs into structured, 
        queryable, spatially-indexed knowledge.
        """
    )

    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        st.markdown("---")
        
        # Document selection
        if ss.documents:
            st.subheader("Processed Documents")
            doc_names = list(ss.documents.keys())
            selected_doc = st.selectbox(
                "Select Document",
                options=[None] + doc_names,
                format_func=lambda x: "Choose a document..." if x is None else ss.documents[x].get("name", x),
            )
            if selected_doc:
                ss.current_doc_id = selected_doc
        else:
            st.info("No documents processed yet. Upload a document to get started.")
            ss.current_doc_id = None

        st.markdown("---")
        st.markdown("### Pipeline Stages")
        st.markdown("""
        1. **Triage** - Document classification
        2. **Extraction** - Multi-strategy extraction
        3. **PageIndex** - Hierarchical navigation
        4. **Query** - Natural language queries
        """)

        st.markdown("---")
        st.markdown("### Configuration")
        st.markdown(f"**Refinery Dir:** `.refinery/`")
        st.markdown(f"**Rules:** `config/extraction_rules.yaml`")

    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Home",
        "1️⃣ Triage",
        "2️⃣ Extraction",
        "3️⃣ PageIndex",
        "4️⃣ Query"
    ])

    with tab1:
        show_home_tab()

    with tab2:
        from streamlit_app.pages import triage_page
        triage_page.show()

    with tab3:
        from streamlit_app.pages import extraction_page
        extraction_page.show()

    with tab4:
        # Import PageIndex page
        import importlib.util
        pageindex_path = Path(__file__).parent / "pages" / "3_PageIndex.py"
        if pageindex_path.exists():
            spec = importlib.util.spec_from_file_location("pageindex_page", pageindex_path)
            pageindex_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pageindex_module)
            pageindex_module.show()
        else:
            st.info("PageIndex page not available. Please ensure 3_PageIndex.py exists.")

    with tab5:
        # Import Query page
        query_path = Path(__file__).parent / "pages" / "4_Query.py"
        if query_path.exists():
            spec = importlib.util.spec_from_file_location("query_page", query_path)
            query_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(query_module)
            query_module.show()
        else:
            st.info("Query page not available. Please ensure 4_Query.py exists.")


def show_home_tab():
    """Display the home tab with document upload and overview."""
    st.header("Document Upload & Processing")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload a PDF file to process through the pipeline"
    )

    if uploaded_file is not None:
        # Save uploaded file
        upload_dir = Path(".refinery/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / uploaded_file.name
        
        if not file_path.exists():
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ File uploaded: {uploaded_file.name}")

        # Process button
        if st.button("🚀 Process Document", type="primary"):
            process_document(file_path)

    # Show processed documents
    if ss.documents:
        st.markdown("---")
        st.subheader("Processed Documents")
        
        for doc_id, doc_info in ss.documents.items():
            with st.expander(f"📄 {doc_info.get('name', doc_id)}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Origin Type", doc_info.get("origin_type", "N/A"))
                with col2:
                    st.metric("Layout Complexity", doc_info.get("layout_complexity", "N/A"))
                with col3:
                    st.metric("Strategy", doc_info.get("strategy", "N/A"))
                
                if st.button(f"Select {doc_info.get('name', doc_id)}", key=f"select_{doc_id}"):
                    ss.current_doc_id = doc_id
                    st.rerun()

    # Pipeline overview
    st.markdown("---")
    st.subheader("Pipeline Overview")
    
    st.markdown("""
    The Document Intelligence Refinery processes documents through 5 stages:
    
    1. **Triage Agent** - Classifies documents and determines extraction strategy
    2. **Structure Extraction** - Multi-strategy extraction with confidence-gated escalation
    3. **Semantic Chunking** - Generates Logical Document Units (LDUs) with rules
    4. **PageIndex Builder** - Creates hierarchical navigation structure
    5. **Query Interface** - LangGraph agent with full provenance tracking
    """)

    # Architecture diagram
    st.markdown("### Architecture")
    st.code("""
    INPUT (PDFs, Docs, Images)
        ↓
    ┌─────────────────────────────────────┐
    │ Stage 1: Triage Agent               │
    │ - Document classification            │
    │ - Origin type detection             │
    │ - Layout complexity analysis         │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │ Stage 2: Structure Extraction      │
    │ - Strategy A: Fast Text             │
    │ - Strategy B: Layout-Aware          │
    │ - Strategy C: Vision-Augmented      │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │ Stage 3: Semantic Chunking          │
    │ - LDU generation                    │
    │ - Table integrity                   │
    │ - Section hierarchy                 │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │ Stage 4: PageIndex Builder          │
    │ - Hierarchical section tree         │
    │ - LLM summaries                    │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │ Stage 5: Query Interface            │
    │ - PageIndex navigation              │
    │ - Semantic search                  │
    │ - Provenance tracking               │
    └─────────────────────────────────────┘
        ↓
    OUTPUT (Structured JSON, Vectors, Provenance)
    """)


def process_document(file_path: Path):
    """Process a document through the pipeline."""
    with st.spinner("Processing document..."):
        try:
            # Import here to avoid circular imports
            from src.agents.triage import TriageAgent
            from src.utils.rules import load_rules
            
            # Load rules
            rules_path = Path("config/extraction_rules.yaml")
            if not rules_path.exists():
                rules_path = Path("rubric/extraction_rules.yaml")
            rules = load_rules(str(rules_path))
            
            # Stage 1: Triage
            st.info("Stage 1: Running Triage Agent...")
            triage = TriageAgent(rules)
            profile = triage.profile_document(file_path, persist=True)
            
            # Store document info
            doc_id = profile.doc_id
            ss.documents[doc_id] = {
                "name": profile.document_name,
                "origin_type": profile.origin_type.value,
                "layout_complexity": profile.layout_complexity.value,
                "strategy": profile.selected_strategy.value,
                "profile": profile.model_dump(),
            }
            ss.current_doc_id = doc_id
            
            st.success(f"✅ Document processed: {profile.document_name}")
            st.json(profile.model_dump())
            
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
