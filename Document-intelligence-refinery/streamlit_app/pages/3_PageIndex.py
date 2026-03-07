"""PageIndex page - Hierarchical navigation structure."""
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from streamlit import session_state as ss

from src.agents.indexer import PageIndexBuilder
from src.models import ExtractedDocument


def show():
    """Display the PageIndex page."""
    st.header("Stage 3: PageIndex Builder")
    st.markdown(
        """
        Hierarchical navigation structure with section summaries.
        Enables efficient document navigation without reading entire documents.
        """
    )

    # Check if document is selected
    if not ss.current_doc_id or ss.current_doc_id not in ss.documents:
        st.warning("⚠️ Please select a document from the sidebar or upload a new one.")
        return

    doc_id = ss.current_doc_id
    doc_info = ss.documents[doc_id]

    # Load PageIndex if available
    pageindex_path = Path(f".refinery/pageindex/{doc_info['name']}_pageindex.json")
    
    if pageindex_path.exists():
        with open(pageindex_path) as f:
            pageindex_data = json.load(f)
        
        st.success("✅ PageIndex found")
        
        # Display PageIndex tree
        st.subheader("PageIndex Tree")
        display_pageindex_tree(pageindex_data)
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sections", count_sections(pageindex_data))
        with col2:
            st.metric("Total Pages", pageindex_data.get("total_pages", 0))
        with col3:
            st.metric("Max Depth", get_max_depth(pageindex_data))
        
        # Download button
        st.download_button(
            label="📥 Download PageIndex JSON",
            data=json.dumps(pageindex_data, indent=2),
            file_name=f"{doc_info['name']}_pageindex.json",
            mime="application/json"
        )
    else:
        st.info("ℹ️ PageIndex not found. Run extraction first to generate PageIndex.")
        
        if st.button("🔨 Build PageIndex", type="primary"):
            build_pageindex(doc_id, doc_info)


def display_pageindex_tree(node: dict, level: int = 0):
    """Recursively display PageIndex tree."""
    indent = "  " * level
    node_type = node.get("node_type", "unknown")
    label = node.get("label", "Untitled")
    page_num = node.get("page_number")
    
    # Display node
    if page_num:
        st.markdown(f"{indent}**{label}** (Page {page_num})")
    else:
        st.markdown(f"{indent}**{label}**")
    
    # Display summary if available
    summary = node.get("summary")
    if summary:
        st.markdown(f"{indent}  *{summary}*")
    
    # Display children
    children = node.get("children", [])
    if children:
        with st.expander(f"{indent}  {len(children)} child(ren)", expanded=level < 2):
            for child in children:
                display_pageindex_tree(child, level + 1)


def count_sections(node: dict) -> int:
    """Count total sections in PageIndex tree."""
    count = 1  # Count current node
    for child in node.get("children", []):
        count += count_sections(child)
    return count


def get_max_depth(node: dict, current_depth: int = 0) -> int:
    """Get maximum depth of PageIndex tree."""
    max_depth = current_depth
    for child in node.get("children", []):
        child_depth = get_max_depth(child, current_depth + 1)
        max_depth = max(max_depth, child_depth)
    return max_depth


def build_pageindex(doc_id: str, doc_info: dict):
    """Build PageIndex for a document."""
    with st.spinner("Building PageIndex..."):
        try:
            from src.agents.extractor import ExtractionRouter
            from src.utils.rules import load_rules
            
            # Load rules
            rules_path = Path("config/extraction_rules.yaml")
            if not rules_path.exists():
                rules_path = Path("rubric/extraction_rules.yaml")
            rules = load_rules(str(rules_path))
            
            # Load extracted document
            extracted_path = Path(f".refinery/extractions/{doc_info['name']}_extracted.json")
            if not extracted_path.exists():
                st.error("❌ Extracted document not found. Please run extraction first.")
                return
            
            with open(extracted_path) as f:
                extracted_data = json.load(f)
            
            extracted_doc = ExtractedDocument(**extracted_data)
            
            # Build PageIndex
            indexer = PageIndexBuilder()
            pageindex = indexer.build(extracted_doc)
            
            # Save PageIndex
            pageindex_dir = Path(".refinery/pageindex")
            pageindex_dir.mkdir(parents=True, exist_ok=True)
            
            pageindex_path = pageindex_dir / f"{doc_info['name']}_pageindex.json"
            with open(pageindex_path, "w") as f:
                json.dump(pageindex.model_dump(), f, indent=2)
            
            st.success("✅ PageIndex built successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error building PageIndex: {str(e)}")
            st.exception(e)
