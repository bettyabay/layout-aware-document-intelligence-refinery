"""Triage page - Document classification and profiling."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from streamlit import session_state as ss

from src.agents.triage import TriageAgent
from src.utils.rules import load_rules


def show():
    """Display the Triage page."""
    st.header("Stage 1: Triage Agent")
    st.markdown(
        """
        Document classification and profiling. Analyzes document characteristics
        to determine the optimal extraction strategy.
        """
    )

    # Check if document is selected
    if not ss.current_doc_id or ss.current_doc_id not in ss.documents:
        st.warning("⚠️ Please select a document from the sidebar or upload a new one.")
        return

    doc_id = ss.current_doc_id
    doc_info = ss.documents[doc_id]

    # Load profile if available
    profile_path = Path(f".refinery/profiles/{doc_info['name']}.json")
    if profile_path.exists():
        import json
        with open(profile_path) as f:
            profile_data = json.load(f)
        
        st.success("✅ Document profile found")
        
        # Display profile information
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Origin Type", profile_data.get("origin_type", "N/A"))
        with col2:
            st.metric("Layout Complexity", profile_data.get("layout_complexity", "N/A"))
        with col3:
            st.metric("Domain Hint", profile_data.get("domain_hint", "N/A"))
        with col4:
            st.metric("Confidence", f"{profile_data.get('triage_confidence_score', 0):.2%}")

        # Triage signals
        st.subheader("Triage Signals")
        signals = profile_data.get("triage_signals", {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Char Density", f"{signals.get('avg_char_density', 0):.4f}")
            st.metric("Avg Whitespace Ratio", f"{signals.get('avg_whitespace_ratio', 0):.2%}")
        with col2:
            st.metric("Avg Image Area Ratio", f"{signals.get('avg_image_area_ratio', 0):.2%}")
            st.metric("Table Density", f"{signals.get('table_density', 0):.2f}")
        with col3:
            st.metric("Figure Density", f"{signals.get('figure_density', 0):.2f}")

        # Selected strategy
        st.subheader("Selected Strategy")
        strategy = profile_data.get("selected_strategy", "N/A")
        st.info(f"**Strategy:** {strategy}")
        
        # Strategy decision tree
        st.markdown("### Strategy Decision")
        if strategy == "A":
            st.success("✅ Fast Text Extraction (Strategy A) - Native digital PDF with simple layout")
        elif strategy == "B":
            st.info("ℹ️ Layout-Aware Extraction (Strategy B) - Multi-column or table-heavy document")
        elif strategy == "C":
            st.warning("⚠️ Vision-Augmented Extraction (Strategy C) - Scanned document or complex layout")

        # Full profile JSON
        with st.expander("View Full Profile JSON"):
            st.json(profile_data)

    else:
        st.info("No profile found. Run triage to generate profile.")
        
        if st.button("🔍 Run Triage Agent"):
            with st.spinner("Running Triage Agent..."):
                try:
                    # Load rules
                    rules_path = Path("config/extraction_rules.yaml")
                    if not rules_path.exists():
                        rules_path = Path("rubric/extraction_rules.yaml")
                    rules = load_rules(str(rules_path))
                    
                    # Find document file
                    upload_dir = Path(".refinery/uploads")
                    doc_name = doc_info["name"]
                    pdf_path = upload_dir / doc_name
                    
                    if not pdf_path.exists():
                        st.error(f"Document file not found: {pdf_path}")
                        return
                    
                    # Run triage
                    triage = TriageAgent(rules)
                    profile = triage.profile_document(pdf_path, persist=True)
                    
                    st.success("✅ Triage completed!")
                    st.json(profile.model_dump())
                    
                    # Update session state
                    ss.documents[doc_id]["profile"] = profile.model_dump()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error running triage: {str(e)}")
                    st.exception(e)
