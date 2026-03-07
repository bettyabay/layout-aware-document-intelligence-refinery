"""Extraction page - Multi-strategy extraction with escalation."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from streamlit import session_state as ss

from src.agents.extractor import ExtractionRouter
from src.agents.triage import TriageAgent
from src.utils.rules import load_rules


def show():
    """Display the Extraction page."""
    st.header("Stage 2: Structure Extraction Layer")
    st.markdown(
        """
        Multi-strategy extraction with confidence-gated escalation:
        - **Strategy A**: Fast Text (pdfplumber)
        - **Strategy B**: Layout-Aware (MinerU/Docling)
        - **Strategy C**: Vision-Augmented (VLM via OpenRouter)
        """
    )

    # Check if document is selected
    if not ss.current_doc_id or ss.current_doc_id not in ss.documents:
        st.warning("⚠️ Please select a document from the sidebar or upload a new one.")
        return

    doc_id = ss.current_doc_id
    doc_info = ss.documents[doc_id]

    # Load extraction ledger
    ledger_path = Path(".refinery/extraction_ledger.jsonl")
    extraction_entries = []
    if ledger_path.exists():
        import json
        with open(ledger_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("doc_id") == doc_id:
                        extraction_entries.append(entry)

    if extraction_entries:
        st.success(f"✅ Found {len(extraction_entries)} extraction entry(ies)")
        
        # Show latest extraction
        latest = extraction_entries[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Strategy", latest.get("final_strategy", "N/A"))
        with col2:
            st.metric("Confidence", f"{latest.get('confidence_score', 0):.2%}")
        with col3:
            st.metric("Cost (USD)", f"${latest.get('cost_estimate_usd', 0):.4f}")
        with col4:
            st.metric("Time (ms)", f"{latest.get('processing_time_ms', 0):.0f}")

        # Strategy sequence
        st.subheader("Strategy Sequence")
        sequence = latest.get("strategy_sequence", [])
        if sequence:
            strategy_labels = {
                "A": "Fast Text",
                "B": "Layout-Aware",
                "C": "Vision-Augmented"
            }
            sequence_str = " → ".join([strategy_labels.get(s, s) for s in sequence])
            st.info(f"**Path:** {sequence_str}")

        # Extraction details
        with st.expander("View Extraction Details"):
            st.json(latest)

        # Show extracted document preview
        st.subheader("Extracted Document Preview")
        
        # Try to load extracted document
        extracted_path = Path(f".refinery/extracted/{doc_id}.json")
        if extracted_path.exists():
            import json
            with open(extracted_path) as f:
                extracted = json.load(f)
            
            st.metric("Pages", len(extracted.get("pages", [])))
            st.metric("LDUs", len(extracted.get("ldus", [])))
            
            # Show first page preview
            if extracted.get("pages"):
                first_page = extracted["pages"][0]
                st.text_area(
                    "First Page Text (preview)",
                    value="\n".join([block.get("text", "") for block in first_page.get("text_blocks", [])[:5]]),
                    height=200
                )
        else:
            st.info("Extracted document file not found. Run extraction to generate it.")

    else:
        st.info("No extraction found. Run extraction to process document.")
        
        if st.button("🚀 Run Extraction"):
            with st.spinner("Running extraction..."):
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
                    
                    # Get or create profile
                    profile_path = Path(f".refinery/profiles/{doc_id}.json")
                    if profile_path.exists():
                        import json
                        with open(profile_path) as f:
                            profile_data = json.load(f)
                        from src.models import DocumentProfile
                        profile = DocumentProfile(**profile_data)
                    else:
                        triage = TriageAgent(rules)
                        profile = triage.profile_document(pdf_path, persist=True)
                    
                    # Run extraction
                    router = ExtractionRouter(rules)
                    extracted_dict, ledger_entry = router.run(pdf_path, profile)
                    
                    st.success("✅ Extraction completed!")
                    
                    # Save extracted document
                    extracted_dir = Path(".refinery/extracted")
                    extracted_dir.mkdir(parents=True, exist_ok=True)
                    import json
                    with open(extracted_dir / f"{doc_id}.json", "w") as f:
                        json.dump(extracted_dict, f, indent=2)
                    
                    st.json(ledger_entry.model_dump())
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error running extraction: {str(e)}")
                    st.exception(e)
