"""Extraction page - Multi-strategy extraction with escalation."""
import sys
import time
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
        
        # Main metrics with better formatting
        st.markdown("### 📊 Extraction Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        final_strategy = latest.get("final_strategy", "N/A")
        strategy_emoji = {"A": "⚡", "B": "📐", "C": "👁️"}.get(final_strategy, "❓")
        
        with col1:
            st.metric(
                "Final Strategy", 
                f"{strategy_emoji} {final_strategy}",
                help="The strategy that successfully extracted the document"
            )
        with col2:
            confidence = latest.get('confidence_score', 0)
            confidence_color = "normal" if confidence >= 0.7 else "inverse" if confidence >= 0.4 else "off"
            st.metric(
                "Confidence", 
                f"{confidence:.1%}",
                delta=f"{'High' if confidence >= 0.7 else 'Medium' if confidence >= 0.4 else 'Low'}",
                delta_color=confidence_color,
                help="Extraction confidence score"
            )
        with col3:
            cost = latest.get('cost_estimate_usd', 0)
            st.metric(
                "Cost (USD)", 
                f"${cost:.6f}",
                help="Total extraction cost"
            )
        with col4:
            time_ms = latest.get('processing_time_ms', 0)
            st.metric(
                "Processing Time", 
                f"{time_ms}ms",
                help="Total extraction time in milliseconds"
            )

        # Strategy sequence with visual flow
        st.markdown("### 🔄 Strategy Execution Path")
        sequence = latest.get("strategy_sequence", [])
        if sequence:
            strategy_info = {
                "A": {"name": "Fast Text", "emoji": "⚡", "desc": "pdfplumber"},
                "B": {"name": "Layout-Aware", "emoji": "📐", "desc": "MinerU/Docling"},
                "C": {"name": "Vision-Augmented", "emoji": "👁️", "desc": "VLM via OpenRouter"}
            }
            
            # Create visual flow
            cols = st.columns(len(sequence))
            for i, strategy in enumerate(sequence):
                strategy_key = strategy if isinstance(strategy, str) else strategy.value if hasattr(strategy, 'value') else str(strategy)
                info = strategy_info.get(strategy_key, {"name": strategy_key, "emoji": "❓", "desc": ""})
                is_final = (i == len(sequence) - 1)
                
                with cols[i]:
                    if is_final:
                        st.markdown(f"""
                        <div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 10px; color: white;'>
                        <h3>{info['emoji']} Strategy {strategy_key}</h3>
                        <p style='margin: 5px 0;'><strong>{info['name']}</strong></p>
                        <p style='font-size: 0.9em; opacity: 0.9;'>{info['desc']}</p>
                        <p style='margin-top: 10px; font-size: 0.8em;'>✓ Final</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='text-align: center; padding: 15px; background: #f0f2f6; 
                        border-radius: 10px; border: 2px solid #d1d5db;'>
                        <h3>{info['emoji']} Strategy {strategy_key}</h3>
                        <p style='margin: 5px 0;'><strong>{info['name']}</strong></p>
                        <p style='font-size: 0.9em; color: #6b7280;'>{info['desc']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Add arrows between strategies
            if len(sequence) > 1:
                arrow_cols = st.columns(len(sequence) - 1)
                for i in range(len(sequence) - 1):
                    with arrow_cols[i]:
                        st.markdown("<div style='text-align: center; font-size: 24px; color: #667eea;'>→</div>", unsafe_allow_html=True)

        # Budget and status info
        budget_status = latest.get("budget_status", "unknown")
        budget_cap = latest.get("budget_cap_usd", 0)
        if budget_status == "cap_reached":
            st.warning(f"⚠️ Budget cap reached (${budget_cap:.4f})")
        elif budget_cap > 0:
            st.info(f"💰 Budget status: {budget_status.replace('_', ' ').title()} (Cap: ${budget_cap:.4f})")

        # Extraction details
        with st.expander("📋 View Full Extraction Ledger Entry"):
            st.json(latest)

        # Show extracted document preview
        st.markdown("### 📄 Extracted Document Content")
        
        # Try to load extracted document
        extracted_path = Path(f".refinery/extracted/{doc_id}.json")
        if extracted_path.exists():
            import json
            with open(extracted_path) as f:
                extracted = json.load(f)
            
            # Document statistics
            pages = extracted.get("pages", [])
            ldus = extracted.get("ldus", [])
            total_text_blocks = sum(len(page.get("text_blocks", [])) for page in pages)
            total_tables = sum(len(page.get("tables", [])) for page in pages)
            total_figures = sum(len(page.get("figures", [])) for page in pages)
            
            stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
            with stat_col1:
                st.metric("📄 Pages", len(pages))
            with stat_col2:
                st.metric("📝 LDUs", len(ldus))
            with stat_col3:
                st.metric("📋 Text Blocks", total_text_blocks)
            with stat_col4:
                st.metric("📊 Tables", total_tables)
            with stat_col5:
                st.metric("🖼️ Figures", total_figures)
            
            # Content preview
            if pages:
                tab1, tab2, tab3 = st.tabs(["📝 Text Content", "📊 Tables", "🖼️ Figures"])
                
                with tab1:
                    if total_text_blocks > 0:
                        # Show text from all pages
                        all_text = []
                        for page in pages:
                            page_num = page.get("page_number", 0)
                            text_blocks = page.get("text_blocks", [])
                            if text_blocks:
                                page_text = "\n".join([block.get("text", "") for block in text_blocks])
                                all_text.append(f"--- Page {page_num} ---\n{page_text}\n")
                        
                        if all_text:
                            st.text_area(
                                "Extracted Text Content",
                                value="\n".join(all_text),
                                height=400,
                                help="Text extracted from all pages"
                            )
                        else:
                            st.info("No text blocks found in extracted content.")
                    else:
                        st.warning("⚠️ No text content was extracted. This may indicate a scanned document that requires OCR.")
                        st.info("💡 Try running extraction with Strategy C (Vision-Augmented) if you have OPENROUTER_API_KEY configured.")
                
                with tab2:
                    if total_tables > 0:
                        for page in pages:
                            tables = page.get("tables", [])
                            if tables:
                                st.markdown(f"**Page {page.get('page_number', 0)}**")
                                for i, table in enumerate(tables):
                                    st.markdown(f"**Table {i+1}**")
                                    # Display table if it has data
                                    if table.get("data"):
                                        st.dataframe(table.get("data"))
                                    else:
                                        st.json(table)
                    else:
                        st.info("No tables found in the document.")
                
                with tab3:
                    if total_figures > 0:
                        for page in pages:
                            figures = page.get("figures", [])
                            if figures:
                                st.markdown(f"**Page {page.get('page_number', 0)}**")
                                for i, figure in enumerate(figures):
                                    st.markdown(f"**Figure {i+1}**")
                                    st.json(figure)
                    else:
                        st.info("No figures found in the document.")
            else:
                st.warning("No pages found in extracted document.")
        else:
            st.info("Extracted document file not found. Run extraction to generate it.")

    else:
        st.info("ℹ️ No extraction found. Run extraction to process document.")
        
        # Show expected strategy based on profile
        profile_path = Path(f".refinery/profiles/{doc_id}.json")
        if profile_path.exists():
            import json
            with open(profile_path) as f:
                profile_data = json.load(f)
            strategy = profile_data.get("selected_strategy", "unknown")
            origin_type = profile_data.get("origin_type", "unknown")
            
            if strategy == "vision_augmented" or origin_type == "scanned_image":
                st.info(
                    "ℹ️ **Note:** This document is classified as scanned. "
                    "The pipeline will always try all strategies in order: "
                    "**Strategy A** (Fast Text) → **Strategy B** (Layout-Aware) → **Strategy C** (Vision-Augmented). "
                    "Strategy B will be attempted even for scanned documents. "
                    "Make sure `OPENROUTER_API_KEY` is set in your `.env` file if you want to use vision models (Strategy C)."
                )
        
        st.markdown("---")
        
        if st.button("🚀 Run Extraction", type="primary", use_container_width=True):
            # Create progress container
            progress_container = st.container()
            status_container = progress_container.empty()
            log_expander = progress_container.expander("📋 Detailed Progress Log", expanded=True)
            log_display = log_expander.empty()
            
            # Initialize log
            progress_log = []
            
            def progress_callback(status_type: str, message: str):
                """Callback function to update progress in real-time."""
                timestamp = time.strftime("%H:%M:%S")
                icon_map = {
                    "info": "ℹ️",
                    "success": "✅",
                    "warning": "⚠️",
                    "error": "❌"
                }
                icon = icon_map.get(status_type, "•")
                log_entry = f"[{timestamp}] {icon} {message}"
                progress_log.append(log_entry)
                
                # Update log display using markdown in empty container (no key needed)
                log_display.markdown(
                    f"```\n" + "\n".join(progress_log) + "\n```"
                )
                
                # Update status
                if status_type == "success" and "complete" in message.lower():
                    status_container.success(message)
                elif status_type == "error":
                    status_container.error(message)
                elif status_type == "warning":
                    status_container.warning(message)
                else:
                    status_container.info(message)
            
            try:
                import time
                
                # Load rules
                progress_callback("info", "📋 Loading extraction rules...")
                rules_path = Path("config/extraction_rules.yaml")
                if not rules_path.exists():
                    rules_path = Path("rubric/extraction_rules.yaml")
                rules = load_rules(str(rules_path))
                progress_callback("success", "   ✅ Rules loaded")
                
                # Find document file
                progress_callback("info", f"📄 Locating document: {doc_info['name']}")
                upload_dir = Path(".refinery/uploads")
                doc_name = doc_info["name"]
                pdf_path = upload_dir / doc_name
                
                if not pdf_path.exists():
                    progress_callback("error", f"   ❌ Document file not found: {pdf_path}")
                    st.error(f"Document file not found: {pdf_path}")
                    return
                
                progress_callback("success", f"   ✅ Document found: {pdf_path}")
                
                # Get or create profile
                profile_path = Path(f".refinery/profiles/{doc_id}.json")
                if profile_path.exists():
                    progress_callback("info", "📊 Loading existing document profile...")
                    import json
                    with open(profile_path) as f:
                        profile_data = json.load(f)
                    from src.models import DocumentProfile
                    profile = DocumentProfile(**profile_data)
                    progress_callback("success", f"   ✅ Profile loaded: {profile.selected_strategy.value}")
                else:
                    progress_callback("info", "📊 Creating new document profile...")
                    triage = TriageAgent(rules)
                    profile = triage.profile_document(pdf_path, persist=True)
                    progress_callback("success", f"   ✅ Profile created: {profile.selected_strategy.value}")
                
                # Run extraction with progress callback
                progress_callback("info", "🚀 Starting multi-strategy extraction pipeline...")
                router = ExtractionRouter(rules)
                extracted_dict, ledger_entry = router.run(pdf_path, profile, progress_callback=progress_callback)
                
                # Save extracted document
                progress_callback("info", "💾 Saving extracted document...")
                extracted_dir = Path(".refinery/extracted")
                extracted_dir.mkdir(parents=True, exist_ok=True)
                import json
                with open(extracted_dir / f"{doc_id}.json", "w") as f:
                    json.dump(extracted_dict, f, indent=2)
                progress_callback("success", "   ✅ Extracted document saved")
                
                # Show final summary
                st.markdown("---")
                st.subheader("📊 Extraction Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Final Strategy", ledger_entry.final_strategy.value)
                with col2:
                    st.metric("Confidence", f"{ledger_entry.confidence_score:.2%}")
                with col3:
                    st.metric("Cost (USD)", f"${ledger_entry.cost_estimate_usd:.6f}")
                with col4:
                    st.metric("Time", f"{ledger_entry.processing_time_ms}ms")
                
                # Strategy sequence visualization
                st.subheader("🔄 Strategy Sequence")
                sequence = ledger_entry.strategy_sequence
                if sequence:
                    strategy_labels = {
                        "A": "Fast Text",
                        "B": "Layout-Aware",
                        "C": "Vision-Augmented"
                    }
                    sequence_display = []
                    for i, s in enumerate(sequence):
                        strategy_name = s.value if hasattr(s, 'value') else s
                        label = strategy_labels.get(strategy_name, strategy_name)
                        if i < len(sequence) - 1:
                            sequence_display.append(f"**{label}** →")
                        else:
                            sequence_display.append(f"**{label}** ✓")
                    st.markdown(" ".join(sequence_display))
                
                # Show ledger entry details
                with st.expander("📋 Full Extraction Ledger Entry"):
                    st.json(ledger_entry.model_dump())
                
                st.rerun()
                
            except Exception as e:
                progress_callback("error", f"❌ Extraction failed: {str(e)}")
                st.error(f"❌ Error running extraction: {str(e)}")
                st.exception(e)
