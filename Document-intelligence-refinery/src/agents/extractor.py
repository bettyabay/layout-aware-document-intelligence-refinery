"""Extraction Router - Stage 2: Multi-strategy extraction with escalation."""

import time
from pathlib import Path

from src.agents.chunker import ChunkingEngine
from src.models import DocumentProfile, ExtractionLedgerEntry, StrategyName
from src.strategies import FastTextExtractor, LayoutExtractor, VisionExtractor
from src.utils.ledger import append_jsonl


class ExtractionRouter:
    """Router for multi-strategy extraction with confidence-gated escalation."""

    def __init__(self, rules: dict):
        self.rules = rules
        self.strategies = {
            StrategyName.A: FastTextExtractor(),
            StrategyName.B: LayoutExtractor(),
            StrategyName.C: VisionExtractor(),
        }

    def _next_strategy(self, current: StrategyName) -> StrategyName | None:
        """Get next strategy in escalation sequence."""
        if current == StrategyName.A:
            return StrategyName.B
        if current == StrategyName.B:
            return StrategyName.C
        return None

    def run(
        self, 
        pdf_path: str | Path, 
        profile: DocumentProfile | None = None,
        progress_callback: callable = None
    ) -> tuple[dict, ExtractionLedgerEntry]:
        """Run extraction with escalation.
        
        Args:
            pdf_path: Path to PDF file
            profile: Document profile (optional, will run triage if not provided)
            progress_callback: Optional callback function(status, details) for progress updates
        """
        pdf_path = Path(pdf_path)
        
        # Load profile if not provided
        if profile is None:
            if progress_callback:
                progress_callback("info", "Running triage to create document profile...")
            from src.agents.triage import TriageAgent
            triage = TriageAgent(self.rules)
            profile = triage.profile_document(pdf_path, persist=True)
            if progress_callback:
                progress_callback("success", f"Profile created: {profile.selected_strategy.value}")

        confidence_cfg = self.rules.get("confidence_thresholds", {})
        threshold_ab = float(self.rules.get("escalation", {}).get("a_to_b_threshold", 0.70))
        threshold_bc = float(self.rules.get("escalation", {}).get("b_to_c_threshold", 0.40))
        max_budget = float(self.rules.get("escalation", {}).get("max_vision_cost_per_doc", 0.05))

        # Always start with Strategy A, regardless of triage recommendation
        # This ensures we try all strategies in order: A → B → C
        current = StrategyName.A
        
        if progress_callback:
            progress_callback("info", f"Starting extraction pipeline - always beginning with Strategy A")
            progress_callback("info", f"Triage recommended: {profile.selected_strategy.value}, but we'll try all strategies in order")
            progress_callback("info", f"Escalation thresholds - A→B: {threshold_ab:.2%}, B→C: {threshold_bc:.2%}")

        sequence: list[StrategyName] = []
        total_cost = 0.0
        start = time.perf_counter()
        last_doc = None
        last_conf = 0.0
        notes: str | None = None

        while current is not None:
            strategy_name = current.value
            strategy_label = {
                StrategyName.A: "Fast Text (pdfplumber)",
                StrategyName.B: "Layout-Aware (MinerU/Docling)",
                StrategyName.C: "Vision-Augmented (VLM)"
            }.get(current, strategy_name)
            
            sequence.append(current)
            
            if progress_callback:
                progress_callback("info", f"🔄 Trying Strategy {current.value}: {strategy_label}")
            
            extractor = self.strategies[current]
            
            # Show engine info for layout-aware extraction
            if current == StrategyName.B and hasattr(extractor, 'engine'):
                layout_cfg = self.rules.get("layout_strategy", {})
                engine = layout_cfg.get("engine", "pdfplumber")
                if progress_callback:
                    progress_callback("info", f"   🔧 Using engine: {engine}")
                    # Check if OCR will be used
                    if hasattr(extractor, 'ocr_available') and extractor.ocr_available:
                        if profile.origin_type.value in ["scanned_image", "mixed"]:
                            progress_callback("info", f"   📸 Document is scanned - OCR will be used")
            
            try:
                strategy_start = time.perf_counter()
                if progress_callback:
                    progress_callback("info", f"   ⏳ Extracting with {strategy_label}...")
                
                extracted, confidence, cost = extractor.extract(pdf_path, profile, self.rules)
                
                strategy_time = int((time.perf_counter() - strategy_start) * 1000)
                
                if progress_callback:
                    progress_callback("success", 
                        f"   ✅ Strategy {current.value} completed in {strategy_time}ms\n"
                        f"   📊 Confidence: {confidence:.2%} | Cost: ${cost:.6f} | Pages: {len(extracted.pages)}"
                    )
            except Exception as e:
                if progress_callback:
                    progress_callback("error", f"   ❌ Strategy {current.value} failed: {str(e)}")
                
                if current == StrategyName.C:
                    # Fallback to layout if vision fails
                    if progress_callback:
                        progress_callback("warning", "   ⚠️ Falling back to Strategy B (Layout-Aware)...")
                    sequence.append(StrategyName.B)
                    extractor = self.strategies[StrategyName.B]
                    extracted, confidence, cost = extractor.extract(pdf_path, profile, self.rules)
                    last_doc = extracted
                    last_conf = confidence
                    total_cost += cost
                    notes = f"vision_fallback: {str(e)}"
                    if progress_callback:
                        progress_callback("success", f"   ✅ Fallback to Strategy B successful")
                    break
                raise

            total_cost += cost
            last_doc = extracted
            last_conf = confidence

            # Check if we should escalate
            threshold = threshold_ab if current == StrategyName.A else threshold_bc
            if progress_callback:
                progress_callback("info", 
                    f"   🔍 Evaluating escalation: Confidence {confidence:.2%} vs Threshold {threshold:.2%}"
                )
            
            if confidence >= threshold or current == StrategyName.C:
                if progress_callback:
                    if confidence >= threshold:
                        progress_callback("success", 
                            f"   ✅ Confidence {confidence:.2%} meets threshold {threshold:.2%} - stopping escalation"
                        )
                    else:
                        progress_callback("info", "   ℹ️ Already at highest strategy (C) - stopping")
                break  # Good enough or already at highest strategy
            else:
                next_strategy = self._next_strategy(current)
                if next_strategy:
                    if progress_callback:
                        progress_callback("warning", 
                            f"   ⚠️ Confidence {confidence:.2%} below threshold {threshold:.2%} - escalating to Strategy {next_strategy.value}"
                        )
                current = next_strategy

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        final_strategy = sequence[-1]
        budget_status = "cap_reached" if total_cost >= max_budget else "under_cap"

        if progress_callback:
            progress_callback("info", f"📝 Generating Logical Document Units (LDUs)...")
        
        # Generate LDUs
        last_doc.ldus = ChunkingEngine(self.rules).build(last_doc)
        last_doc.metadata.strategy_sequence = sequence
        
        if progress_callback:
            progress_callback("success", f"   ✅ Generated {len(last_doc.ldus)} LDUs")

        if progress_callback:
            progress_callback("info", "🌳 Building PageIndex hierarchy...")
        
        # Build PageIndex
        from src.agents.indexer import PageIndexBuilder
        indexer = PageIndexBuilder(self.rules)
        last_doc.page_index = indexer.build(last_doc)
        
        if progress_callback:
            progress_callback("success", "   ✅ PageIndex built")

        # Create ledger entry
        entry = ExtractionLedgerEntry(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            doc_id=profile.doc_id,
            document_name=profile.document_name,
            strategy_sequence=sequence,
            final_strategy=final_strategy,
            confidence_score=last_conf,
            cost_estimate_usd=round(total_cost, 6),
            processing_time_ms=elapsed_ms,
            budget_cap_usd=max_budget,
            budget_status=budget_status,
            notes=notes,
        )

        # Append to ledger
        ledger_path = Path(".refinery/extraction_ledger.jsonl")
        append_jsonl(entry.model_dump(), ledger_path)

        # Save extracted document
        extracted_dir = Path(".refinery/extracted")
        extracted_dir.mkdir(parents=True, exist_ok=True)
        import json
        from src.utils.ledger import write_json
        write_json(last_doc.model_dump(), extracted_dir / f"{profile.doc_id}.json")

        # Save PageIndex
        pageindex_dir = Path(".refinery/pageindex")
        pageindex_dir.mkdir(parents=True, exist_ok=True)
        if last_doc.page_index:
            write_json(last_doc.page_index.model_dump(), pageindex_dir / f"{profile.doc_id}_pageindex.json")

        # Ingest into vector store
        if progress_callback:
            progress_callback("info", "🔍 Ingesting into vector store for semantic search...")
        
        try:
            from src.db.vector_store import VectorStore
            vector_store = VectorStore()
            vector_store.ingest(profile.doc_id, [ldu.model_dump() for ldu in last_doc.ldus])
            if progress_callback:
                progress_callback("success", f"   ✅ Ingested {len(last_doc.ldus)} chunks into vector store")
        except Exception as e:
            # Log but don't fail if vector store fails
            if progress_callback:
                progress_callback("warning", f"   ⚠️ Vector store ingestion failed: {str(e)}")
            import logging
            logging.warning(f"Failed to ingest into vector store: {e}")
        
        if progress_callback:
            progress_callback("success", 
                f"🎉 Extraction complete!\n"
                f"   Final Strategy: {final_strategy.value}\n"
                f"   Total Confidence: {last_conf:.2%}\n"
                f"   Total Cost: ${total_cost:.6f}\n"
                f"   Total Time: {elapsed_ms}ms"
            )

        return last_doc.model_dump(), entry
