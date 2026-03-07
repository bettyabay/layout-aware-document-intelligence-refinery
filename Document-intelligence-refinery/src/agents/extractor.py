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
        self, pdf_path: str | Path, profile: DocumentProfile | None = None
    ) -> tuple[dict, ExtractionLedgerEntry]:
        """Run extraction with escalation."""
        pdf_path = Path(pdf_path)
        
        # Load profile if not provided
        if profile is None:
            from src.agents.triage import TriageAgent
            triage = TriageAgent(self.rules)
            profile = triage.profile_document(pdf_path, persist=True)

        confidence_cfg = self.rules.get("confidence_thresholds", {})
        threshold_ab = float(self.rules.get("escalation", {}).get("a_to_b_threshold", 0.70))
        threshold_bc = float(self.rules.get("escalation", {}).get("b_to_c_threshold", 0.40))
        max_budget = float(self.rules.get("escalation", {}).get("max_vision_cost_per_doc", 0.05))

        sequence: list[StrategyName] = []
        total_cost = 0.0
        start = time.perf_counter()
        current = profile.selected_strategy
        last_doc = None
        last_conf = 0.0
        notes: str | None = None

        while current is not None:
            sequence.append(current)
            extractor = self.strategies[current]
            try:
                extracted, confidence, cost = extractor.extract(pdf_path, profile, self.rules)
            except Exception as e:
                if current == StrategyName.C:
                    # Fallback to layout if vision fails
                    sequence.append(StrategyName.B)
                    extractor = self.strategies[StrategyName.B]
                    extracted, confidence, cost = extractor.extract(pdf_path, profile, self.rules)
                    last_doc = extracted
                    last_conf = confidence
                    total_cost += cost
                    notes = f"vision_fallback: {str(e)}"
                    break
                raise

            total_cost += cost
            last_doc = extracted
            last_conf = confidence

            # Check if we should escalate
            threshold = threshold_ab if current == StrategyName.A else threshold_bc
            if confidence >= threshold or current == StrategyName.C:
                break  # Good enough or already at highest strategy
            current = self._next_strategy(current)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        final_strategy = sequence[-1]
        budget_status = "cap_reached" if total_cost >= max_budget else "under_cap"

        # Generate LDUs
        last_doc.ldus = ChunkingEngine(self.rules).build(last_doc)
        last_doc.metadata.strategy_sequence = sequence

        # Build PageIndex
        from src.agents.indexer import PageIndexBuilder
        indexer = PageIndexBuilder(self.rules)
        last_doc.page_index = indexer.build(last_doc)

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
        append_jsonl(ledger_path, entry.model_dump())

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
        try:
            from src.db.vector_store import VectorStore
            vector_store = VectorStore()
            vector_store.ingest(profile.doc_id, [ldu.model_dump() for ldu in last_doc.ldus])
        except Exception as e:
            # Log but don't fail if vector store fails
            import logging
            logging.warning(f"Failed to ingest into vector store: {e}")

        return last_doc.model_dump(), entry
