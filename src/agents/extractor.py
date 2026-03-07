from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.agents.triage import TriageAgent
from src.agents.chunker import ChunkingEngine
from src.models import DocumentProfile, ExtractionLedgerEntry, StrategyName
from src.strategies import FastTextExtractor, LayoutExtractor, VisionExtractor
from src.utils.ledger import append_jsonl
from src.utils.rules import load_rules


class ExtractionRouter:
    def __init__(self, rules: dict):
        self.rules = rules
        self.strategies = {
            StrategyName.A: FastTextExtractor(),
            StrategyName.B: LayoutExtractor(),
            StrategyName.C: VisionExtractor(),
        }

    def _next_strategy(self, current: StrategyName) -> StrategyName | None:
        if current == StrategyName.A:
            return StrategyName.B
        if current == StrategyName.B:
            return StrategyName.C
        return None

    def run(self, pdf_path: str | Path, profile: DocumentProfile | None = None) -> tuple[dict, ExtractionLedgerEntry]:
        pdf_path = Path(pdf_path)
        triage = TriageAgent(self.rules)
        profile = profile or triage.profile_document(pdf_path, persist=True)

        confidence_cfg = self.rules.get("confidence", {})
        threshold_ab = float(confidence_cfg.get("escalate_threshold_ab", 0.45))
        threshold_bc = float(confidence_cfg.get("escalate_threshold_bc", 0.40))
        max_budget = float(self.rules.get(
            "vision", {}).get("max_cost_per_doc_usd", 2.0))

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
                extracted, confidence, cost = extractor.extract(
                    pdf_path, profile, self.rules)
            except Exception:
                if current == StrategyName.C:
                    sequence.append(StrategyName.B)
                    extractor = self.strategies[StrategyName.B]
                    extracted, confidence, cost = extractor.extract(
                        pdf_path, profile, self.rules)
                    last_doc = extracted
                    last_conf = confidence
                    total_cost += cost
                    notes = "layout_vision_fallback"
                    break
                raise
            total_cost += cost
            last_doc = extracted
            last_conf = confidence

            threshold = threshold_ab if current == StrategyName.A else threshold_bc
            if confidence >= threshold or current == StrategyName.C:
                break
            current = self._next_strategy(current)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        final_strategy = sequence[-1]
        budget_status = "cap_reached" if total_cost >= max_budget else "under_cap"

        last_doc.ldus = ChunkingEngine().build(last_doc)
        last_doc.metadata.strategy_sequence = sequence

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

        append_jsonl(Path(".refinery/extraction_ledger.jsonl"),
                     entry.model_dump())
        return last_doc.model_dump(), entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run extraction router with escalation guard")
    parser.add_argument("--input", required=True, help="Path to input PDF")
    parser.add_argument(
        "--rules", default="rubric/extraction_rules.yaml", help="Rules file path")
    args = parser.parse_args()

    rules = load_rules(args.rules)
    router = ExtractionRouter(rules)
    extracted, entry = router.run(args.input)
    print("Extraction completed")
    print(entry.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
