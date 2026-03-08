"""Stage 2: Extraction Test Script"""
from pathlib import Path
from src.agents.extractor import ExtractionRouter
from src.utils.rules import load_rules

rules = load_rules('rubric/extraction_rules.yaml')
router = ExtractionRouter(rules)

pdf_path = Path('data/class_b/2020_Audited_Financial_Statement_Report.pdf')
print(f"Processing: {pdf_path.name}")
print("=" * 60)

extracted, ledger = router.run(pdf_path)

print(f"\n✓ Strategy Sequence: {' -> '.join(s.value for s in ledger.strategy_sequence)}")
print(f"✓ Final Strategy: {ledger.final_strategy.value}")
print(f"✓ Confidence Score: {ledger.confidence_score:.2f}")
print(f"✓ Cost Estimate: ${ledger.cost_estimate_usd:.4f}")
print(f"✓ Pages Extracted: {len(extracted.get('pages', []))}")
print(f"✓ Doc ID: {ledger.doc_id}")
print(f"\n✓ Ledger entry written to: .refinery/extraction_ledger.jsonl")

# Show table count if available
tables = extracted.get('tables', [])
if tables:
    print(f"✓ Tables Found: {len(tables)}")
    if len(tables) > 0:
        print(f"  First table has {len(tables[0].get('rows', []))} rows")
