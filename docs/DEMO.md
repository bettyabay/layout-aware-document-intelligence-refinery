# Interim Submission Demo Guide

This guide explains how to run the interim demo and what to expect.

## Prerequisites

1. **Python Environment**: Python 3.11+ with all dependencies installed
   ```bash
   pip install -e ".[dev]"
   ```

2. **Document Files**: Ensure test documents are in the `data/` directory:
   - `data/class_a/CBE_ANNUAL_REPORT_2023-24.pdf`
   - `data/class_b/Annual_Report_JUNE-2023.pdf`
   - `data/class_c/fta_performance_survey_final_report_2022.pdf`
   - `data/class_d/tax_expenditure_ethiopia_2021_22.pdf`

3. **System Requirements**:
   - **Minimum**: 8 GB RAM (for fast_text extraction only)
   - **Recommended**: 16+ GB RAM (for layout-aware extraction with Docling)
   - **Virtual Memory**: Ensure sufficient paging file size on Windows

4. **Optional ML Dependencies** (for layout-aware extraction):
   ```bash
   pip install docling  # For Strategy B (Docling)
   # OR
   pip install mineru  # For Strategy B (MinerU)
   ```
   **Note**: If you encounter memory issues, the demo will automatically fall back to fast_text extraction.

5. **Optional API Key** (for vision extraction):
   ```bash
   # Windows PowerShell
   $env:OPENROUTER_API_KEY="your_key_here"
   
   # Linux/Mac
   export OPENROUTER_API_KEY=your_key_here
   ```
   **Note**: Vision extraction is only needed for scanned documents. The demo will work without it.

## Running the Demo

### Step 0: Ensure Minimum Documents (Optional)

To meet the requirement of 12 processed documents (3 per class):

```bash
python demo/ensure_profiles.py
```

**Note**: This script processes the available documents. To truly have 12 different documents, you would need 3 different PDF files per class in the `data/` directory. The current setup has 1 document per class, so this script will process those 4 documents.

### Step 1: Run the Demo Script

```bash
python demo/interim_demo.py
```

This will:
- Process one document from each class (A, B, C, D)
- Generate DocumentProfile JSON files in `.refinery/profiles/`
- Create extraction_ledger.jsonl entries
- Print cost analysis for each strategy
- Display extraction summaries

### Step 2: Verify Outputs

After running the demo, check:

1. **Profiles Directory**:
   ```bash
   ls .refinery/profiles/
   ```
   Should contain at least 4 JSON files (one per document class).

2. **Extraction Ledger**:
   ```bash
   cat .refinery/extraction_ledger.jsonl | head -20
   ```
   Should show JSONL entries with extraction details.

### Step 3: Start Web UI

```bash
uvicorn src.web.app:app --reload
```

Then open `http://localhost:8000` in your browser.

## Expected Outputs

### Console Output

The demo script will print:

1. **Document Profile Summary** for each document:
   - Document ID
   - Origin type (native_digital, scanned_image, mixed)
   - Layout complexity (single_column, multi_column, table_heavy, etc.)
   - Domain hint (financial, legal, technical, etc.)
   - Estimated cost tier
   - Language and confidence
   - Page count and file size

2. **Cost Analysis Table**:
   ```
   Strategy            Cost/Page        Total Cost       For N pages
   -----------------------------------------------------------------
   fast_text          $0.000000        $0.000000        $0.000000
   layout_aware       $0.000500        $0.012500        $0.012500
   vision_augmented   $0.003000        $0.075000        $0.075000
   ```

3. **Extraction Summary**:
   - Number of text blocks extracted
   - Number of tables extracted
   - Number of figures extracted

4. **Ledger Summary**:
   - Strategies used
   - Confidence scores
   - Costs and processing times
   - Escalation paths

### Generated Files

1. **`.refinery/profiles/*.json`**: DocumentProfile JSON files
   ```json
   {
     "doc_id": "CBE_ANNUAL_REPORT_2023-24",
     "origin_type": "mixed",
     "layout_complexity": "table_heavy",
     "domain_hint": "financial",
     "estimated_cost": "needs_layout_model",
     ...
   }
   ```

2. **`.refinery/extraction_ledger.jsonl`**: Extraction audit log
   ```json
   {"document_id": "...", "page_num": 1, "strategy_used": "fast_text", "confidence_score": 0.85, ...}
   {"document_id": "...", "page_num": 2, "strategy_used": "fast_text", "confidence_score": 0.85, ...}
   ```

## Web UI Screenshots

### Home Page (Upload)
- Upload form for new documents
- List of processed documents
- Strategy selection dropdown

### Triage View (`/triage/{doc_id}`)
- Document classification details
- Origin type, layout complexity, domain hint
- Full profile JSON display

### Extraction View (`/extraction/{doc_id}`)
- **Left Panel**: PDF viewer with page navigation
- **Right Panel**: Extracted JSON with strategy tabs
- Confidence scores and cost information
- Switch between strategies to compare results

### Ledger View (`/ledger`)
- Table of all extraction attempts
- Filter by document, strategy, success status
- Escalation paths and confidence scores

## Document Classes

### Class A: Annual Financial Reports
- **Example**: CBE Annual Report
- **Characteristics**: Mixed origin, complex layouts, tables, figures
- **Expected Strategy**: Layout-aware or vision (escalates from fast_text)

### Class B: Scanned Documents
- **Example**: Annual Report (scanned)
- **Characteristics**: Scanned images, no font metadata
- **Expected Strategy**: Vision-augmented (direct escalation)

### Class C: Technical Assessment Reports
- **Example**: FTA Performance Survey
- **Characteristics**: Mixed layouts, narrative + tables
- **Expected Strategy**: Layout-aware (may escalate from fast_text)

### Class D: Structured Data Reports
- **Example**: Tax Expenditure Report
- **Characteristics**: Native digital, table-heavy
- **Expected Strategy**: Fast text or layout-aware

## Troubleshooting

### Error: "Document not found"
- Ensure PDF files are in the correct `data/` subdirectories
- Check file names match exactly (case-sensitive)

### Error: "Docling import failed" or Memory Issues
**Symptoms**: 
- "The paging file is too small"
- "Unable to allocate X MiB for an array"
- "bad allocation" errors
- "std::bad_alloc" errors

**Solutions**:
1. **Increase Virtual Memory** (Windows):
   - Control Panel → System → Advanced System Settings
   - Performance Settings → Advanced → Virtual Memory
   - Increase paging file size (recommend 8-16 GB)

2. **Use Fast Text Only**:
   - The demo will automatically fall back to fast_text extraction
   - This works for native digital PDFs without complex layouts
   - Profile and cost analysis will still be generated

3. **Process Smaller Documents**:
   - Docling requires significant memory for large documents
   - Try with smaller PDFs (< 50 pages) first

4. **Skip Layout-Aware**:
   - The demo will continue with fast_text extraction
   - All other outputs (profiles, ledger) will still be generated

### Error: "OPENROUTER_API_KEY not set" or "401 Unauthorized"
**Symptoms**:
- "OPENROUTER_API_KEY is not set; VisionExtractor will fail at runtime"
- "401 Client Error: Unauthorized"

**Solutions**:
1. **Get API Key**:
   - Sign up at https://openrouter.ai
   - Generate an API key from your dashboard

2. **Set Environment Variable**:
   ```bash
   # Windows PowerShell
   $env:OPENROUTER_API_KEY="your_key_here"
   
   # Windows CMD
   set OPENROUTER_API_KEY=your_key_here
   
   # Linux/Mac
   export OPENROUTER_API_KEY=your_key_here
   ```

3. **Skip Vision Extraction**:
   - The demo will automatically fall back to other strategies
   - Vision is only needed for scanned documents
   - Fast text and layout-aware will still work

### Error: "All extraction strategies exhausted"
**Cause**: All strategies (fast_text, layout-aware, vision) failed

**Solutions**:
1. **Check Ledger**: Review `.refinery/extraction_ledger.jsonl` for error details
2. **Automatic Fallback**: The demo will attempt fast_text as a fallback
3. **Check Document**: Ensure PDF is not corrupted
4. **System Resources**: Ensure sufficient memory and disk space
5. **Partial Success**: Even if extraction fails, profiles and cost analysis are still generated

### No ledger entries created
- Check that `.refinery/` directory is writable
- Verify extraction completed successfully (check console output)
- Even failed extractions should create ledger entries (with `success: false`)

## Next Steps

After running the demo:

1. **Review Profiles**: Check `.refinery/profiles/` for classification results
2. **Analyze Ledger**: Review extraction strategies and costs in the ledger
3. **Explore Web UI**: Upload documents and view extraction results
4. **Run Tests**: Execute `pytest tests/test_all_classes.py` for comprehensive validation

## Cost Analysis Summary

Based on the demo output, you can analyze:

- **Strategy A (Fast Text)**: $0.00 per page (CPU-only)
- **Strategy B (Layout-aware)**: $0.0005 per page (Docling) or $0.0007 (MinerU)
- **Strategy C (Vision)**: ~$0.003 per page (API-based, varies by model)

For a typical 25-page document:
- Fast Text: $0.00
- Layout-aware: $0.0125
- Vision: $0.075

The router automatically selects the cheapest strategy that meets confidence thresholds.
