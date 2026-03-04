# Demo Scripts

This directory contains scripts for the interim submission demo.

## Files

- **`interim_demo.py`**: Main demo script that processes one document from each class
- **`ensure_profiles.py`**: Helper script to ensure minimum document count

## Quick Start

1. **Run the main demo**:
   ```bash
   python demo/interim_demo.py
   ```

2. **Ensure minimum profiles** (if needed):
   ```bash
   python demo/ensure_profiles.py
   ```

## Output

The demo scripts generate:

- **`.refinery/profiles/*.json`**: DocumentProfile files for each processed document
- **`.refinery/extraction_ledger.jsonl`**: Extraction audit log with all attempts

## Requirements

- Python 3.11+
- All project dependencies installed
- PDF documents in `data/` directory:
  - `data/class_a/CBE_ANNUAL_REPORT_2023-24.pdf`
  - `data/class_b/Annual_Report_JUNE-2023.pdf`
  - `data/class_c/fta_performance_survey_final_report_2022.pdf`
  - `data/class_d/tax_expenditure_ethiopia_2021_22.pdf`

## Note on 12-Document Requirement

The interim submission requires at least 12 processed documents (3 per class). 

**Current Setup**: The `data/` directory contains 1 document per class (4 total).

**To Meet Requirement**: You have two options:

1. **Add More Documents**: Place 3 different PDF files per class in the `data/` subdirectories
2. **Use Existing Documents**: The `ensure_profiles.py` script will process the available documents, but you'll only have 4 unique profiles

For a complete demo with 12 unique documents, add 2 more PDFs to each class directory.
