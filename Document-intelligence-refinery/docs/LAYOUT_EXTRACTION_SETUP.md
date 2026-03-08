# Layout Extraction Setup (Docling, MinerU & OCR)

This guide explains how to set up and use Docling, MinerU, and OCR for layout-aware extraction (Strategy B).

## Installation

### OCR Support (Required for Strategy B)

Strategy B uses OCR (Tesseract) for scanned documents. Install:

1. **Tesseract OCR Engine** (system-level):
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki) or use `choco install tesseract`
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr` (Ubuntu/Debian) or `sudo yum install tesseract` (RHEL/CentOS)

2. **Python libraries** (already in dependencies):
   ```bash
   pip install pytesseract pdf2image
   ```

### Docling (Optional)

```bash
pip install docling
```

Docling is IBM's document understanding library that provides:
- OCR capabilities for scanned documents (built-in)
- Table structure extraction
- Layout-aware text extraction
- Multi-column document handling

### MinerU (Optional)

MinerU can be installed from GitHub:

```bash
pip install git+https://github.com/opendatalab/MinerU.git
```

Or if using a specific branch:
```bash
pip install git+https://github.com/opendatalab/MinerU.git@main
```

**Note**: MinerU may require additional dependencies. Check the [MinerU repository](https://github.com/opendatalab/MinerU) for full installation instructions.

## Configuration

The layout extraction engine is configured in `config/extraction_rules.yaml`:

```yaml
layout_strategy:
  engine: docling  # Options: "docling" or "mineru"
  
  mineru:
    output_dir: ".refinery/mineru_json"
    output_extension: ".mineru.json"
  
  docling:
    model_variant: "default"
```

## Usage

The LayoutExtractor automatically:
1. Checks if Docling/MinerU is installed
2. Uses the configured engine from `extraction_rules.yaml`
3. Falls back to enhanced pdfplumber if neither is available

## How It Works

### Strategy B Extraction Priority

Strategy B follows this priority order:

1. **Docling** (if configured and available)
   - DocumentConverter processes the PDF
   - OCR is enabled for scanned documents (`do_ocr=True`)
   - Table structure is extracted (`do_table_structure=True`)
   - Output is converted to our internal format

2. **MinerU** (if configured and available)
   - Checks for pre-processed MinerU JSON output
   - Parses MinerU JSON structure
   - Converts to our internal format
   - Falls back to OCR if output not found

3. **OCR (Tesseract)** (if document is scanned or Docling/MinerU unavailable)
   - Converts PDF pages to images (300 DPI)
   - Uses Tesseract OCR to extract text with bounding boxes
   - Groups words into lines and blocks
   - Creates text blocks with spatial information
   - **Automatically used for scanned documents**

4. **Enhanced pdfplumber** (fallback)
   - Better table detection
   - Layout-aware text grouping
   - Improved bounding box handling
   - **Only used if OCR is not available**

### OCR Features

- **Automatic detection**: OCR is automatically used for scanned documents
- **High quality**: Uses 300 DPI for better text recognition
- **Layout preservation**: Maintains bounding boxes and reading order
- **Confidence filtering**: Only includes text with >30% confidence
- **Free**: No API costs, runs locally

## Troubleshooting

### OCR Not Working
- **Tesseract not found**: Install Tesseract OCR engine (see Installation section)
- **Windows**: Make sure Tesseract is in your PATH or set `pytesseract.pytesseract.tesseract_cmd` in code
- **Test OCR**: `python -c "import pytesseract; print(pytesseract.get_tesseract_version())"`
- **PDF to image conversion fails**: Ensure `poppler` is installed (required by pdf2image)
  - Windows: Download from [poppler-windows](http://blog.alivate.com.au/poppler-windows/)
  - macOS: `brew install poppler`
  - Linux: `sudo apt-get install poppler-utils`

### Docling Not Found
- Install: `pip install docling`
- Check Python version compatibility
- Verify installation: `python -c "import docling; print(docling.__version__)"`

### MinerU Not Found
- Install from GitHub (see above)
- Ensure all MinerU dependencies are installed
- Pre-process documents with MinerU if needed

### Extraction Fails
- Check logs for specific error messages
- Verify PDF is not corrupted
- Try the fallback methods (OCR → pdfplumber)
- Check that the document has extractable content
- For scanned documents, ensure OCR is properly configured

## Performance

- **Docling**: Best for complex layouts, tables, and scanned documents
- **MinerU**: Good for academic papers and structured documents
- **pdfplumber (fallback)**: Fast but limited layout awareness
