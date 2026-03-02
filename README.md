# Layout-Aware Document Intelligence Refinery

A production-grade, multi-stage agentic pipeline for document extraction with provenance tracking. Transforms unstructured PDFs into structured, queryable, spatially-indexed knowledge with confidence-gated escalation strategies.

## Overview

The Document Intelligence Refinery is a 5-stage agentic pipeline designed to solve the "last mile" problem in enterprise AI deployments: extracting structured, queryable data from unstructured documents while preserving spatial provenance and semantic context.

### The Problem

Traditional document processing approaches suffer from three critical failure modes:

1. **Structure Collapse**: OCR flattens two-column layouts, breaks tables, and drops headers
2. **Context Poverty**: Naive chunking severs logical units, producing hallucinated answers
3. **Provenance Blindness**: Most pipelines cannot answer "Where exactly in the 400-page report does this number come from?"

### The Solution

This refinery implements a multi-stage pipeline that:
- **Classifies** documents before extraction to select optimal strategies
- **Extracts** content using confidence-gated escalation (fast → layout-aware → vision-augmented)
- **Chunks** semantically, preserving table integrity and section hierarchy
- **Indexes** documents with hierarchical PageIndex navigation
- **Queries** with full provenance chains, enabling audit and verification

## Architecture: The 5-Stage Pipeline

```
INPUT (PDFs, Docs, Images)
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Triage Agent                                   │
│ - Document classification                                │
│ - Origin type detection (digital vs scanned)            │
│ - Layout complexity analysis                             │
│ - Domain hint classification                             │
│ - Extraction cost estimation                            │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 2: Structure Extraction Layer                    │
│ - Strategy A: Fast Text (pdfplumber)                    │
│ - Strategy B: Layout-Aware (MinerU/Docling)            │
│ - Strategy C: Vision-Augmented (VLM via OpenRouter)     │
│ - Confidence-gated escalation guard                     │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 3: Semantic Chunking Engine                       │
│ - Logical Document Units (LDUs)                         │
│ - Table integrity preservation                          │
│ - Figure-caption binding                                │
│ - Section hierarchy maintenance                         │
│ - Cross-reference resolution                            │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 4: PageIndex Builder                              │
│ - Hierarchical section tree                            │
│ - LLM-generated summaries                              │
│ - Named entity extraction                               │
│ - Data type classification                             │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 5: Query Interface Agent                          │
│ - PageIndex navigation                                  │
│ - Semantic vector search                                │
│ - Structured SQL queries                                │
│ - Provenance chain generation                           │
└─────────────────────────────────────────────────────────┘
    ↓
OUTPUT (Structured JSON, RAG-ready vectors, SQL facts, Provenance)
```

## Key Features

### 🎯 Multi-Strategy Extraction

- **Fast Text Extraction**: For native digital PDFs with simple layouts
- **Layout-Aware Extraction**: For multi-column documents and tables using MinerU/Docling
- **Vision-Augmented Extraction**: For scanned documents and complex layouts using VLMs

### 🛡️ Confidence-Gated Escalation

Every extraction strategy measures confidence and automatically escalates to more sophisticated (and expensive) strategies when needed, preventing "garbage in, hallucination out" failures.

### 📊 Semantic Chunking

The chunking engine respects five critical rules:
1. Table cells never split from header rows
2. Figure captions always stored as metadata
3. Numbered lists kept as single Logical Document Units
4. Section headers stored as parent metadata
5. Cross-references resolved and stored as relationships

### 🗺️ PageIndex Navigation

Inspired by VectifyAI's PageIndex, provides hierarchical navigation that allows LLMs to locate information without reading entire documents—dramatically improving retrieval precision for section-specific queries.

### 🔍 Full Provenance Tracking

Every extracted fact carries:
- Document name and page number
- Bounding box coordinates
- Content hash for verification
- Confidence scores

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or poetry for package management

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd layout-aware-document-intelligence-refinery
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install optional ML dependencies** (for MinerU/Docling):
   ```bash
   pip install -e ".[ml]"
   ```
   
   Note: MinerU and Docling may require additional setup. See their respective repositories:
   - [MinerU](https://github.com/opendatalab/MinerU)
   - [Docling](https://github.com/DS4SD/docling)

5. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

6. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Model Configuration
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_VLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Extraction Configuration
EXTRACTION_CONFIDENCE_THRESHOLD=0.7
MAX_VISION_COST_PER_DOCUMENT=5.0
```

### Extraction Rules

Modify `rubric/extraction_rules.yaml` to adjust:
- Confidence thresholds
- Chunking rules
- Cost limits
- Domain classification keywords

## Usage

### Basic Pipeline Execution

```python
from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from pathlib import Path

# Stage 1: Triage
triage = TriageAgent(profiles_dir=Path(".refinery/profiles"))
profile = triage.classify_document(Path("data/example.pdf"))

# Stage 2: Extraction
router = ExtractionRouter()
extracted = router.extract(Path("data/example.pdf"), profile)

# Stage 3: Chunking
chunker = ChunkingEngine()
ldus = chunker.chunk(extracted)

# Stage 4: PageIndex
indexer = PageIndexBuilder()
pageindex = indexer.build(extracted, ldus)

# Stage 5: Query (see query_agent.py)
```

### Web Interface

Start the FastAPI web server:

```bash
uvicorn src.web.app:app --reload
```

Access the web UI at `http://localhost:8000`

### Command Line

```bash
# Process a single document
python -m src.cli process --input data/example.pdf

# Process a directory
python -m src.cli process --input data/class_a/ --output .refinery/

# Query a processed document
python -m src.cli query --doc-id doc_123 --query "What is the revenue for Q3?"
```

## Project Structure

```
layout-aware-document-intelligence-refinery/
├── README.md                 # This file
├── pyproject.toml            # Project configuration and dependencies
├── .gitignore               # Git ignore rules
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
├── docker-compose.yml       # Docker Compose configuration (optional)
├── Dockerfile               # Docker image definition
│
├── rubric/                  # Configuration and evaluation
│   ├── extraction_rules.yaml      # Extraction thresholds and rules
│   └── evaluation_criteria.md     # Evaluation rubric
│
├── src/                     # Source code
│   ├── models/              # Pydantic data models
│   │   ├── document_profile.py    # Document classification profile
│   │   ├── extracted_document.py  # Extracted content structure
│   │   ├── ldu.py                 # Logical Document Units
│   │   ├── page_index.py          # PageIndex navigation tree
│   │   └── provenance.py          # Provenance tracking models
│   │
│   ├── agents/              # Pipeline stage agents
│   │   ├── triage.py        # Stage 1: Document classification
│   │   ├── extractor.py     # Stage 2: Extraction router
│   │   ├── chunker.py       # Stage 3: Semantic chunking
│   │   ├── indexer.py       # Stage 4: PageIndex builder
│   │   └── query_agent.py   # Stage 5: Query interface
│   │
│   ├── strategies/          # Extraction strategies
│   │   ├── base.py          # Abstract base class
│   │   ├── fast_text.py     # Strategy A: Fast text extraction
│   │   ├── layout_aware.py  # Strategy B: Layout-aware extraction
│   │   └── vision_augmented.py # Strategy C: Vision-augmented extraction
│   │
│   ├── utils/               # Utility modules
│   │   ├── confidence_scorer.py  # Confidence scoring logic
│   │   ├── budget_guard.py       # Cost budget enforcement
│   │   ├── adapters.py           # Format adapters (Docling, MinerU)
│   │   └── validators.py         # Data validation utilities
│   │
│   └── web/                 # Web interface
│       ├── app.py           # FastAPI application
│       ├── templates/       # Jinja2 templates
│       └── static/          # Static assets
│
├── tests/                   # Test suite
│   ├── test_triage.py       # Triage agent tests
│   ├── test_extractors.py   # Extraction strategy tests
│   ├── test_chunker.py      # Chunking engine tests
│   └── test_integration.py  # End-to-end integration tests
│
├── .refinery/               # Pipeline artifacts (gitignored)
│   ├── profiles/            # Document profiles (JSON)
│   ├── extraction_ledger.jsonl  # Extraction audit log
│   └── pageindex/           # PageIndex trees (JSON)
│
├── data/                    # Document corpus (gitignored)
│   ├── class_a/             # Annual financial reports
│   ├── class_b/             # Scanned government/legal docs
│   ├── class_c/             # Technical assessment reports
│   └── class_d/             # Structured data reports
│
├── docs/                    # Documentation
│   ├── DOMAIN_NOTES.md      # Domain knowledge and failure modes
│   ├── ARCHITECTURE.md      # Architecture deep dive
│   └── API_REFERENCE.md     # API documentation
│
└── notebooks/               # Jupyter notebooks
    └── exploration.ipynb    # Domain exploration and analysis
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_triage.py
```

### Code Quality

```bash
# Format code
black src tests

# Sort imports
isort src tests

# Type checking
mypy src

# Linting
ruff check src tests
```

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality. Hooks run automatically on commit and check:
- Code formatting (Black)
- Import sorting (isort)
- Linting (Ruff)
- Type checking (mypy)
- YAML/JSON/TOML validation

## Dependencies

### Core Dependencies

- **pydantic**: Data validation and settings management
- **pdfplumber**: PDF text extraction
- **pymupdf**: PDF processing and rendering
- **langchain**: LLM orchestration framework
- **langgraph**: Agentic workflow construction
- **chromadb**: Vector database for embeddings
- **fastapi**: Web framework for API and UI
- **uvicorn**: ASGI server
- **jinja2**: Template engine

### Optional ML Dependencies

- **docling**: IBM Research's document understanding framework
- **mineru**: OpenDataLab's PDF parsing framework
- **torch**: PyTorch for ML models
- **transformers**: Hugging Face transformers

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure code quality (`pytest`, `black`, `mypy`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

This project is inspired by:
- [MinerU](https://github.com/opendatalab/MinerU) - OpenDataLab's PDF parsing framework
- [Docling](https://github.com/DS4SD/docling) - IBM Research's document understanding
- [PageIndex](https://github.com/VectifyAI/PageIndex) - VectifyAI's navigation index
- [Chunkr](https://github.com/lumina-ai-inc/chunkr) - YC S24's RAG-optimized chunking
- [Marker](https://github.com/VikParuchuri/marker) - High-accuracy PDF-to-Markdown converter

## Roadmap

- [ ] Complete Stage 1: Triage Agent implementation
- [ ] Complete Stage 2: Multi-strategy extraction with escalation
- [ ] Complete Stage 3: Semantic chunking engine
- [ ] Complete Stage 4: PageIndex builder
- [ ] Complete Stage 5: Query agent with provenance
- [ ] Web UI for document upload and querying
- [ ] Docker containerization
- [ ] Performance benchmarking
- [ ] Cost optimization strategies

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Built for the FDE Program Week 3 Challenge: The Document Intelligence Refinery**
