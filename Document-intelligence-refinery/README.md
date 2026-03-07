# Document Intelligence Refinery

A production-grade, multi-stage agentic pipeline for document extraction with provenance tracking. Transforms unstructured PDFs into structured, queryable, spatially-indexed knowledge.

## Overview

The Document Intelligence Refinery implements a 5-stage pipeline:

1. **Triage Agent** - Document classification and profiling
2. **Structure Extraction Layer** - Multi-strategy extraction with escalation logic
3. **Semantic Chunking Engine** - Logical Document Unit (LDU) generation
4. **PageIndex Builder** - Hierarchical navigation structure
5. **Query Interface Agent** - LangGraph agent with provenance tracking

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Install optional ML dependencies
pip install -e ".[dev]"
```

## Configuration

1. Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

2. Set `OPENROUTER_API_KEY` for vision model support (optional, only needed for scanned documents)

## Usage

### Streamlit UI

```bash
streamlit run streamlit_app/app.py
```

### Python API

```python
from pathlib import Path
from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.utils.rules import load_rules

# Load rules
rules = load_rules("config/extraction_rules.yaml")

# Stage 1: Triage
triage = TriageAgent(rules)
profile = triage.profile_document(Path("data/example.pdf"), persist=True)

# Stage 2: Extraction
router = ExtractionRouter(rules)
extracted, ledger_entry = router.run(Path("data/example.pdf"), profile)
```

## Project Structure

```
Document-intelligence-refinery/
├── src/
│   ├── models/          # Pydantic models
│   ├── agents/          # Pipeline stage agents
│   ├── strategies/      # Extraction strategies
│   ├── db/              # Vector store and fact table
│   └── utils/           # Utility modules
├── streamlit_app/       # Streamlit web UI
├── config/              # Configuration files
├── .refinery/           # Artifacts directory
└── pyproject.toml       # Project configuration
```

## Features

- **Multi-Strategy Extraction**: Fast text → Layout-aware → Vision-augmented
- **Confidence-Gated Escalation**: Automatic strategy escalation based on confidence
- **Semantic Chunking**: 5 core rules for preserving document structure
- **Provenance Tracking**: Full audit trail for every extracted fact
- **Free Model Support**: Uses free tier models from OpenRouter when possible

## License

MIT License
