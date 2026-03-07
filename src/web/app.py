"""FastAPI web application for extraction visualization.

This module provides a web UI for:
- Uploading and processing documents
- Viewing DocumentProfile (triage results)
- Side-by-side PDF and extracted JSON visualization
- Viewing extraction ledger
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from starlette.requests import Request

from src.agents.chunker import ChunkingEngine
from src.agents.extractor import ExtractionRouter
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryAgent
from src.agents.triage import TriageAgent
from src.models.document_profile import DocumentProfile
from src.models.ldu import LDU
from src.utils.fact_table import FactTable
from src.utils.vector_store import VectorStore

# Add custom Jinja2 filters
def tojson_pretty(value):
    """Convert value to pretty-printed JSON."""
    return json.dumps(value, indent=2)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Document Intelligence Refinery", version="0.1.0")

# Setup paths
BASE_DIR = Path(__file__).parent.parent.parent
TEMPLATES_DIR = BASE_DIR / "src" / "web" / "templates"
STATIC_DIR = BASE_DIR / "src" / "web" / "static"
REFINERY_DIR = BASE_DIR / ".refinery"
PROFILES_DIR = REFINERY_DIR / "profiles"
CHUNKS_DIR = REFINERY_DIR / "chunks"
VALIDATION_LOG = REFINERY_DIR / "chunk_validation.log"
LEDGER_PATH = REFINERY_DIR / "extraction_ledger.jsonl"
UPLOAD_DIR = REFINERY_DIR / "uploads"

# Create directories
REFINERY_DIR.mkdir(exist_ok=True)
PROFILES_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Setup templates and static files
jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
jinja_env.filters["tojson_pretty"] = tojson_pretty
jinja_env.filters["tojson"] = lambda v: json.dumps(v)

def render_template(template_name: str, context: dict) -> str:
    """Render a Jinja2 template."""
    template = jinja_env.get_template(template_name)
    return template.render(**context)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize agents
triage_agent = TriageAgent(profiles_dir=PROFILES_DIR)
extraction_router = ExtractionRouter()
chunking_engine = ChunkingEngine()

# Initialize query agent components (lazy initialization)
_query_agent: Optional[QueryAgent] = None
_vector_store: Optional[VectorStore] = None
_fact_table: Optional[FactTable] = None


def get_query_agent() -> QueryAgent:
    """Get or create the query agent instance."""
    global _query_agent, _vector_store, _fact_table
    
    if _query_agent is None:
        vector_store_dir = REFINERY_DIR / "vector_store"
        fact_table_path = REFINERY_DIR / "facts.db"
        pageindex_dir = REFINERY_DIR / "pageindex"
        
        _vector_store = VectorStore(persist_directory=vector_store_dir)
        _fact_table = FactTable(db_path=fact_table_path)
        
        _query_agent = QueryAgent(
            vector_store=_vector_store,
            fact_table=_fact_table,
            pageindex_dir=pageindex_dir,
            llm_api_key=None,  # Will read from env
        )
    
    return _query_agent


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page with document upload form."""
    # Get list of processed documents
    processed_docs = []
    if PROFILES_DIR.exists():
        for profile_file in PROFILES_DIR.glob("*.json"):
            try:
                profile_data = json.loads(profile_file.read_text(encoding="utf-8"))
                processed_docs.append(
                    {
                        "doc_id": profile_data.get("doc_id", profile_file.stem),
                        "path": profile_data.get("metadata", {}).get("path", ""),
                    }
                )
            except Exception:
                continue

    return HTMLResponse(
        render_template("index.html", {"request": request, "processed_docs": processed_docs})
    )


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    strategy: Optional[str] = Form(None),
):
    """Upload and process a document."""
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Uploaded file: {file_path}")

        # Run triage
        profile = triage_agent.classify_document(file_path)

        # Run extraction if requested
        extracted_doc = None
        if strategy:
            # Override strategy selection if specified
            extracted_doc = extraction_router.extract(profile, str(file_path))

        return JSONResponse(
            {
                "success": True,
                "doc_id": profile.doc_id,
                "profile": profile.model_dump(),
                "extracted": extracted_doc.model_dump() if extracted_doc else None,
            }
        )
    except Exception as exc:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/triage/{doc_id}", response_class=HTMLResponse)
async def view_triage(request: Request, doc_id: str):
    """View DocumentProfile for a document."""
    profile_path = PROFILES_DIR / f"{doc_id}.json"
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail=f"Profile not found for {doc_id}")

    try:
        profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
        profile = DocumentProfile.model_validate(profile_data)

        return HTMLResponse(
            render_template(
                "triage.html",
                {
                    "request": request,
                    "doc_id": doc_id,
                    "profile": profile,
                    "profile_json": profile.to_json(indent=2),
                },
            )
        )
    except Exception as exc:
        logger.exception("Failed to load profile")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/extraction/{doc_id}", response_class=HTMLResponse)
async def view_extraction(
    request: Request,
    doc_id: str,
):
    """View extraction results side-by-side with PDF."""
    # Load profile
    profile_path = PROFILES_DIR / f"{doc_id}.json"
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail=f"Profile not found for {doc_id}")

    try:
        profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
        profile = DocumentProfile.model_validate(profile_data)
        document_path = profile.metadata.path

        if not Path(document_path).exists():
            raise HTTPException(
                status_code=404, detail=f"Document file not found: {document_path}"
            )

        # Get extraction results from ledger
        extraction_results = []
        if LEDGER_PATH.exists():
            with LEDGER_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get("document_id") == doc_id:
                            extraction_results.append(entry)

        # Group by strategy
        strategies_data = {}
        for entry in extraction_results:
            strat = entry.get("strategy_used", "unknown")
            if strat not in strategies_data:
                strategies_data[strat] = {
                    "confidence": entry.get("confidence_score", 0.0),
                    "cost": entry.get("cost_estimate", {}),
                    "processing_time": entry.get("processing_time_seconds", 0.0),
                    "escalation_path": entry.get("escalation_path", []),
                    "success": entry.get("success", False),
                }

        # Get strategy from query parameter
        strategy = request.query_params.get("strategy")
        
        # If specific strategy requested, try to get extraction result
        extracted_json = None
        confidence = 0.0
        strategy_used = None

        if strategy and strategy in strategies_data:
            strategy_used = strategy
            confidence = strategies_data[strategy]["confidence"]
            # Try to re-extract with the specified strategy for display
            try:
                from src.strategies.fast_text import FastTextExtractor
                from src.strategies.layout_aware import LayoutExtractor
                from src.strategies.vision_augmented import VisionExtractor

                if strategy == "fast_text":
                    extractor = FastTextExtractor()
                elif strategy == "layout_aware":
                    extractor = LayoutExtractor()
                elif strategy == "vision_augmented":
                    extractor = VisionExtractor()
                else:
                    extractor = None

                if extractor:
                    extracted_doc = extractor.extract(document_path)
                    extracted_json = extracted_doc.model_dump_json(indent=2)
            except Exception as exc:
                logger.warning(f"Failed to re-extract with {strategy}: {exc}")

        # Get PDF URL for serving
        pdf_url = f"/api/pdf/{doc_id}"

        return HTMLResponse(
            render_template(
                "extraction.html",
                {
                    "request": request,
                    "doc_id": doc_id,
                    "document_path": document_path,
                    "pdf_url": pdf_url,
                    "profile": profile,
                    "strategies": list(strategies_data.keys()),
                    "strategies_data": strategies_data,
                    "current_strategy": strategy or (list(strategies_data.keys())[0] if strategies_data else None),
                    "extracted_json": extracted_json,
                    "confidence": confidence,
                    "strategy_used": strategy_used or "none",
                },
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to load extraction view")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/ledger", response_class=HTMLResponse)
async def view_ledger(request: Request, limit: int = 100):
    """View extraction ledger."""
    from datetime import datetime
    
    entries = []
    if LEDGER_PATH.exists():
        try:
            with LEDGER_PATH.open("r", encoding="utf-8") as f:
                lines = f.readlines()
                # Get last N entries
                for line in lines[-limit:]:
                    if line.strip():
                        entry = json.loads(line)
                        # Format timestamp
                        if entry.get("timestamp"):
                            try:
                                ts = float(entry["timestamp"])
                                entry["timestamp_formatted"] = datetime.fromtimestamp(ts).strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                            except (ValueError, TypeError, OSError):
                                entry["timestamp_formatted"] = str(entry.get("timestamp", "N/A"))
                        else:
                            entry["timestamp_formatted"] = "N/A"
                        entries.append(entry)
        except Exception as exc:
            logger.warning(f"Failed to read ledger: {exc}")

    return HTMLResponse(
        render_template("ledger.html", {"request": request, "entries": entries, "limit": limit})
    )


@app.get("/api/pdf/{doc_id}")
async def serve_pdf(doc_id: str):
    """Serve PDF file for a document."""
    from fastapi.responses import FileResponse
    
    profile_path = PROFILES_DIR / f"{doc_id}.json"
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail=f"Profile not found for {doc_id}")

    try:
        profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
        profile = DocumentProfile.model_validate(profile_data)
        document_path = Path(profile.metadata.path)

        if not document_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Document file not found: {document_path}"
            )

        return FileResponse(
            str(document_path),
            media_type="application/pdf",
            filename=document_path.name,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to serve PDF")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/extraction/{doc_id}")
async def get_extraction_json(doc_id: str, strategy: str):
    """API endpoint to get extraction JSON for a document and strategy."""
    profile_path = PROFILES_DIR / f"{doc_id}.json"
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail=f"Profile not found for {doc_id}")

    try:
        profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
        profile = DocumentProfile.model_validate(profile_data)
        document_path = profile.metadata.path

        if not Path(document_path).exists():
            raise HTTPException(
                status_code=404, detail=f"Document file not found: {document_path}"
            )

        # Extract with specified strategy
        from src.strategies.fast_text import FastTextExtractor
        from src.strategies.layout_aware import LayoutExtractor
        from src.strategies.vision_augmented import VisionExtractor

        if strategy == "fast_text":
            extractor = FastTextExtractor()
        elif strategy == "layout_aware":
            extractor = LayoutExtractor()
        elif strategy == "vision_augmented":
            extractor = VisionExtractor()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")

        extracted_doc = extractor.extract(document_path)
        return JSONResponse(extracted_doc.model_dump())
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get extraction JSON")
        raise HTTPException(status_code=500, detail=str(exc))


def load_chunks_for_document(doc_id: str) -> list[LDU]:
    """Load chunks for a document, either from storage or by re-chunking.
    
    Args:
        doc_id: Document ID.
        
    Returns:
        List of LDUs.
    """
    chunks_path = CHUNKS_DIR / f"{doc_id}.json"
    
    # Try to load from storage
    if chunks_path.exists():
        try:
            chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
            return [LDU.model_validate(chunk) for chunk in chunks_data]
        except Exception as exc:
            logger.warning(f"Failed to load chunks from {chunks_path}: {exc}")
    
    # If not found, try to re-chunk from extracted document
    profile_path = PROFILES_DIR / f"{doc_id}.json"
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail=f"Profile not found for {doc_id}")
    
    try:
        profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
        profile = DocumentProfile.model_validate(profile_data)
        document_path = Path(profile.metadata.path)
        
        if not document_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Document file not found: {document_path}"
            )
        
        # Extract and chunk
        extracted_doc = extraction_router.extract(profile, str(document_path))
        chunks = chunking_engine.chunk(extracted_doc)
        
        # Save chunks for future use
        chunks_data = [chunk.model_dump() for chunk in chunks]
        chunks_path.write_text(json.dumps(chunks_data, indent=2), encoding="utf-8")
        logger.info(f"Saved {len(chunks)} chunks to {chunks_path}")
        
        return chunks
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to load or generate chunks")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/chunks/{doc_id}", response_class=HTMLResponse)
async def view_chunks(
    request: Request,
    doc_id: str,
    chunk_type: Optional[str] = None,
    section: Optional[str] = None,
    search: Optional[str] = None,
):
    """View all chunks for a document with filters."""
    try:
        chunks = load_chunks_for_document(doc_id)
        
        # Apply filters
        if chunk_type:
            chunks = [c for c in chunks if c.chunk_type == chunk_type]
        if section:
            chunks = [c for c in chunks if c.parent_section == section]
        if search:
            search_lower = search.lower()
            chunks = [c for c in chunks if search_lower in c.content.lower()]
        
        # Get unique sections and chunk types for filters
        sections = sorted(set(c.parent_section for c in chunks if c.parent_section))
        chunk_types = sorted(set(c.chunk_type for c in chunks))
        
        # Build section hierarchy
        section_tree = {}
        for chunk in chunks:
            if chunk.parent_section:
                if chunk.parent_section not in section_tree:
                    section_tree[chunk.parent_section] = []
                section_tree[chunk.parent_section].append(chunk.content_hash)
        
        return HTMLResponse(
            render_template(
                "chunks.html",
                {
                    "request": request,
                    "doc_id": doc_id,
                    "chunks": chunks,
                    "sections": sections,
                    "chunk_types": chunk_types,
                    "section_tree": section_tree,
                    "current_filters": {
                        "chunk_type": chunk_type,
                        "section": section,
                        "search": search,
                    },
                    "total_chunks": len(chunks),
                },
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to load chunks")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/chunk/{chunk_id}", response_class=HTMLResponse)
async def view_chunk_detail(request: Request, chunk_id: str, doc_id: str):
    """View single chunk with full metadata and provenance."""
    try:
        chunks = load_chunks_for_document(doc_id)
        chunk = next((c for c in chunks if c.content_hash == chunk_id), None)
        
        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        
        # Find related chunks (children, cross-references)
        related_chunks = []
        for ref in chunk.cross_references:
            related = next((c for c in chunks if c.content_hash == ref.target_id), None)
            if related:
                related_chunks.append(related)
        
        child_chunks = [c for c in chunks if c.content_hash in chunk.children]
        
        return HTMLResponse(
            render_template(
                "chunk_detail.html",
                {
                    "request": request,
                    "doc_id": doc_id,
                    "chunk": chunk,
                    "related_chunks": related_chunks,
                    "child_chunks": child_chunks,
                },
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to load chunk detail")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/validation/{doc_id}", response_class=HTMLResponse)
async def view_validation(request: Request, doc_id: str):
    """View validation results for a document."""
    try:
        chunks = load_chunks_for_document(doc_id)
        
        # Run validation
        overall_success, results, violations = chunking_engine.validate_chunks_detailed(chunks)
        
        # Load validation log if exists
        validation_entries = []
        if VALIDATION_LOG.exists():
            try:
                with VALIDATION_LOG.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip() and doc_id in line:
                            validation_entries.append(json.loads(line))
            except Exception as exc:
                logger.warning(f"Failed to read validation log: {exc}")
        
        # Group violations by rule
        violations_by_rule = {}
        for violation in violations:
            rule = violation.get("rule", "unknown")
            if rule not in violations_by_rule:
                violations_by_rule[rule] = []
            violations_by_rule[rule].append(violation)
        
        return HTMLResponse(
            render_template(
                "validation.html",
                {
                    "request": request,
                    "doc_id": doc_id,
                    "overall_success": overall_success,
                    "results": results,
                    "violations": violations,
                    "violations_by_rule": violations_by_rule,
                    "validation_entries": validation_entries,
                    "total_chunks": len(chunks),
                },
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to load validation results")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/chunks/{doc_id}")
async def get_chunks_json(doc_id: str):
    """API endpoint to get chunks as JSON."""
    try:
        chunks = load_chunks_for_document(doc_id)
        return JSONResponse([chunk.model_dump() for chunk in chunks])
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get chunks JSON")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/query/{doc_id}", response_class=HTMLResponse)
async def query_document(request: Request, doc_id: str):
    """Query interface for a document."""
    try:
        # Check if document exists
        profile_path = PROFILES_DIR / f"{doc_id}.json"
        if not profile_path.exists():
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
        doc_name = profile_data.get("metadata", {}).get("path", doc_id)
        
        # Check if chunks exist
        chunks_path = CHUNKS_DIR / f"{doc_id}_chunks.json"
        has_chunks = chunks_path.exists()
        
        return HTMLResponse(
            render_template(
                "query.html",
                {
                    "request": request,
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "has_chunks": has_chunks,
                },
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to load query interface")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/query/{doc_id}")
async def execute_query(doc_id: str, query: str = Form(...)):
    """Execute a query on a document."""
    try:
        # Get query agent
        query_agent = get_query_agent()
        
        # Get document name
        profile_path = PROFILES_DIR / f"{doc_id}.json"
        if profile_path.exists():
            profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
            doc_name = profile_data.get("metadata", {}).get("path", doc_id)
        else:
            doc_name = doc_id
        
        # Execute query
        result = query_agent.query(query=query, doc_id=doc_id, doc_name=doc_name)
        
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("Failed to execute query")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/setup-query/{doc_id}")
async def setup_query_for_document(doc_id: str):
    """Set up query agent for a document (load chunks into vector store)."""
    try:
        # Load chunks
        chunks = load_chunks_for_document(doc_id)
        
        if not chunks:
            raise HTTPException(status_code=404, detail=f"No chunks found for {doc_id}")
        
        # Get document name
        profile_path = PROFILES_DIR / f"{doc_id}.json"
        if profile_path.exists():
            profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
            doc_name = profile_data.get("metadata", {}).get("path", doc_id)
        else:
            doc_name = doc_id
        
        # Get query agent components
        query_agent = get_query_agent()
        
        # Add chunks to vector store
        query_agent.vector_store.add_ldus(doc_id=doc_id, doc_name=doc_name, ldus=chunks)
        
        # Extract facts
        fact_count = query_agent.fact_table.extract_facts_from_ldus(
            doc_id=doc_id, doc_name=doc_name, ldus=chunks
        )
        
        return JSONResponse({
            "success": True,
            "message": f"Set up query agent for {doc_id}",
            "chunks_added": len(chunks),
            "facts_extracted": fact_count,
        })
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to set up query agent")
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
