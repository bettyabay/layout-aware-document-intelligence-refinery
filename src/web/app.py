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

from src.agents.extractor import ExtractionRouter
from src.agents.triage import TriageAgent
from src.models.document_profile import DocumentProfile

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
LEDGER_PATH = REFINERY_DIR / "extraction_ledger.jsonl"
UPLOAD_DIR = REFINERY_DIR / "uploads"

# Create directories
REFINERY_DIR.mkdir(exist_ok=True)
PROFILES_DIR.mkdir(parents=True, exist_ok=True)
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
triage_agent = TriageAgent()
extraction_router = ExtractionRouter()


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
