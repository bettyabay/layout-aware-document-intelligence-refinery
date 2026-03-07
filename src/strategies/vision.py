"""
Vision extraction (Strategy C): VLM when API key set, else Tesseract OCR.
On VLM failure or no key → OCR. On OCR unavailable → raise (router falls back to Layout).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from io import BytesIO

try:
    import fitz
except Exception:
    fitz = None
import pdfplumber

from src.models import (
    BBox,
    DocumentProfile,
    ExtractedDocument,
    ExtractedMetadata,
    ExtractedPage,
    LDU,
    PageIndexNode,
    ProvenanceChain,
    StrategyName,
    TextBlock,
    content_hash_for_text,
)
from src.services.model_gateway import ModelGateway
from src.strategies.base import ExtractionStrategy


VISION_JSON_PROMPT = """Extract all text from this page image. Return a single JSON object with this exact shape, no other text:
{"blocks": [{"text": "...", "x0": 0, "y0": 0, "x1": 0, "y1": 0}, ...]}
Use coordinates in points (72 points = 1 inch). One block per paragraph or logical segment. If no text, return {"blocks": []}."""

DPI = 150


def _default_dpi(rules: dict) -> int:
    return int(rules.get("vision", {}).get("dpi", DPI))


def _build_base_doc_from_fitz(
    pdf_path: Path, profile: DocumentProfile, page_count: int
) -> ExtractedDocument:
    if fitz is None:
        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = [
                ExtractedPage(
                    page_number=pno,
                    width=float(p.width),
                    height=float(p.height),
                    text_blocks=[],
                    ldu_ids=[],
                )
                for pno, p in enumerate(pdf.pages, start=1)
            ]
    else:
        doc = fitz.open(str(pdf_path))
        try:
            pages = [
                ExtractedPage(
                    page_number=pno,
                    width=doc.load_page(pno - 1).rect.width,
                    height=doc.load_page(pno - 1).rect.height,
                    text_blocks=[],
                    ldu_ids=[],
                )
                for pno in range(1, page_count + 1)
            ]
        finally:
            doc.close()

    return ExtractedDocument(
        doc_id=profile.doc_id,
        document_name=profile.document_name,
        pages=pages,
        metadata=ExtractedMetadata(
            source_strategy=StrategyName.C,
            confidence_score=0.65,
            strategy_sequence=[StrategyName.C],
        ),
        ldus=[],
        provenance_chains=[],
    )


def _build_ldus_provenance_and_index(
    extracted: ExtractedDocument, profile: DocumentProfile
) -> None:
    ldus: list[LDU] = []
    provenance_chains: list[ProvenanceChain] = []
    page_nodes: list[PageIndexNode] = []

    for page in extracted.pages:
        page_ldu_ids: list[str] = []
        block_nodes: list[PageIndexNode] = []
        for i, block in enumerate(page.text_blocks):
            content_hash = content_hash_for_text(block.text)
            chain = ProvenanceChain(
                document_name=profile.document_name,
                page_number=page.page_number,
                bbox=block.bbox,
                content_hash=content_hash,
            )
            provenance_chains.append(chain)
            ldu = LDU(
                id=f"ldu-{block.id}",
                text=block.text,
                content_hash=content_hash,
                parent_section=f"page_{page.page_number}",
                page_refs=[page.page_number],
                provenance_chain=[chain],
            )
            ldus.append(ldu)
            page_ldu_ids.append(ldu.id)
            block_nodes.append(
                PageIndexNode(
                    id=block.id,
                    node_type="text_block",
                    label=(block.text[:80] if block.text else None),
                    page_number=page.page_number,
                    bbox=block.bbox,
                    children=[],
                )
            )
        page.ldu_ids = page_ldu_ids
        page_nodes.append(
            PageIndexNode(
                id=f"page-{page.page_number}",
                node_type="page",
                label=f"Page {page.page_number}",
                page_number=page.page_number,
                bbox=BBox(x0=0.0, y0=0.0, x1=page.width, y1=page.height),
                children=block_nodes,
            )
        )

    for i in range(1, len(ldus)):
        if ldus[i].page_refs and ldus[i - 1].page_refs and ldus[i].page_refs[0] == ldus[i - 1].page_refs[0]:
            ldus[i].previous_chunk_id = ldus[i - 1].id
            ldus[i - 1].next_chunk_id = ldus[i].id

    extracted.ldus = ldus
    extracted.provenance_chains = provenance_chains
    extracted.page_index = PageIndexNode(
        id=f"doc-{profile.doc_id}",
        node_type="document",
        label=profile.document_name,
        children=page_nodes,
    )


def _parse_vlm_blocks(raw: str, page_width: float, page_height: float) -> list[dict]:
    raw = (raw or "").strip()
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        return []
    try:
        data = json.loads(json_match.group(0))
        blocks = data.get("blocks") if isinstance(data, dict) else []
        if not isinstance(blocks, list):
            return []
        out = []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            text = (b.get("text") or "").strip()
            x0 = float(b.get("x0", 0))
            y0 = float(b.get("y0", 0))
            x1 = float(b.get("x1", page_width))
            y1 = float(b.get("y1", page_height))
            if x1 < x0:
                x1, x0 = x0, x1
            if y1 < y0:
                y1, y0 = y0, y1
            out.append({"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
        return out
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


class VisionExtractor(ExtractionStrategy):
    name = "C"

    @staticmethod
    def _render_page_png(pdf_path: Path, page_number: int, dpi: int = DPI) -> bytes:
        if fitz is not None:
            doc = fitz.open(str(pdf_path))
            try:
                page = doc.load_page(page_number - 1)
                pix = page.get_pixmap(dpi=dpi)
                return pix.tobytes("png")
            finally:
                doc.close()
        with pdfplumber.open(str(pdf_path)) as pdf:
            page = pdf.pages[page_number - 1]
            image = page.to_image(resolution=dpi).original
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()

    def _ocr_extract(
        self, pdf_path: Path, profile: DocumentProfile, rules: dict, base_doc: ExtractedDocument, dpi: int
    ) -> tuple[ExtractedDocument, float, float]:
        """Path B: pymupdf render @ dpi → PIL → pytesseract.image_to_string. One block per page. Raises if pytesseract/Pillow missing."""
        try:
            from PIL import Image
            import pytesseract
        except ImportError as e:
            raise ImportError(
                "pytesseract and Pillow are required for OCR fallback. Install: pip install pytesseract Pillow; install Tesseract system binary."
            ) from e

        for page in base_doc.pages:
            png_bytes = self._render_page_png(pdf_path, page.page_number, dpi=dpi)
            img = Image.open(BytesIO(png_bytes)).convert("RGB")
            text = (pytesseract.image_to_string(img) or "").strip()
            page.text_blocks.append(
                TextBlock(
                    id=f"p{page.page_number}-ocr",
                    text=text,
                    bbox=BBox(x0=0.0, y0=0.0, x1=page.width, y1=page.height),
                    reading_order=0,
                )
            )

        _build_ldus_provenance_and_index(base_doc, profile)
        return base_doc, 0.65, 0.0

    def extract(self, pdf_path: Path, profile: DocumentProfile, rules: dict) -> tuple[ExtractedDocument, float, float]:
        vision_cfg = rules.get("vision", {})
        per_page_cost = float(vision_cfg.get("estimated_cost_per_page_usd", 0.02))
        runtime_model_cfg = rules.get("runtime_model", {})
        max_cost = float(runtime_model_cfg.get("max_vision_budget_usd", vision_cfg.get("max_cost_per_doc_usd", 2.0)))
        prefer_vlm = vision_cfg.get("prefer_vlm", True)
        dpi = _default_dpi(rules)

        if fitz is None:
            with pdfplumber.open(str(pdf_path)) as pdf:
                page_count = len(pdf.pages)
        else:
            doc = fitz.open(str(pdf_path))
            page_count = doc.page_count
            doc.close()

        base_doc = _build_base_doc_from_fitz(pdf_path, profile, page_count)
        gateway = ModelGateway(rules, runtime_config=runtime_model_cfg)
        selected_override = runtime_model_cfg.get("vision_override") if not runtime_model_cfg.get("auto_select", True) else None
        provider, model_name = gateway.select_vision_model(override=selected_override)
        has_vision_api = gateway.providers.get(provider) is not None and gateway.is_paid_provider(provider)
        paid_provider = gateway.is_paid_provider(provider)

        if has_vision_api and prefer_vlm:
            total_cost = 0.0
            effective_pages = page_count
            if paid_provider and per_page_cost > 0:
                effective_pages = max(1, min(page_count, int(max_cost // per_page_cost)))
            vlm_ok = True
            for pno in range(1, effective_pages + 1):
                if paid_provider and total_cost >= max_cost:
                    break
                try:
                    image_bytes = self._render_page_png(pdf_path, pno, dpi=dpi)
                    result = gateway.generate_vision(
                        provider=provider,
                        model_name=model_name,
                        prompt=VISION_JSON_PROMPT,
                        image_bytes=image_bytes,
                    )
                    total_cost += result.estimated_cost_usd
                    page = base_doc.pages[pno - 1]
                    blocks = _parse_vlm_blocks(result.text, page.width, page.height)
                    if blocks:
                        for i, b in enumerate(blocks):
                            page.text_blocks.append(
                                TextBlock(
                                    id=f"p{pno}-vlm{i}",
                                    text=b["text"],
                                    bbox=BBox(x0=b["x0"], y0=b["y0"], x1=b["x1"], y1=b["y1"]),
                                    reading_order=i,
                                )
                            )
                    else:
                        vlm_ok = False
                        break
                except Exception:
                    vlm_ok = False
                    break

            if vlm_ok and any(p.text_blocks for p in base_doc.pages):
                base_doc.pages = base_doc.pages[:effective_pages]
                _build_ldus_provenance_and_index(base_doc, profile)
                confidence = 0.65
                if paid_provider and total_cost >= max_cost:
                    confidence = max(0.45, confidence - 0.15)
                base_doc.metadata.confidence_score = confidence
                return base_doc, confidence, min(total_cost, max_cost) if paid_provider else 0.0

            for p in base_doc.pages:
                p.text_blocks.clear()

        return self._ocr_extract(pdf_path, profile, rules, base_doc, dpi)
