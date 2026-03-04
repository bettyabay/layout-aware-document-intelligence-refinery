"""Vision-augmented extraction strategy using OpenRouter-hosted VLMs (Strategy C).

This strategy is designed as the high-cost, high-fidelity fallback for cases
where fast text and layout-aware models are insufficient – particularly for
scanned PDFs, handwriting, or highly degraded layouts.

It works page-by-page:

* Renders each page to an image (via ``pdf2image``).
* Sends the image plus a structured extraction prompt to a vision model via
  OpenRouter's HTTP API.
* Expects a JSON response with text blocks, tables, and figures.
* Normalises that JSON into the internal ``ExtractedDocument`` schema.

Notes:
    - This implementation assumes an OpenAI-compatible JSON API exposed by
      OpenRouter at ``https://openrouter.ai/api/v1/chat/completions`` and an
      API key provided via the ``OPENROUTER_API_KEY`` environment variable.
    - The actual model identifier (e.g. ``openai/gpt-4o-mini`` or
      ``google/gemini-pro-vision``) is configurable.
"""

from __future__ import annotations

import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import pdfplumber
import requests
from pdf2image import convert_from_path

from src.models.document_profile import DocumentProfile
from src.models.extracted_document import BoundingBox, ExtractedDocument, Figure, Table, TextBlock
from src.strategies.base import ExtractionStrategy
from src.utils.budget_guard import BudgetGuard


logger = logging.getLogger(__name__)


VISION_PROMPT = """
You are a document understanding engine.

Extract all content from this document page as JSON with the following shape:

{
  "text_blocks": [
    {"content": "...", "bbox": [x0, y0, x1, y1], "page_num": <int>}
  ],
  "tables": [
    {
      "headers": ["..."],
      "rows": [["...", "..."]],
      "bbox": [x0, y0, x1, y1],
      "page_num": <int>
    }
  ],
  "figures": [
    {
      "caption": "...",
      "bbox": [x0, y0, x1, y1],
      "page_num": <int>
    }
  ]
}

Requirements:
- Coordinates should be in the image coordinate space (top-left origin) and
  approximate the visible bounding boxes of the content.
- Maintain reading order in the `text_blocks` array.
- If a category is empty, return an empty list for it.
- Respond with **only** valid JSON, no surrounding text.
"""


class VisionExtractor(ExtractionStrategy):
    """Vision-augmented extraction using OpenRouter-hosted models."""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        max_cost_usd: float = 0.50,
        cost_per_1k_tokens_usd: float = 0.002,
    ) -> None:
        super().__init__(name="vision_augmented")
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY is not set; VisionExtractor will fail at runtime.")

        self.budget = BudgetGuard(
            max_cost_usd=max_cost_usd,
            cost_per_1k_tokens_usd=cost_per_1k_tokens_usd,
        )

    # ------------------------------------------------------------------ #
    # Core extraction
    # ------------------------------------------------------------------ #
    def extract(self, document_path: str) -> ExtractedDocument:
        """Extract structured content using a vision-language model."""
        pdf_path = Path(document_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"Document not found: {pdf_path}")

        images = convert_from_path(str(pdf_path))

        text_blocks: List[TextBlock] = []
        tables: List[Table] = []
        figures: List[Figure] = []

        for page_index, image in enumerate(images, start=1):
            if not self.budget.can_spend_more():
                logger.warning(
                    "Budget exhausted before processing all pages",
                    extra={"page_index": page_index, "path": str(pdf_path)},
                )
                break

            page_json = self._call_vision_model_for_page(image, page_index)
            tb, tbls, figs = self._parse_page_json(page_json, page_index)
            text_blocks.extend(tb)
            tables.extend(tbls)
            figures.extend(figs)

        reading_order = list(range(len(text_blocks)))

        return ExtractedDocument(
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            reading_order=reading_order,
        )

    # ------------------------------------------------------------------ #
    # API interaction
    # ------------------------------------------------------------------ #
    def _call_vision_model_for_page(self, image, page_num: int) -> Dict:
        """Call the OpenRouter vision model for a single page image with retries."""
        # Encode image as PNG bytes
        buf = BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        # OpenAI-compatible content structure with image URL (data URI style).
        # Some OpenRouter models may require a different image format; adapt as needed.
        import base64

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/png;base64,{b64_image}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = [
            {
                "role": "system",
                "content": "You are a precise document structure extractor.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT.strip()},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        last_error: Exception | None = None

        for attempt in range(self.budget.max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    # Rate limit or server error → backoff and retry.
                    self.budget.sleep_before_retry(attempt)
                    continue

                response.raise_for_status()
                data = response.json()

                # Update budget with approximate token usage if available.
                usage = data.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                if prompt_tokens or completion_tokens:
                    self.budget.record_usage(prompt_tokens, completion_tokens)

                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if not content:
                    logger.warning("Empty content from vision model", extra={"page_num": page_num})
                    return {"text_blocks": [], "tables": [], "figures": []}

                # content is expected to be a JSON string
                return json.loads(content)
            except Exception as exc:  # pragma: no cover - external failure path
                last_error = exc
                logger.exception(
                    "Vision model call failed",
                    extra={"page_num": page_num, "attempt": attempt},
                )
                self.budget.sleep_before_retry(attempt)

        # If we reach here, all retries failed.
        raise RuntimeError(f"Vision model call failed for page {page_num}") from last_error

    # ------------------------------------------------------------------ #
    # JSON parsing and normalisation
    # ------------------------------------------------------------------ #
    def _parse_page_json(
        self, payload: Dict, page_num: int
    ) -> Tuple[List[TextBlock], List[Table], List[Figure]]:
        """Convert page-level JSON into internal models."""
        text_blocks: List[TextBlock] = []
        tables: List[Table] = []
        figures: List[Figure] = []

        for tb in payload.get("text_blocks") or []:
            try:
                content = str(tb.get("content", "") or "")
                if not content.strip():
                    continue
                bbox_list = tb.get("bbox") or [0, 0, 0, 0]
                bbox = self._bbox_from_list(bbox_list)
                page = int(tb.get("page_num", page_num) or page_num)
                text_blocks.append(TextBlock(content=content, bbox=bbox, page_num=page))
            except Exception:
                logger.exception("Failed to parse text block", extra={"tb": tb})

        for raw_table in payload.get("tables") or []:
            try:
                headers = [str(h) for h in (raw_table.get("headers") or [])]
                rows_list = raw_table.get("rows") or []
                rows = [[str(cell) for cell in (row or [])] for row in rows_list]
                if not headers and not rows:
                    continue
                bbox_list = raw_table.get("bbox") or [0, 0, 0, 0]
                bbox = self._bbox_from_list(bbox_list)
                page = int(raw_table.get("page_num", page_num) or page_num)
                tables.append(Table(headers=headers, rows=rows, bbox=bbox, page_num=page))
            except Exception:
                logger.exception("Failed to parse table", extra={"table": raw_table})

        for raw_fig in payload.get("figures") or []:
            try:
                caption = str(raw_fig.get("caption", "") or "")
                bbox_list = raw_fig.get("bbox") or [0, 0, 0, 0]
                bbox = self._bbox_from_list(bbox_list)
                page = int(raw_fig.get("page_num", page_num) or page_num)
                figures.append(Figure(caption=caption, bbox=bbox, page_num=page))
            except Exception:
                logger.exception("Failed to parse figure", extra={"figure": raw_fig})

        return text_blocks, tables, figures

    @staticmethod
    def _bbox_from_list(coords: List[float]) -> BoundingBox:
        """Create a BoundingBox from a list [x0, y0, x1, y1] with safe defaults."""
        try:
            x0, y0, x1, y1 = (float(c) for c in (coords + [0, 0, 0, 0])[:4])
        except Exception:
            x0 = y0 = x1 = y1 = 0.0
        return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

    # ------------------------------------------------------------------ #
    # Confidence and cost estimation
    # ------------------------------------------------------------------ #
    def confidence_score(self, document_path: str) -> float:
        """Return a heuristic confidence score for vision extraction.

        Vision models are typically robust on scanned/complex layouts; we return
        a fixed high confidence to distinguish from heuristic-based fast text.
        """
        return 0.9

    def cost_estimate(self, document_path: str) -> Dict[str, float]:
        """Rough cost estimate based on page count and configured pricing."""
        pdf_path = Path(document_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"Document not found: {pdf_path}")

        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = len(pdf.pages)

        # Assume an upper bound of tokens per page for estimation.
        avg_tokens_per_page = 1500
        total_tokens = pages * avg_tokens_per_page
        total_cost = (total_tokens / 1000.0) * self.budget.cost_per_1k_tokens_usd

        return {
            "total_cost_usd": total_cost,
            "cost_per_page": total_cost / pages if pages > 0 else 0.0,
        }

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #
    def can_handle(self, profile: DocumentProfile) -> bool:
        """Return True when the profile suggests escalation to vision is needed.

        Triggers when:
            * Origin type is ``scanned_image``, or
            * ``estimated_cost`` has been set to ``needs_vision_model`` by triage.
        """
        if profile.origin_type == "scanned_image":
            return True

        if getattr(profile, "estimated_cost", "") == "needs_vision_model":
            return True

        return False


__all__ = ["VisionExtractor", "VISION_PROMPT"]

