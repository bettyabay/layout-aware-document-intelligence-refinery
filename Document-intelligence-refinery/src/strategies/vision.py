"""Strategy C: Vision-Augmented Extraction using free OpenRouter models."""

import base64
import os
from pathlib import Path

import httpx
from pdf2image import convert_from_path

from src.models import (
    BBox,
    DocumentProfile,
    ExtractedDocument,
    ExtractedMetadata,
    ExtractedPage,
    StrategyName,
    TextBlock,
)
from src.strategies.base import ExtractionStrategy
from src.utils.rules import load_rules


class VisionExtractor(ExtractionStrategy):
    """Vision-augmented extraction using free OpenRouter VLM models."""

    name = "vision_augmented"

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = "https://openrouter.ai/api/v1"
        # Free models in priority order
        self.free_models = [
            "qwen/qwen2-vl-7b-instruct:free",
            "nvidia/nemotron-4-12b-vl:free",
            "meta-llama/llama-3.2-11b-vision:free",
        ]

    def extract(
        self, pdf_path: Path, profile: DocumentProfile, rules: dict
    ) -> tuple[ExtractedDocument, float, float]:
        """Extract using vision models."""
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Vision extraction requires API key.")

        # Load model config
        model_config_path = Path("config/model_config.yaml")
        if model_config_path.exists():
            model_config = load_rules(model_config_path)
            free_models = model_config.get("free_vlm_models", [])
            if free_models:
                self.free_models = [m.get("model_id") for m in free_models if m.get("model_id")]

        pages: list[ExtractedPage] = []
        total_cost = 0.0
        max_cost = float(rules.get("escalation", {}).get("max_vision_cost_per_doc", 0.05))

        # Convert PDF to images
        try:
            images = convert_from_path(str(pdf_path), dpi=200)
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {e}")

        for page_num, image in enumerate(images, start=1):
            if total_cost >= max_cost:
                break  # Budget limit reached

            # Convert image to base64
            import io
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode()

            # Try each free model in order
            extracted_text = None
            for model_id in self.free_models:
                try:
                    extracted_text, cost = self._call_vision_api(model_id, image_b64, page_num)
                    total_cost += cost
                    if extracted_text:
                        break  # Success
                except Exception:
                    continue  # Try next model

            if not extracted_text:
                extracted_text = f"[Page {page_num} - Vision extraction failed]"

            # Create text block
            text_block = TextBlock(
                id=f"p{page_num}-v1",
                text=extracted_text,
                bbox=BBox(
                    x0=0.0,
                    y0=0.0,
                    x1=float(image.width),
                    y1=float(image.height),
                ),
                reading_order=0,
            )

            pages.append(
                ExtractedPage(
                    page_number=page_num,
                    width=float(image.width),
                    height=float(image.height),
                    text_blocks=[text_block],
                    tables=[],
                    figures=[],
                    ldu_ids=[],
                )
            )

        # Calculate confidence (lower for vision extraction)
        confidence = 0.6 if pages else 0.3

        extracted = ExtractedDocument(
            doc_id=profile.doc_id,
            document_name=profile.document_name,
            pages=pages,
            metadata=ExtractedMetadata(
                source_strategy=StrategyName.C,
                confidence_score=confidence,
                strategy_sequence=[StrategyName.C],
            ),
            ldus=[],
            page_index=None,
            provenance_chains=[],
        )

        return extracted, confidence, total_cost

    def _call_vision_api(self, model_id: str, image_b64: str, page_num: int) -> tuple[str, float]:
        """Call OpenRouter vision API."""
        prompt = f"""Extract all text from this document page {page_num}. 
        Preserve the structure and layout. Include tables, lists, and all content.
        Return only the extracted text, no explanations."""

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/document-intelligence-refinery",
                "X-Title": "Document Intelligence Refinery",
            },
            json={
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 4096,
            },
            timeout=60.0,
        )

        response.raise_for_status()
        data = response.json()
        
        # Extract text
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Estimate cost (free models should be $0, but track usage)
        cost = 0.0
        
        return text, cost
