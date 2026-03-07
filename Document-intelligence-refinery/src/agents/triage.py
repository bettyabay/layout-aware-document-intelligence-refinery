"""Triage Agent - Stage 1: Document classification and profiling."""

import hashlib
from pathlib import Path

import pdfplumber

from src.agents.domain_classifier import DomainClassifier, create_domain_classifier
from src.models import (
    DocumentProfile,
    EstimatedExtractionCost,
    LanguageInfo,
    LayoutComplexity,
    OriginType,
    StrategyName,
    TriageSignals,
)
from src.utils.language import detect_language
from src.utils.ledger import write_json
from src.utils.rules import load_rules


class TriageAgent:
    """Triage agent for document classification."""

    def __init__(self, rules: dict, domain_classifier: DomainClassifier | None = None):
        self.rules = rules
        if domain_classifier is not None:
            self.domain_classifier = domain_classifier
        else:
            classifier_name = (
                (rules or {}).get("triage", {}).get("domain_classifier", "keyword")
            )
            self.domain_classifier = create_domain_classifier(str(classifier_name))

    def _triage_cfg(self) -> dict:
        """Get triage configuration."""
        return self.rules.get("triage", {})

    @staticmethod
    def _safe_image_area(img: dict) -> float:
        """Calculate safe image area."""
        x0 = img.get("x0", 0) or 0
        x1 = img.get("x1", 0) or 0
        top = img.get("top", 0) or 0
        bottom = img.get("bottom", 0) or 0
        return max(0.0, (x1 - x0)) * max(0.0, (bottom - top))

    def classify_origin_type(
        self,
        avg_char_count: float,
        avg_image_ratio: float,
        scanned_pages_ratio: float,
        form_fillable_ratio: float,
    ) -> OriginType:
        """Classify document origin type."""
        triage = self._triage_cfg()
        scanned_ratio_threshold = float(triage.get("scanned_pages_ratio_threshold", 0.85))
        scanned_max_chars = float(triage.get("scanned_max_char_count", 30))
        scanned_min_image = float(triage.get("scanned_min_image_ratio", 0.5))
        native_min_chars = float(triage.get("native_min_char_count", 100))
        form_fillable_ratio_threshold = float(triage.get("form_fillable_ratio_threshold", 0.20))

        if form_fillable_ratio >= form_fillable_ratio_threshold:
            return OriginType.FORM_FILLABLE

        if scanned_pages_ratio >= scanned_ratio_threshold or (
            avg_char_count <= scanned_max_chars and avg_image_ratio >= scanned_min_image
        ):
            return OriginType.SCANNED_IMAGE
        if avg_char_count >= native_min_chars and avg_image_ratio < scanned_min_image:
            return OriginType.NATIVE_DIGITAL
        return OriginType.MIXED

    def classify_layout_complexity(
        self, table_density: float, figure_density: float, column_variation: float
    ) -> LayoutComplexity:
        """Classify layout complexity."""
        triage = self._triage_cfg()
        table_heavy_density_threshold = float(triage.get("table_heavy_density_threshold", 0.15))
        figure_heavy_density_threshold = float(triage.get("figure_heavy_density_threshold", 0.15))
        multi_column_variation_threshold = float(triage.get("multi_column_variation_threshold", 0.35))
        single_column_max_table_density = float(triage.get("single_column_max_table_density", 0.08))
        single_column_max_variation = float(triage.get("single_column_max_variation", 0.20))

        if table_density >= table_heavy_density_threshold:
            return LayoutComplexity.TABLE_HEAVY
        if figure_density >= figure_heavy_density_threshold:
            return LayoutComplexity.FIGURE_HEAVY
        if column_variation >= multi_column_variation_threshold:
            return LayoutComplexity.MULTI_COLUMN
        if (
            table_density < single_column_max_table_density
            and column_variation < single_column_max_variation
        ):
            return LayoutComplexity.SINGLE_COLUMN
        return LayoutComplexity.MIXED

    def select_strategy(
        self, origin_type: OriginType, layout_complexity: LayoutComplexity
    ) -> StrategyName:
        """Select extraction strategy based on classification."""
        if origin_type == OriginType.SCANNED_IMAGE:
            return StrategyName.C
        if layout_complexity in [
            LayoutComplexity.MULTI_COLUMN,
            LayoutComplexity.TABLE_HEAVY,
            LayoutComplexity.FIGURE_HEAVY,
        ]:
            return StrategyName.B
        if origin_type == OriginType.NATIVE_DIGITAL and layout_complexity == LayoutComplexity.SINGLE_COLUMN:
            return StrategyName.A
        return StrategyName.B  # Default to layout-aware

    def estimate_cost(self, strategy: StrategyName) -> EstimatedExtractionCost:
        """Estimate extraction cost."""
        if strategy == StrategyName.A:
            return EstimatedExtractionCost.FREE
        if strategy == StrategyName.B:
            return EstimatedExtractionCost.NEEDS_LAYOUT_MODEL
        return EstimatedExtractionCost.NEEDS_VISION_MODEL

    def profile_document(
        self, pdf_path: Path | str, persist: bool = True
    ) -> DocumentProfile:
        """Profile a document and create DocumentProfile."""
        pdf_path = Path(pdf_path)
        
        # Analyze PDF
        total_chars = 0
        total_pages = 0
        pages_with_fonts = 0
        total_image_area = 0.0
        total_page_area = 0.0
        table_count = 0
        figure_count = 0
        column_variations = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                total_pages += 1
                page_area = page.width * page.height
                total_page_area += page_area

                # Extract text
                text = page.extract_text() or ""
                chars = len(text)
                total_chars += chars

                # Check for fonts
                if page.chars:
                    pages_with_fonts += 1

                # Image area
                images = page.images or []
                page_image_area = sum(self._safe_image_area(img) for img in images)
                total_image_area += page_image_area

                # Tables
                tables = page.find_tables()
                table_count += len(tables)

                # Figures (simplified - count images as figures)
                figure_count += len(images)

                # Column detection (simplified)
                if chars > 0:
                    # Rough column detection based on text distribution
                    words = page.extract_words()
                    if words:
                        x_coords = [w.get("x0", 0) for w in words]
                        if x_coords:
                            x_span = max(x_coords) - min(x_coords)
                            column_variation = x_span / page.width if page.width > 0 else 0
                            column_variations.append(column_variation)

        # Calculate averages
        avg_chars_per_page = total_chars / total_pages if total_pages > 0 else 0
        avg_char_density = total_chars / total_page_area if total_page_area > 0 else 0
        font_ratio = pages_with_fonts / total_pages if total_pages > 0 else 0
        avg_image_ratio = total_image_area / total_page_area if total_page_area > 0 else 0
        scanned_pages_ratio = 1.0 - font_ratio
        table_density = table_count / total_pages if total_pages > 0 else 0
        figure_density = figure_count / total_pages if total_pages > 0 else 0
        avg_column_variation = (
            sum(column_variations) / len(column_variations) if column_variations else 0
        )

        # Classify
        origin_type = self.classify_origin_type(
            avg_chars_per_page, avg_image_ratio, scanned_pages_ratio, 0.0
        )
        layout_complexity = self.classify_layout_complexity(
            table_density, figure_density, avg_column_variation
        )
        strategy = self.select_strategy(origin_type, layout_complexity)
        cost = self.estimate_cost(strategy)

        # Domain classification
        sample_text = ""
        with pdfplumber.open(str(pdf_path)) as pdf:
            # Sample first few pages for domain classification
            for i, page in enumerate(pdf.pages[:3]):
                text = page.extract_text() or ""
                sample_text += text + " "
        domain_hint = self.domain_classifier.classify(sample_text)

        # Language detection
        language = detect_language(sample_text)

        # Triage signals
        signals = TriageSignals(
            avg_char_density=avg_char_density,
            avg_whitespace_ratio=1.0 - (total_chars / (total_page_area * 0.1)) if total_page_area > 0 else 0.5,
            avg_image_area_ratio=avg_image_ratio,
            table_density=table_density,
            figure_density=figure_density,
        )

        # Confidence score (simplified)
        confidence = 0.8 if font_ratio > 0.5 else 0.5

        # Generate doc_id
        doc_id = hashlib.md5(str(pdf_path).encode()).hexdigest()[:16]

        profile = DocumentProfile(
            doc_id=doc_id,
            document_name=pdf_path.name,
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            language=language,
            domain_hint=domain_hint,
            estimated_extraction_cost=cost,
            triage_signals=signals,
            selected_strategy=strategy,
            triage_confidence_score=confidence,
        )

        # Persist if requested
        if persist:
            profiles_dir = Path(".refinery/profiles")
            profiles_dir.mkdir(parents=True, exist_ok=True)
            profile_path = profiles_dir / f"{doc_id}.json"
            write_json(profile.model_dump(), profile_path)

        return profile
