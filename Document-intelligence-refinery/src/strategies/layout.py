"""Strategy B: Layout-Aware Extraction using Docling or MinerU."""

from pathlib import Path
from typing import Optional

import pdfplumber

from src.models import (
    BBox,
    DocumentProfile,
    ExtractedDocument,
    ExtractedMetadata,
    ExtractedPage,
    StrategyName,
    TableObject,
    TextBlock,
)
from src.strategies.base import ExtractionStrategy


class LayoutExtractor(ExtractionStrategy):
    """Layout-aware extraction using Docling or MinerU."""

    name = "layout_aware"

    def __init__(self):
        self.engine: Optional[str] = None
        self.docling_available = False
        self.mineru_available = False
        self.ocr_available = False
        
        # Try to import Docling
        try:
            import docling
            self.docling_available = True
        except ImportError:
            pass
        
        # Try to import MinerU (if available)
        try:
            # MinerU might be installed differently
            import mineru
            self.mineru_available = True
        except ImportError:
            pass
        
        # Try to import OCR libraries
        try:
            import pytesseract
            from pdf2image import convert_from_path
            self.ocr_available = True
        except ImportError:
            pass

    def extract(
        self, pdf_path: Path, profile: DocumentProfile, rules: dict
    ) -> tuple[ExtractedDocument, float, float]:
        """Extract with layout awareness using configured engine with OCR support."""
        # Get engine from rules
        layout_cfg = rules.get("layout_strategy", {})
        engine = layout_cfg.get("engine", "docling")
        self.engine = engine
        
        # Check if document is scanned - if so, prefer OCR
        is_scanned = profile.origin_type.value in ["scanned_image", "mixed"]
        
        # Try Docling first if configured (Docling has built-in OCR)
        if engine == "docling":
            if self.docling_available:
                try:
                    return self._extract_with_docling(pdf_path, profile, rules)
                except Exception as e:
                    import logging
                    logging.warning(f"Docling extraction failed: {e}, trying OCR fallback")
                    if is_scanned and self.ocr_available:
                        return self._extract_with_ocr(pdf_path, profile, rules)
                    return self._extract_with_pdfplumber(pdf_path, profile, rules)
            else:
                # Docling not available - use OCR if scanned, else pdfplumber
                if is_scanned and self.ocr_available:
                    import logging
                    logging.info("Docling not available, using OCR for scanned document")
                    return self._extract_with_ocr(pdf_path, profile, rules)
                else:
                    import logging
                    logging.info("Docling not available, using pdfplumber fallback")
                    return self._extract_with_pdfplumber(pdf_path, profile, rules)
        
        # Try MinerU if configured
        elif engine == "mineru":
            if self.mineru_available:
                try:
                    return self._extract_with_mineru(pdf_path, profile, rules)
                except Exception as e:
                    import logging
                    logging.warning(f"MinerU extraction failed: {e}, trying OCR fallback")
                    if is_scanned and self.ocr_available:
                        return self._extract_with_ocr(pdf_path, profile, rules)
                    return self._extract_with_pdfplumber(pdf_path, profile, rules)
            else:
                # MinerU not available - use OCR if scanned, else pdfplumber
                if is_scanned and self.ocr_available:
                    import logging
                    logging.info("MinerU not available, using OCR for scanned document")
                    return self._extract_with_ocr(pdf_path, profile, rules)
                else:
                    import logging
                    logging.info("MinerU not available, using pdfplumber fallback")
                    return self._extract_with_pdfplumber(pdf_path, profile, rules)
        else:
            # Default: use OCR if scanned, else pdfplumber
            if is_scanned and self.ocr_available:
                return self._extract_with_ocr(pdf_path, profile, rules)
            else:
                return self._extract_with_pdfplumber(pdf_path, profile, rules)

    def _extract_with_docling(
        self, pdf_path: Path, profile: DocumentProfile, rules: dict
    ) -> tuple[ExtractedDocument, float, float]:
        """Extract using Docling."""
        try:
            # Try different import paths for Docling
            try:
                from docling.document_converter import DocumentConverter
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
            except ImportError:
                # Alternative import path
                from docling import DocumentConverter
                from docling.datamodel import InputFormat, PdfPipelineOptions
            
            # Configure Docling pipeline
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True  # Enable OCR for scanned documents
            pipeline_options.do_table_structure = True  # Extract table structure
            
            converter = DocumentConverter(
                format=InputFormat.PDF,
                pipeline_options=pipeline_options
            )
            
            # Convert document
            result = converter.convert(str(pdf_path))
            
            # Get document as dict (Docling API may vary)
            try:
                doc_json = result.document.export_to_dict()
            except AttributeError:
                # Alternative: get as dict directly
                doc_json = result.document.model_dump() if hasattr(result.document, 'model_dump') else {}
            
            # Parse Docling output - handle different output formats
            pages: list[ExtractedPage] = []
            total_chars = 0
            total_tables = 0
            
            # Docling output structure may vary - try multiple formats
            content = None
            if isinstance(doc_json, dict):
                # Format 1: Direct content
                content = doc_json.get("content", [])
                # Format 2: Nested structure
                if not content and "document" in doc_json:
                    content = doc_json["document"].get("content", [])
                # Format 3: Pages array
                if not content and "pages" in doc_json:
                    content = doc_json["pages"]
            
            # If we have a Docling document object directly
            if hasattr(result, 'document'):
                doc = result.document
                # Try to get pages from document
                if hasattr(doc, 'pages'):
                    content = doc.pages
                elif hasattr(doc, 'content'):
                    content = doc.content
            
            if content:
                page_num = 1
                
                for item in content:
                    # Handle different item formats
                    item_dict = item if isinstance(item, dict) else item.model_dump() if hasattr(item, 'model_dump') else {}
                    
                    if item_dict.get("type") == "page" or "blocks" in item_dict or "content" in item_dict:
                        text_blocks: list[TextBlock] = []
                        tables: list[TableObject] = []
                        
                        # Get blocks/content
                        blocks = item_dict.get("content", item_dict.get("blocks", []))
                        
                        for block in blocks:
                            block_dict = block if isinstance(block, dict) else block.model_dump() if hasattr(block, 'model_dump') else {}
                            block_type = block_dict.get("type", "")
                            
                            if block_type in ["paragraph", "text", "heading"]:
                                text = block_dict.get("text", block_dict.get("content", ""))
                                if text:
                                    # Get bbox
                                    bbox_data = block_dict.get("bbox", block_dict.get("bounding_box", {}))
                                    if isinstance(bbox_data, (list, tuple)) and len(bbox_data) >= 4:
                                        bbox = BBox(
                                            x0=float(bbox_data[0]),
                                            y0=float(bbox_data[1]),
                                            x1=float(bbox_data[2]),
                                            y1=float(bbox_data[3]),
                                        )
                                    elif isinstance(bbox_data, dict):
                                        bbox = BBox(
                                            x0=float(bbox_data.get("x0", 0)),
                                            y0=float(bbox_data.get("y0", 0)),
                                            x1=float(bbox_data.get("x1", 100)),
                                            y1=float(bbox_data.get("y1", 100)),
                                        )
                                    else:
                                        # Default bbox
                                        bbox = BBox(x0=0, y0=0, x1=595.2, y1=841.68)
                                    
                                    text_blocks.append(
                                        TextBlock(
                                            id=f"p{page_num}-b{len(text_blocks)}",
                                            text=str(text),
                                            bbox=bbox,
                                            reading_order=len(text_blocks),
                                        )
                                    )
                                    total_chars += len(str(text))
                            
                            elif block_type == "table":
                                # Extract table
                                table_data = block_dict.get("table", block_dict.get("data", {}))
                                
                                # Handle different table formats
                                if isinstance(table_data, list):
                                    # List of rows
                                    headers = table_data[0] if table_data else []
                                    rows = table_data[1:] if len(table_data) > 1 else []
                                elif isinstance(table_data, dict):
                                    headers = table_data.get("header", table_data.get("headers", []))
                                    rows = table_data.get("rows", table_data.get("data", []))
                                else:
                                    headers = []
                                    rows = []
                                
                                # Get bbox
                                bbox_data = block_dict.get("bbox", block_dict.get("bounding_box", {}))
                                if isinstance(bbox_data, (list, tuple)) and len(bbox_data) >= 4:
                                    bbox = BBox(
                                        x0=float(bbox_data[0]),
                                        y0=float(bbox_data[1]),
                                        x1=float(bbox_data[2]),
                                        y1=float(bbox_data[3]),
                                    )
                                elif isinstance(bbox_data, dict):
                                    bbox = BBox(
                                        x0=float(bbox_data.get("x0", 0)),
                                        y0=float(bbox_data.get("y0", 0)),
                                        x1=float(bbox_data.get("x1", 100)),
                                        y1=float(bbox_data.get("y1", 100)),
                                    )
                                else:
                                    bbox = BBox(x0=0, y0=0, x1=595.2, y1=841.68)
                                
                                tables.append(
                                    TableObject(
                                        id=f"p{page_num}-t{len(tables)}",
                                        title=block_dict.get("title"),
                                        headers=[str(h) if h else "" for h in headers] if headers else [],
                                        rows=[[str(cell) if cell else "" for cell in row] for row in rows] if rows else [],
                                        bbox=bbox,
                                        reading_order=len(tables),
                                    )
                                )
                                total_tables += 1
                        
                        # Get page dimensions
                        page_width = float(item_dict.get("width", item_dict.get("page_width", 595.2)))
                        page_height = float(item_dict.get("height", item_dict.get("page_height", 841.68)))
                        
                        pages.append(
                            ExtractedPage(
                                page_number=page_num,
                                width=page_width,
                                height=page_height,
                                text_blocks=text_blocks,
                                tables=tables,
                                figures=[],
                                ldu_ids=[],
                            )
                        )
                        page_num += 1
            
            # If no pages extracted, fallback
            if not pages:
                raise ValueError("Docling did not extract any pages - falling back to pdfplumber")
            
            # Calculate confidence
            confidence = self._calculate_confidence(total_chars, len(pages), profile, total_tables)
            
            extracted = ExtractedDocument(
                doc_id=profile.doc_id,
                document_name=profile.document_name,
                pages=pages,
                metadata=ExtractedMetadata(
                    source_strategy=StrategyName.B,
                    confidence_score=confidence,
                    strategy_sequence=[StrategyName.B],
                ),
                ldus=[],
                page_index=None,
                provenance_chains=[],
            )
            
            return extracted, confidence, 0.0
            
        except Exception as e:
            # Fallback to pdfplumber if Docling fails
            import logging
            logging.warning(f"Docling extraction failed: {e}, falling back to pdfplumber")
            return self._extract_with_pdfplumber(pdf_path, profile, rules)

    def _extract_with_mineru(
        self, pdf_path: Path, profile: DocumentProfile, rules: dict
    ) -> tuple[ExtractedDocument, float, float]:
        """Extract using MinerU."""
        try:
            # MinerU typically outputs JSON files
            # Check for existing MinerU output
            mineru_cfg = rules.get("layout_strategy", {}).get("mineru", {})
            output_dir = Path(mineru_cfg.get("output_dir", ".refinery/mineru_json"))
            output_ext = mineru_cfg.get("output_extension", ".mineru.json")
            
            mineru_output = output_dir / f"{pdf_path.stem}{output_ext}"
            
            if mineru_output.exists():
                # Parse MinerU JSON output
                import json
                with open(mineru_output) as f:
                    mineru_data = json.load(f)
                
                # Convert MinerU format to our format
                pages: list[ExtractedPage] = []
                total_chars = 0
                total_tables = 0
                
                # MinerU structure varies, adapt as needed
                if "pages" in mineru_data:
                    for page_idx, page_data in enumerate(mineru_data["pages"], start=1):
                        text_blocks: list[TextBlock] = []
                        tables: list[TableObject] = []
                        
                        # Extract text blocks
                        for block in page_data.get("blocks", []):
                            if block.get("type") == "text":
                                text = block.get("text", "")
                                if text:
                                    bbox_data = block.get("bbox", {})
                                    bbox = BBox(
                                        x0=float(bbox_data.get("x0", 0)),
                                        y0=float(bbox_data.get("y0", 0)),
                                        x1=float(bbox_data.get("x1", 100)),
                                        y1=float(bbox_data.get("y1", 100)),
                                    )
                                    text_blocks.append(
                                        TextBlock(
                                            id=f"p{page_idx}-b{len(text_blocks)}",
                                            text=text,
                                            bbox=bbox,
                                            reading_order=len(text_blocks),
                                        )
                                    )
                                    total_chars += len(text)
                            
                            elif block.get("type") == "table":
                                table_data = block.get("table", {})
                                headers = table_data.get("headers", [])
                                rows = table_data.get("rows", [])
                                
                                bbox_data = block.get("bbox", {})
                                bbox = BBox(
                                    x0=float(bbox_data.get("x0", 0)),
                                    y0=float(bbox_data.get("y0", 0)),
                                    x1=float(bbox_data.get("x1", 100)),
                                    y1=float(bbox_data.get("y1", 100)),
                                )
                                
                                tables.append(
                                    TableObject(
                                        id=f"p{page_idx}-t{len(tables)}",
                                        title=block.get("title"),
                                        headers=[str(h) for h in headers] if headers else [],
                                        rows=[[str(cell) for cell in row] for row in rows] if rows else [],
                                        bbox=bbox,
                                        reading_order=len(tables),
                                    )
                                )
                                total_tables += 1
                        
                        pages.append(
                            ExtractedPage(
                                page_number=page_idx,
                                width=float(page_data.get("width", 595.2)),
                                height=float(page_data.get("height", 841.68)),
                                text_blocks=text_blocks,
                                tables=tables,
                                figures=[],
                                ldu_ids=[],
                            )
                        )
                
                confidence = self._calculate_confidence(total_chars, len(pages), profile, total_tables)
                
                extracted = ExtractedDocument(
                    doc_id=profile.doc_id,
                    document_name=profile.document_name,
                    pages=pages,
                    metadata=ExtractedMetadata(
                        source_strategy=StrategyName.B,
                        confidence_score=confidence,
                        strategy_sequence=[StrategyName.B],
                    ),
                    ldus=[],
                    page_index=None,
                    provenance_chains=[],
                )
                
                return extracted, confidence, 0.0
            else:
                # MinerU output not found, fallback
                import logging
                logging.warning(f"MinerU output not found: {mineru_output}, falling back to pdfplumber")
                return self._extract_with_pdfplumber(pdf_path, profile, rules)
                
        except Exception as e:
            # Fallback to pdfplumber if MinerU fails
            import logging
            logging.warning(f"MinerU extraction failed: {e}, falling back to pdfplumber")
            return self._extract_with_pdfplumber(pdf_path, profile, rules)

    def _extract_with_ocr(
        self, pdf_path: Path, profile: DocumentProfile, rules: dict
    ) -> tuple[ExtractedDocument, float, float]:
        """Extract text using OCR (Tesseract) for scanned documents."""
        if not self.ocr_available:
            raise ImportError("OCR libraries (pytesseract, pdf2image) not available")
        
        import pytesseract
        from pdf2image import convert_from_path
        from PIL import Image
        
        pages: list[ExtractedPage] = []
        total_chars = 0
        total_tables = 0
        
        # Convert PDF pages to images
        try:
            pdf_images = convert_from_path(str(pdf_path), dpi=300)  # Higher DPI for better OCR
        except Exception as e:
            import logging
            logging.error(f"Failed to convert PDF to images: {e}")
            # Fallback to pdfplumber
            return self._extract_with_pdfplumber(pdf_path, profile, rules)
        
        # Extract text from each page using OCR
        for page_num, image in enumerate(pdf_images, start=1):
            text_blocks: list[TextBlock] = []
            tables: list[TableObject] = []
            
            # Get page dimensions
            page_width = float(image.width)
            page_height = float(image.height)
            
            # Use pytesseract to extract text with layout information
            try:
                # Get detailed OCR data with bounding boxes
                ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                
                # Group words into lines and blocks
                words_by_line: dict[float, list] = {}  # y-coordinate -> list of word dicts
                
                for i in range(len(ocr_data['text'])):
                    word_text = ocr_data['text'][i].strip()
                    if not word_text or ocr_data['conf'][i] == '-1':  # Skip empty or invalid
                        continue
                    
                    # Get word bounding box
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    conf = int(ocr_data['conf'][i])
                    
                    # Only include words with reasonable confidence (>30)
                    if conf < 30:
                        continue
                    
                    # Use center y-coordinate for line grouping
                    word_y = y + h / 2
                    
                    # Group by y-coordinate (within 10 pixels)
                    line_key = None
                    for key_y in words_by_line.keys():
                        if abs(key_y - word_y) < 10:
                            line_key = key_y
                            break
                    
                    if line_key is None:
                        line_key = word_y
                        words_by_line[line_key] = []
                    
                    words_by_line[line_key].append({
                        'text': word_text,
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'x1': x + w,
                        'y1': y + h,
                    })
                
                # Convert lines to text blocks
                sorted_lines = sorted(words_by_line.items(), key=lambda x: x[0])
                current_block_lines = []
                current_block_bbox = None
                
                for line_y, words in sorted_lines:
                    line_text = " ".join([w['text'] for w in words])
                    if not line_text.strip():
                        continue
                    
                    # Calculate line bbox
                    line_x0 = min(w['x'] for w in words)
                    line_y0 = min(w['y'] for w in words)
                    line_x1 = max(w['x1'] for w in words)
                    line_y1 = max(w['y1'] for w in words)
                    
                    # Group lines into blocks (if close together)
                    if current_block_bbox is None:
                        # Start new block
                        current_block_lines = [line_text]
                        current_block_bbox = (line_x0, line_y0, line_x1, line_y1)
                    elif abs(line_y0 - current_block_bbox[3]) < 30:  # Within 30 pixels
                        # Add to current block
                        current_block_lines.append(line_text)
                        current_block_bbox = (
                            min(current_block_bbox[0], line_x0),
                            min(current_block_bbox[1], line_y0),
                            max(current_block_bbox[2], line_x1),
                            max(current_block_bbox[3], line_y1),
                        )
                    else:
                        # Save current block and start new one
                        if current_block_lines:
                            block_text = " ".join(current_block_lines)
                            if block_text.strip():
                                from src.strategies.fast_text import normalize_bbox
                                x0, y0, x1, y1 = normalize_bbox(
                                    current_block_bbox[0],
                                    current_block_bbox[1],
                                    current_block_bbox[2],
                                    current_block_bbox[3],
                                )
                                text_blocks.append(
                                    TextBlock(
                                        id=f"p{page_num}-b{len(text_blocks)}",
                                        text=block_text,
                                        bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                                        reading_order=len(text_blocks),
                                    )
                                )
                                total_chars += len(block_text)
                        
                        current_block_lines = [line_text]
                        current_block_bbox = (line_x0, line_y0, line_x1, line_y1)
                
                # Add last block
                if current_block_lines:
                    block_text = " ".join(current_block_lines)
                    if block_text.strip():
                        from src.strategies.fast_text import normalize_bbox
                        x0, y0, x1, y1 = normalize_bbox(
                            current_block_bbox[0],
                            current_block_bbox[1],
                            current_block_bbox[2],
                            current_block_bbox[3],
                        )
                        text_blocks.append(
                            TextBlock(
                                id=f"p{page_num}-b{len(text_blocks)}",
                                text=block_text,
                                bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                                reading_order=len(text_blocks),
                            )
                        )
                        total_chars += len(block_text)
                
                # Try to extract tables using OCR table detection
                # Note: This is a simplified approach - for better table extraction,
                # consider using specialized table detection libraries
                try:
                    # Use pytesseract's table detection if available
                    ocr_tables = pytesseract.image_to_string(image, config='--psm 6')
                    # Simple heuristic: if we find tab-separated content, it might be a table
                    # This is a basic implementation - can be improved
                except Exception:
                    pass  # Table extraction is optional
                
            except Exception as e:
                import logging
                logging.warning(f"OCR extraction failed for page {page_num}: {e}")
                # Continue with next page
            
            pages.append(
                ExtractedPage(
                    page_number=page_num,
                    width=page_width,
                    height=page_height,
                    text_blocks=text_blocks,
                    tables=tables,
                    figures=[],
                    ldu_ids=[],
                )
            )
        
        # Calculate confidence (OCR typically has lower confidence than native text)
        confidence = self._calculate_confidence(total_chars, len(pages), profile, total_tables)
        # Adjust confidence for OCR (typically 0.7-0.85 range)
        confidence = min(0.85, max(0.70, confidence))
        
        extracted = ExtractedDocument(
            doc_id=profile.doc_id,
            document_name=profile.document_name,
            pages=pages,
            metadata=ExtractedMetadata(
                source_strategy=StrategyName.B,
                confidence_score=confidence,
                strategy_sequence=[StrategyName.B],
            ),
            ldus=[],
            page_index=None,
            provenance_chains=[],
        )
        
        return extracted, confidence, 0.0  # OCR is free (local processing)

    def _extract_with_pdfplumber(
        self, pdf_path: Path, profile: DocumentProfile, rules: dict
    ) -> tuple[ExtractedDocument, float, float]:
        """Fallback extraction using enhanced pdfplumber."""
        pages: list[ExtractedPage] = []
        total_chars = 0
        total_tables = 0

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text_blocks: list[TextBlock] = []
                tables: list[TableObject] = []

                # Extract tables first
                pdf_tables = page.find_tables()
                for table_idx, table in enumerate(pdf_tables):
                    try:
                        table_data = table.extract()
                        if table_data and len(table_data) > 1:
                            headers = table_data[0] if table_data else []
                            rows = table_data[1:] if len(table_data) > 1 else []
                            
                            # Get table bbox
                            bbox = table.bbox if hasattr(table, "bbox") else (0, 0, page.width, page.height)
                            
                            # Normalize bbox
                            from src.strategies.fast_text import normalize_bbox
                            x0, y0, x1, y1 = normalize_bbox(
                                float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                            )
                            
                            tables.append(
                                TableObject(
                                    id=f"p{page_num}-t{table_idx}",
                                    title=None,
                                    headers=[str(h) if h else "" for h in headers],
                                    rows=[[str(cell) if cell else "" for cell in row] for row in rows],
                                    bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                                    reading_order=len(tables),
                                )
                            )
                            total_tables += 1
                    except Exception:
                        pass  # Skip malformed tables

                # Extract text blocks - use extract_text() first for better reliability
                full_text = page.extract_text()
                if full_text and full_text.strip():
                    # Split into lines and group into blocks
                    lines = [line.strip() for line in full_text.split("\n") if line.strip()]
                    
                    # Group consecutive lines into blocks
                    current_block = []
                    for line in lines:
                        if line:
                            current_block.append(line)
                            # Create block every few lines or on paragraph breaks
                            if len(current_block) >= 3:
                                block_text = " ".join(current_block)
                                if block_text.strip():
                                    block_bbox = BBox(x0=0, y0=0, x1=page.width, y1=page.height)
                                    text_blocks.append(
                                        TextBlock(
                                            id=f"p{page_num}-b{len(text_blocks)}",
                                            text=block_text,
                                            bbox=block_bbox,
                                            reading_order=len(text_blocks),
                                        )
                                    )
                                    total_chars += len(block_text)
                                current_block = []
                    
                    # Add remaining block
                    if current_block:
                        block_text = " ".join(current_block)
                        if block_text.strip():
                            block_bbox = BBox(x0=0, y0=0, x1=page.width, y1=page.height)
                            text_blocks.append(
                                TextBlock(
                                    id=f"p{page_num}-b{len(text_blocks)}",
                                    text=block_text,
                                    bbox=block_bbox,
                                    reading_order=len(text_blocks),
                                )
                            )
                            total_chars += len(block_text)
                
                # Fallback: Use extract_words() if extract_text() didn't work
                if not text_blocks:
                    words = page.extract_words() or []
                    if words:
                        # Sort by reading order (top to bottom, left to right)
                        sorted_words = sorted(words, key=lambda w: (w.get("top", 0), w.get("x0", 0)))
                        
                        current_line = []
                        current_y = None
                        
                        for word in sorted_words:
                            word_y = word.get("top", 0)
                            word_text = word.get("text", "")
                            
                            if not word_text.strip():
                                continue
                            
                            if current_y is None or abs(word_y - current_y) < 5:  # Same line
                                current_line.append((word, word_text))
                                current_y = word_y
                            else:
                                # New line - create block from previous line
                                if current_line:
                                    block_text = " ".join([w[1] for w in current_line])
                                    word_objs = [w[0] for w in current_line]
                                    
                                    from src.strategies.fast_text import normalize_bbox
                                    x0, y0, x1, y1 = normalize_bbox(
                                        float(min(w.get("x0", 0) for w in word_objs)),
                                        float(min(w.get("top", 0) for w in word_objs)),
                                        float(max(w.get("x1", 0) for w in word_objs)),
                                        float(max(w.get("bottom", 0) for w in word_objs)),
                                    )
                                    
                                    text_blocks.append(
                                        TextBlock(
                                            id=f"p{page_num}-b{len(text_blocks)}",
                                            text=block_text,
                                            bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                                            reading_order=len(text_blocks),
                                        )
                                    )
                                    total_chars += len(block_text)
                                
                                current_line = [(word, word_text)]
                                current_y = word_y
                        
                        # Add last line
                        if current_line:
                            block_text = " ".join([w[1] for w in current_line])
                            word_objs = [w[0] for w in current_line]
                            
                            from src.strategies.fast_text import normalize_bbox
                            x0, y0, x1, y1 = normalize_bbox(
                                float(min(w.get("x0", 0) for w in word_objs)),
                                float(min(w.get("top", 0) for w in word_objs)),
                                float(max(w.get("x1", 0) for w in word_objs)),
                                float(max(w.get("bottom", 0) for w in word_objs)),
                            )
                            
                            text_blocks.append(
                                TextBlock(
                                    id=f"p{page_num}-b{len(text_blocks)}",
                                    text=block_text,
                                    bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                                    reading_order=len(text_blocks),
                                )
                            )
                            total_chars += len(block_text)

                pages.append(
                    ExtractedPage(
                        page_number=page_num,
                        width=float(page.width),
                        height=float(page.height),
                        text_blocks=text_blocks,
                        tables=tables,
                        figures=[],
                        ldu_ids=[],
                    )
                )

        # Calculate confidence
        confidence = self._calculate_confidence(total_chars, len(pages), profile, total_tables)

        extracted = ExtractedDocument(
            doc_id=profile.doc_id,
            document_name=profile.document_name,
            pages=pages,
            metadata=ExtractedMetadata(
                source_strategy=StrategyName.B,
                confidence_score=confidence,
                strategy_sequence=[StrategyName.B],
            ),
            ldus=[],
            page_index=None,
            provenance_chains=[],
        )

        return extracted, confidence, 0.0  # Free (local processing)

    def _calculate_confidence(
        self, total_chars: int, num_pages: int, profile: DocumentProfile, num_tables: int
    ) -> float:
        """Calculate confidence score."""
        chars_per_page = total_chars / num_pages if num_pages > 0 else 0
        
        base_confidence = 0.6
        if chars_per_page > 500:
            base_confidence += 0.2
        elif chars_per_page > 200:
            base_confidence += 0.1
        
        # Bonus for table extraction
        if num_tables > 0:
            base_confidence += 0.1
        
        # Higher confidence if using Docling/MinerU
        if self.engine in ["docling", "mineru"] and (self.docling_available or self.mineru_available):
            base_confidence += 0.1
        
        return min(0.95, base_confidence)
