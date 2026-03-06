"""PageIndex Builder - Stage 4 of the Document Intelligence Refinery.

This module implements the PageIndex Builder, which creates a hierarchical
navigation structure over documents. Inspired by VectifyAI's PageIndex, it
allows agents to locate relevant sections before performing vector search.

The PageIndex is a tree where each node is a Section with:
- title, page_start, page_end
- child_sections (nested hierarchy)
- key_entities (extracted named entities)
- summary (LLM-generated, 2-3 sentences)
- data_types_present (tables, figures, equations, etc.)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.models.extracted_document import ExtractedDocument
from src.models.ldu import LDU
from src.models.page_index import DataType, PageIndex, Section
from src.utils.section_chunker import SectionChunker, SectionNode

logger = logging.getLogger(__name__)


class PageIndexBuilder:
    """Builder for hierarchical PageIndex navigation structures.

    Takes LDUs or ExtractedDocument as input and builds a PageIndex tree
    with sections, summaries, entities, and data type classification.
    """

    def __init__(
        self,
        config_path: str = "rubric/extraction_rules.yaml",
        use_llm_summaries: bool = True,
        llm_model: str = "gpt-4o-mini",  # Budget-friendly default
    ):
        """Initialize the PageIndex Builder.

        Args:
            config_path: Path to extraction rules configuration.
            use_llm_summaries: Whether to generate LLM summaries for sections.
            llm_model: LLM model to use for summaries (via OpenRouter).
        """
        self.config_path = config_path
        self.use_llm_summaries = use_llm_summaries
        self.llm_model = llm_model
        self.section_chunker = SectionChunker()

    def build_from_ldus(
        self,
        doc_id: str,
        doc_name: str,
        chunks: List[LDU],
    ) -> PageIndex:
        """Build PageIndex from a list of LDUs.

        Args:
            doc_id: Stable identifier for the document.
            doc_name: Human-readable document name.
            chunks: List of Logical Document Units.

        Returns:
            PageIndex tree structure.
        """
        logger.info("Building PageIndex from %d LDUs for document %s", len(chunks), doc_id)

        # Build section hierarchy from header chunks
        section_tree = self.section_chunker.build_section_hierarchy(chunks)

        # If no sections detected, try to infer sections from content patterns
        if section_tree is None or (len(section_tree.children) == 0 and section_tree.title == "Document Root"):
            logger.info("No explicit headers found, inferring sections from content patterns")
            section_tree = self._infer_sections_from_content(chunks)

        # Convert SectionNode tree to PageIndex Section tree
        root_sections = self._convert_section_tree(section_tree, chunks)

        # Enrich sections with entities, summaries, and data types
        for section in self._flatten_sections(root_sections):
            self._enrich_section(section, chunks)

        return PageIndex(
            doc_id=doc_id,
            doc_name=doc_name,
            root_sections=root_sections,
        )

    def build_from_extracted_document(
        self,
        doc_id: str,
        doc_name: str,
        extracted_doc: ExtractedDocument,
    ) -> PageIndex:
        """Build PageIndex from an ExtractedDocument.

        This is a convenience method that first chunks the document, then
        builds the PageIndex. For better control, use build_from_ldus with
        pre-chunked LDUs.

        Args:
            doc_id: Stable identifier for the document.
            doc_name: Human-readable document name.
            extracted_doc: ExtractedDocument to index.

        Returns:
            PageIndex tree structure.
        """
        logger.info("Building PageIndex from ExtractedDocument for %s", doc_id)

        # First, we need to chunk the document
        # Import here to avoid circular dependencies
        from src.agents.chunker import ChunkingEngine

        chunking_engine = ChunkingEngine()
        chunks = chunking_engine.chunk(extracted_doc)

        return self.build_from_ldus(doc_id, doc_name, chunks)

    def _convert_section_tree(
        self,
        section_node: SectionNode,
        chunks: List[LDU],
    ) -> List[Section]:
        """Convert SectionNode tree to PageIndex Section tree.

        Args:
            section_node: Root SectionNode (or None for flat structure).
            chunks: List of all LDUs for page range calculation.

        Returns:
            List of root Section objects.
        """
        if section_node is None:
            # No sections detected - create a single "Document" section
            if chunks:
                page_start = min((min(c.page_refs) for c in chunks if c.page_refs), default=1)
                page_end = max((max(c.page_refs) for c in chunks if c.page_refs), default=1)
            else:
                page_start = page_end = 1

            return [
                Section(
                    title="Document",
                    page_start=page_start,
                    page_end=page_end,
                    child_sections=[],
                )
            ]

        # If root has no children and is just "Document Root", return a single Document section
        if section_node.title == "Document Root" and len(section_node.children) == 0:
            if chunks:
                page_start = min((min(c.page_refs) for c in chunks if c.page_refs), default=1)
                page_end = max((max(c.page_refs) for c in chunks if c.page_refs), default=1)
            else:
                page_start = page_end = 1
            return [
                Section(
                    title="Document",
                    page_start=page_start,
                    page_end=page_end,
                    child_sections=[],
                )
            ]

        # Convert children recursively
        child_sections = [
            self._convert_section_node(child, chunks) for child in section_node.children
        ]

        # If root is "Document Root", return only children (don't include root as a section)
        if section_node.title == "Document Root":
            return child_sections if child_sections else [
                Section(
                    title="Document",
                    page_start=min((min(c.page_refs) for c in chunks if c.page_refs), default=1),
                    page_end=max((max(c.page_refs) for c in chunks if c.page_refs), default=1),
                    child_sections=[],
                )
            ]

        # Create root section with children
        root_section = self._convert_section_node(section_node, chunks)
        root_section.child_sections = child_sections

        return [root_section]

    def _convert_section_node(
        self,
        section_node: SectionNode,
        chunks: List[LDU],
    ) -> Section:
        """Convert a single SectionNode to a Section.

        Args:
            section_node: SectionNode to convert.
            chunks: List of all LDUs for page range calculation.

        Returns:
            Section object.
        """
        # Calculate page range from chunks in this section
        section_chunks = [
            c for c in chunks if c.content_hash in section_node.chunk_ids
        ]

        if section_chunks:
            page_start = min((min(c.page_refs) for c in section_chunks if c.page_refs), default=section_node.page_start)
            page_end = max((max(c.page_refs) for c in section_chunks if c.page_refs), default=section_node.page_end)
        else:
            page_start = section_node.page_start
            page_end = section_node.page_end

        # Convert children
        child_sections = [
            self._convert_section_node(child, chunks) for child in section_node.children
        ]

        return Section(
            title=section_node.title,
            page_start=page_start,
            page_end=page_end,
            child_sections=child_sections,
        )

    def _enrich_section(self, section: Section, chunks: List[LDU]) -> None:
        """Enrich a section with entities, summary, and data types.

        Args:
            section: Section to enrich (modified in place).
            chunks: List of all LDUs.
        """
        # Find chunks belonging to this section
        section_chunks = self._find_chunks_in_section(section, chunks)

        # Extract data types present
        section.data_types_present = self._extract_data_types(section_chunks)

        # Extract named entities
        section.key_entities = self._extract_entities(section_chunks)

        # Generate LLM summary
        if self.use_llm_summaries:
            section.summary = self._generate_summary(section, section_chunks)
        else:
            section.summary = self._generate_heuristic_summary(section_chunks)

        # Recursively enrich child sections
        for child in section.child_sections:
            self._enrich_section(child, chunks)

    def _find_chunks_in_section(self, section: Section, chunks: List[LDU]) -> List[LDU]:
        """Find all chunks that belong to a section based on page range.

        Args:
            section: Section to find chunks for.
            chunks: List of all LDUs.

        Returns:
            List of LDUs in this section.
        """
        section_chunks = []
        for chunk in chunks:
            # Check if chunk overlaps with section page range
            chunk_pages = set(chunk.page_refs)
            section_pages = set(range(section.page_start, section.page_end + 1))
            if chunk_pages & section_pages:  # Overlap
                section_chunks.append(chunk)

        return section_chunks

    def _extract_data_types(self, chunks: List[LDU]) -> List[DataType]:
        """Extract data types present in chunks.

        Args:
            chunks: List of LDUs.

        Returns:
            List of data types found.
        """
        data_types: Set[DataType] = set()

        for chunk in chunks:
            if chunk.chunk_type == "table":
                data_types.add("table")
            elif chunk.chunk_type == "figure":
                data_types.add("figure")
            elif chunk.chunk_type == "list":
                data_types.add("list")
            elif chunk.chunk_type == "header":
                # Headers are metadata, not data types
                pass
            else:
                # Paragraph, footnote, etc. count as text
                data_types.add("text")

        # Check for equations (simple heuristic: LaTeX-like patterns)
        for chunk in chunks:
            if "\\" in chunk.content or "$" in chunk.content:
                data_types.add("equation")

        return sorted(list(data_types))

    def _extract_entities(self, chunks: List[LDU]) -> List[str]:
        """Extract named entities from chunks.

        Uses simple heuristics for now. Can be enhanced with NER models.

        Args:
            chunks: List of LDUs.

        Returns:
            List of extracted entity strings.
        """
        entities: Set[str] = set()

        # Simple heuristic: capitalized phrases, numbers with units, dates
        import re

        for chunk in chunks:
            content = chunk.content

            # Capitalized phrases (potential proper nouns)
            capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
            entities.update(cap for cap in capitalized if len(cap) > 3)

            # Numbers with currency or units
            currency = re.findall(r'\$[\d,]+(?:\.\d+)?', content)
            entities.update(currency)

            # Dates
            dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', content)
            entities.update(dates)

        # Limit to top entities (by frequency or importance)
        return sorted(list(entities))[:10]  # Top 10 entities

    def _generate_summary(self, section: Section, chunks: List[LDU]) -> str:
        """Generate LLM summary for a section.

        Args:
            section: Section to summarize.
            chunks: Chunks in this section.

        Returns:
            LLM-generated summary (2-3 sentences).
        """
        if not chunks:
            return f"Section '{section.title}' (pages {section.page_start}-{section.page_end})."

        # Collect content from chunks
        content_preview = "\n".join(
            chunk.content[:200] for chunk in chunks[:5]  # First 5 chunks, first 200 chars each
        )

        # Check if OpenRouter API key is available
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.warning("OPENROUTER_API_KEY not set, using heuristic summary")
            return self._generate_heuristic_summary(chunks)

        try:
            import requests

            prompt = f"""Summarize the following section from a document in 2-3 sentences.

Section Title: {section.title}
Pages: {section.page_start}-{section.page_end}
Content Preview:
{content_preview[:1000]}

Provide a concise summary of what this section covers:"""

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": "You are a document summarization assistant. Provide concise, factual summaries."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 150,
                    "temperature": 0.3,
                },
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                summary = result["choices"][0]["message"]["content"].strip()
                return summary
            else:
                logger.warning("LLM summary generation failed: %s", response.status_code)
                return self._generate_heuristic_summary(chunks)

        except Exception as e:
            logger.warning("LLM summary generation error: %s", e)
            return self._generate_heuristic_summary(chunks)

    def _generate_heuristic_summary(self, chunks: List[LDU]) -> str:
        """Generate a heuristic summary when LLM is unavailable.

        Args:
            chunks: Chunks to summarize.

        Returns:
            Heuristic summary string.
        """
        if not chunks:
            return "Empty section."

        # Count content types
        num_tables = sum(1 for c in chunks if c.chunk_type == "table")
        num_figures = sum(1 for c in chunks if c.chunk_type == "figure")
        num_lists = sum(1 for c in chunks if c.chunk_type == "list")

        parts = []
        if num_tables > 0:
            parts.append(f"{num_tables} table{'s' if num_tables > 1 else ''}")
        if num_figures > 0:
            parts.append(f"{num_figures} figure{'s' if num_figures > 1 else ''}")
        if num_lists > 0:
            parts.append(f"{num_lists} list{'s' if num_lists > 1 else ''}")

        content_summary = f"Contains {len(chunks)} content units"
        if parts:
            content_summary += f" including {', '.join(parts)}"

        return content_summary

    def _flatten_sections(self, sections: List[Section]) -> List[Section]:
        """Flatten a list of sections and their children recursively.

        Args:
            sections: List of sections.

        Returns:
            Flattened list of all sections.
        """
        result = []
        for section in sections:
            result.append(section)
            result.extend(self._flatten_sections(section.child_sections))
        return result

    def find_sections_by_topic(
        self,
        page_index: PageIndex,
        topic: str,
        top_k: int = 3,
    ) -> List[Section]:
        """Find top-k sections most relevant to a topic.

        Uses simple keyword matching. Can be enhanced with embeddings.

        Args:
            page_index: PageIndex to search.
            topic: Topic string to search for.
            top_k: Number of top sections to return.

        Returns:
            List of top-k sections most relevant to the topic.
        """
        topic_lower = topic.lower()
        all_sections = self._flatten_sections(page_index.root_sections)

        # Score sections by keyword matches
        scored_sections = []
        for section in all_sections:
            score = 0
            title_lower = section.title.lower()
            summary_lower = section.summary.lower()

            # Title matches are worth more
            if topic_lower in title_lower:
                score += 10
            if topic_lower in summary_lower:
                score += 5

            # Entity matches
            for entity in section.key_entities:
                if topic_lower in entity.lower():
                    score += 3

            if score > 0:
                scored_sections.append((score, section))

        # Sort by score and return top-k
        scored_sections.sort(key=lambda x: x[0], reverse=True)
        return [section for _, section in scored_sections[:top_k]]

    def save_page_index(
        self,
        page_index: PageIndex,
        output_dir: Path,
    ) -> Path:
        """Save PageIndex to JSON file.

        Args:
            page_index: PageIndex to save.
            output_dir: Directory to save to.

        Returns:
            Path to saved JSON file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{page_index.doc_id}_pageindex.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(page_index.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info("Saved PageIndex to %s", output_path)
        return output_path

    def load_page_index(self, page_index_path: Path) -> PageIndex:
        """Load PageIndex from JSON file.

        Args:
            page_index_path: Path to JSON file.

        Returns:
            Loaded PageIndex.
        """
        with open(page_index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PageIndex(**data)

    def _infer_sections_from_content(self, chunks: List[LDU]) -> SectionNode:
        """Infer sections from content patterns when explicit headers are missing.

        This method attempts to identify section boundaries by:
        1. Looking for paragraph chunks that might be headers (short, title-like)
        2. Using page boundaries as natural section breaks
        3. Grouping chunks by page ranges

        Args:
            chunks: List of LDUs.

        Returns:
            Root SectionNode with inferred sections.
        """
        import re
        from src.utils.section_chunker import SectionNode

        if not chunks:
            return SectionNode("Document Root", 0, 1)

        # Group chunks by page
        chunks_by_page: Dict[int, List[LDU]] = {}
        for chunk in chunks:
            for page_num in chunk.page_refs:
                if page_num not in chunks_by_page:
                    chunks_by_page[page_num] = []
                chunks_by_page[page_num].append(chunk)

        # Try to identify potential headers (short paragraphs that might be titles)
        potential_headers: List[Tuple[LDU, int]] = []  # (chunk, estimated_level)
        
        for chunk in chunks:
            if chunk.chunk_type == "paragraph":
                content = chunk.content.strip()
                # Look for title-like patterns
                # 1. Short text (likely headers)
                if len(content) < 150:
                    # 2. Check for numbered sections (1., 1.1, etc.)
                    numbered_match = re.match(r'^(\d+\.)+\s+(.+)', content)
                    if numbered_match:
                        level = content.count('.')
                        potential_headers.append((chunk, level))
                    # 3. Check for all caps (common for headers)
                    elif content.isupper() and len(content.split()) <= 10:
                        potential_headers.append((chunk, 1))
                    # 4. Check for common header patterns
                    elif any(pattern in content.lower() for pattern in [
                        'statement', 'report', 'summary', 'introduction', 'conclusion',
                        'chapter', 'section', 'part', 'financial', 'audit'
                    ]) and len(content.split()) <= 15:
                        potential_headers.append((chunk, 1))

        # If we found potential headers, use them to build sections
        if potential_headers:
            # Sort by page and position
            sorted_headers = sorted(
                potential_headers,
                key=lambda x: (
                    min(x[0].page_refs),
                    x[0].bounding_box.get("y0", 0),
                ),
            )

            root = SectionNode("Document Root", 0, min(c[0].page_refs[0] for c in sorted_headers if c[0].page_refs))
            stack: List[SectionNode] = [root]

            for header_chunk, level in sorted_headers:
                # Pop stack until we find appropriate parent
                while len(stack) > 1 and stack[-1].level >= level:
                    stack.pop()

                parent = stack[-1]
                page_num = min(header_chunk.page_refs) if header_chunk.page_refs else 1

                # Create section node
                section = SectionNode(
                    title=header_chunk.content.strip()[:100],  # Limit title length
                    level=level,
                    page_start=page_num,
                    parent=parent,
                )

                parent.add_child(section)
                stack.append(section)
                parent.update_page_end(page_num)

            logger.info(f"Inferred {len(root.children)} sections from content patterns")
            return root

        # Fallback: Create sections by page ranges (group every N pages)
        # This creates a basic structure even without headers
        pages = sorted(chunks_by_page.keys())
        if not pages:
            return SectionNode("Document Root", 0, 1)

        root = SectionNode("Document Root", 0, pages[0])
        pages_per_section = max(5, len(pages) // 10)  # Adaptive: ~10 sections per document

        for i in range(0, len(pages), pages_per_section):
            section_pages = pages[i:i + pages_per_section]
            page_start = section_pages[0]
            page_end = section_pages[-1]

            section = SectionNode(
                title=f"Pages {page_start}-{page_end}",
                level=1,
                page_start=page_start,
                parent=root,
            )
            root.add_child(section)
            root.update_page_end(page_end)

        logger.info(f"Created {len(root.children)} page-based sections as fallback")
        return root


__all__ = ["PageIndexBuilder"]
