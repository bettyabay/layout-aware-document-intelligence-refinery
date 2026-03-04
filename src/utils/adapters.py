"""Adapter utilities for normalising external parser outputs.

This module contains thin adapter classes that convert third-party document
representations (e.g. Docling's :class:`DoclingDocument`) into the internal
Pydantic models used by the refinery – in particular
``src.models.extracted_document.ExtractedDocument``.

The goal is to keep vendor-specific logic isolated, so strategies can work
against a stable internal schema.
"""

from __future__ import annotations

from typing import Any, List, Mapping

from src.models.extracted_document import (
    BoundingBox,
    ExtractedDocument,
    Figure,
    Table,
    TextBlock,
)


class DoclingAdapter:
    """Convert a DoclingDocument into an :class:`ExtractedDocument`.

    This adapter is intentionally defensive: Docling's internal data structures
    may evolve, so we primarily rely on duck-typing and attribute inspection
    instead of tight coupling to specific classes.

    Expected (but not strictly required) Docling structures:

    * ``document.pages`` – iterable of page-like objects
    * Each page exposes an ``elements`` or ``blocks`` collection with
      text/table/figure objects.
    * Elements expose:

      - ``text`` and ``bbox`` for text blocks
      - ``cells``/``rows`` or similar for tables
      - an image/figure-like type for figures

    The adapter maps these into the refinery's ``TextBlock``, ``Table`` and
    ``Figure`` models while preserving page numbers and bounding boxes.
    """

    @classmethod
    def to_extracted_document(cls, docling_document: Any) -> ExtractedDocument:
        """Convert a DoclingDocument-like object into ``ExtractedDocument``.

        Args:
            docling_document: Instance returned by Docling's converter.

        Returns:
            Normalised :class:`ExtractedDocument` instance.
        """
        text_blocks: List[TextBlock] = []
        tables: List[Table] = []
        figures: List[Figure] = []

        pages = getattr(docling_document, "pages", []) or []

        for page_index, page in enumerate(pages, start=1):
            # Docling may expose either ``elements`` or ``blocks`` collections.
            elements = getattr(page, "elements", None)
            if elements is None:
                elements = getattr(page, "blocks", [])

            for element in elements or []:
                cls._convert_element(
                    element=element,
                    page_index=page_index,
                    text_blocks=text_blocks,
                    tables=tables,
                    figures=figures,
                )

        reading_order = list(range(len(text_blocks)))

        return ExtractedDocument(
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            reading_order=reading_order,
        )

    # ------------------------------------------------------------------ #
    # Element dispatch
    # ------------------------------------------------------------------ #
    @classmethod
    def _convert_element(
        cls,
        element: Any,
        page_index: int,
        text_blocks: List[TextBlock],
        tables: List[Table],
        figures: List[Figure],
    ) -> None:
        """Dispatch a Docling element into the appropriate internal model."""
        type_name = type(element).__name__.lower()

        # Heuristic classification based on available attributes and type name.
        if hasattr(element, "cells") or "table" in type_name:
            table = cls._convert_table(element, page_index)
            if table is not None:
                tables.append(table)
            return

        if hasattr(element, "image") or "figure" in type_name or "image" in type_name:
            figure = cls._convert_figure(element, page_index)
            if figure is not None:
                figures.append(figure)
            return

        # Fallback: anything with ``text`` is treated as a text block.
        if hasattr(element, "text"):
            block = cls._convert_text_block(element, page_index)
            if block is not None:
                text_blocks.append(block)

    # ------------------------------------------------------------------ #
    # Converters
    # ------------------------------------------------------------------ #
    @classmethod
    def _bbox_from_docling(cls, bbox_obj: Any) -> BoundingBox | None:
        """Convert a Docling-style bbox into :class:`BoundingBox`.

        Supports a few common shapes by inspection:

        * Attributes: ``x0, y0, x1, y1``
        * Attributes: ``x, y, w, h`` → (x, y, x + w, y + h)
        * Mapping-like ``{'l', 't', 'r', 'b'}``
        """
        if bbox_obj is None:
            return None

        # Attribute-based bboxes
        if all(hasattr(bbox_obj, attr) for attr in ("x0", "y0", "x1", "y1")):
            return BoundingBox(
                x0=float(bbox_obj.x0),
                y0=float(bbox_obj.y0),
                x1=float(bbox_obj.x1),
                y1=float(bbox_obj.y1),
            )

        if all(hasattr(bbox_obj, attr) for attr in ("x", "y", "w", "h")):
            x = float(bbox_obj.x)
            y = float(bbox_obj.y)
            w = float(bbox_obj.w)
            h = float(bbox_obj.h)
            return BoundingBox(x0=x, y0=y, x1=x + w, y1=y + h)

        # Mapping-style bboxes
        if isinstance(bbox_obj, dict):
            keys = bbox_obj.keys()
            if {"l", "t", "r", "b"} <= keys:
                return BoundingBox(
                    x0=float(bbox_obj["l"]),
                    y0=float(bbox_obj["b"]),
                    x1=float(bbox_obj["r"]),
                    y1=float(bbox_obj["t"]),
                )

        return None

    @classmethod
    def _convert_text_block(cls, element: Any, page_index: int) -> TextBlock | None:
        """Convert a Docling text-like element into ``TextBlock``."""
        text = getattr(element, "text", "") or ""
        if not text.strip():
            return None

        bbox_obj = getattr(element, "bbox", None)
        bbox = cls._bbox_from_docling(bbox_obj)
        if bbox is None:
            # If no bbox is available, we still keep the text with a degenerate box.
            bbox = BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)

        return TextBlock(content=text, bbox=bbox, page_num=page_index)

    @classmethod
    def _convert_table(cls, element: Any, page_index: int) -> Table | None:
        """Convert a Docling table-like element into ``Table``."""
        # Try common Docling patterns: rows or cells matrices.
        headers: List[str] = []
        rows: List[List[str]] = []

        # Matrix-style cells [[header_row], [row1], ...]
        matrix = getattr(element, "cells", None) or getattr(element, "rows", None)
        if matrix:
            try:
                header_row = matrix[0]
                headers = [str(c) for c in header_row]
                for row in matrix[1:]:
                    rows.append([str(c) for c in row])
            except Exception:
                # Fallback to treating everything as body rows
                rows = [[str(c) for c in r] for r in matrix]

        if not headers and not rows:
            return None

        bbox_obj = getattr(element, "bbox", None)
        bbox = cls._bbox_from_docling(bbox_obj)
        if bbox is None:
            bbox = BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)

        return Table(headers=headers, rows=rows, bbox=bbox, page_num=page_index)

    @classmethod
    def _convert_figure(cls, element: Any, page_index: int) -> Figure | None:
        """Convert a Docling image/figure-like element into ``Figure``."""
        caption = getattr(element, "caption", "") or ""
        bbox_obj = getattr(element, "bbox", None)
        bbox = cls._bbox_from_docling(bbox_obj)
        if bbox is None:
            bbox = BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)

        return Figure(caption=caption, bbox=bbox, page_num=page_index)


__all__ = ["DoclingAdapter"]


class MinerUAdapter:
    """Adapter for converting MinerU-style JSON into ``ExtractedDocument``.

    MinerU's public APIs and output formats may evolve, and different pipelines
    can be configured to emit different JSON/Markdown schemas. Rather than
    binding tightly to a single version, this adapter expects a *normalised*
    mapping that follows a small, documented contract:

    Normalised schema (what this adapter expects):

    .. code-block:: json

        {
          "pages": [
            {
              "page_num": 1,
              "text_blocks": [
                {"text": "...", "bbox": {"x0": 10, "y0": 20, "x1": 100, "y1": 60}}
              ],
              "tables": [
                {
                  "headers": ["Col1", "Col2"],
                  "rows": [["v1", "v2"]],
                  "bbox": {"x0": 5, "y0": 5, "x1": 200, "y1": 120}
                }
              ],
              "figures": [
                {
                  "caption": "Some figure",
                  "bbox": {"x0": 30, "y0": 40, "x1": 180, "y1": 220}
                }
              ]
            }
          ]
        }

    A thin MinerU-specific parsing layer (inside the MinerU-based strategy) is
    responsible for transforming whatever MinerU actually outputs into this
    normalised mapping. This keeps all MinerU-version-specific logic out of the
    core models.
    """

    @classmethod
    def from_mineru_json(cls, data: Mapping[str, Any]) -> ExtractedDocument:
        """Convert a normalised MinerU JSON mapping into ``ExtractedDocument``.

        Args:
            data: Mapping following the schema documented in the class docstring.
                  It is typically produced by a MinerU extractor after parsing
                  MinerU's raw JSON/Markdown output.
        """
        pages = data.get("pages") or []

        text_blocks: List[TextBlock] = []
        tables: List[Table] = []
        figures: List[Figure] = []

        for page in pages:
            page_num = int(page.get("page_num", 0)) or 0

            # Text blocks
            for tb in page.get("text_blocks") or []:
                text = str(tb.get("text", "") or "")
                if not text.strip():
                    continue
                bbox = cls._bbox_from_mapping(tb.get("bbox"))
                text_blocks.append(
                    TextBlock(
                        content=text,
                        bbox=bbox,
                        page_num=page_num or 0,
                    )
                )

            # Tables
            for raw_table in page.get("tables") or []:
                headers = [str(h) for h in (raw_table.get("headers") or [])]
                rows_list = raw_table.get("rows") or []
                rows = [[str(cell) for cell in (row or [])] for row in rows_list]
                if not headers and not rows:
                    continue
                bbox = cls._bbox_from_mapping(raw_table.get("bbox"))
                tables.append(
                    Table(
                        headers=headers,
                        rows=rows,
                        bbox=bbox,
                        page_num=page_num or 0,
                    )
                )

            # Figures
            for raw_fig in page.get("figures") or []:
                caption = str(raw_fig.get("caption", "") or "")
                bbox = cls._bbox_from_mapping(raw_fig.get("bbox"))
                figures.append(
                    Figure(
                        caption=caption,
                        bbox=bbox,
                        page_num=page_num or 0,
                    )
                )

        reading_order = list(range(len(text_blocks)))

        return ExtractedDocument(
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            reading_order=reading_order,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bbox_from_mapping(bbox_obj: Any) -> BoundingBox:
        """Create a ``BoundingBox`` from a loose mapping.

        The mapping is expected to at least contain numeric ``x0, y0, x1, y1``
        fields. If the mapping is missing or incomplete, a degenerate bbox at
        the origin is returned so that downstream code always has a value.
        """
        if isinstance(bbox_obj, Mapping):
            try:
                x0 = float(bbox_obj.get("x0", 0.0))
                y0 = float(bbox_obj.get("y0", 0.0))
                x1 = float(bbox_obj.get("x1", 0.0))
                y1 = float(bbox_obj.get("y1", 0.0))
                return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
            except (TypeError, ValueError):
                pass

        # Safe fallback
        return BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)


__all__ = ["DoclingAdapter", "MinerUAdapter"]

