"""Extraction strategies for the Document Intelligence Refinery."""

from src.strategies.base import ExtractionStrategy
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout_aware import LayoutExtractor
from src.strategies.layout_mineru import MinerUExtractor
from src.strategies.router import DEFAULT_RULES_PATH, select_layout_strategy

__all__ = [
    "ExtractionStrategy",
    "FastTextExtractor",
    "LayoutExtractor",
    "MinerUExtractor",
    "select_layout_strategy",
    "DEFAULT_RULES_PATH",
]
