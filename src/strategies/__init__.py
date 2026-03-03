"""Extraction strategies for the Document Intelligence Refinery."""

from src.strategies.base import ExtractionStrategy
from src.strategies.fast_text import FastTextExtractor

__all__ = [
    "ExtractionStrategy",
    "FastTextExtractor",
]
