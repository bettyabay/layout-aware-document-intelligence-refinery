"""Strategy router utilities for the extraction layer.

This module centralises the logic for choosing between different extraction
strategies based on:

* The ``DocumentProfile`` produced by the TriageAgent.
* Configuration loaded from ``rubric/extraction_rules.yaml``.

For now the router focuses on Strategy B (layout-aware extraction) and selects
between the Docling-based ``LayoutExtractor`` and the MinerU-based
``MinerUExtractor``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from src.models.document_profile import DocumentProfile
from src.strategies.base import ExtractionStrategy
from src.strategies.layout_aware import LayoutExtractor
from src.strategies.layout_mineru import MinerUExtractor


DEFAULT_RULES_PATH = Path("rubric") / "extraction_rules.yaml"


def _load_rules(rules_path: Path = DEFAULT_RULES_PATH) -> dict:
    """Load extraction rules YAML into a dict.

    The router is intentionally forgiving: if the file is missing or invalid,
    it falls back to an empty configuration and uses sensible defaults.
    """
    if not rules_path.exists():
        return {}

    try:
        with rules_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}
    return data


def select_layout_strategy(
    profile: DocumentProfile, rules_path: Path = DEFAULT_RULES_PATH
) -> Optional[ExtractionStrategy]:
    """Select an appropriate layout-aware extraction strategy.

    Selection is based on a combination of:

    * ``layout_strategy.engine`` from ``extraction_rules.yaml``.
    * The ``DocumentProfile`` (origin type and layout complexity).

    High-level policy:

    * If the config engine is ``mineru``:
        - Prefer ``MinerUExtractor`` when ``can_handle(profile)`` is True.
        - Otherwise fall back to ``LayoutExtractor`` if it can handle.
    * If the config engine is ``docling`` (default) or unknown:
        - Prefer ``LayoutExtractor`` when it can handle.
        - Otherwise fall back to ``MinerUExtractor`` if it can handle.

    Returns:
        An instantiated ``ExtractionStrategy`` or ``None`` if neither layout
        strategy is suitable for the profile.
    """
    rules = _load_rules(rules_path)
    layout_cfg = rules.get("layout_strategy", {}) or {}
    engine = str(layout_cfg.get("engine", "docling")).lower()

    docling_strategy = LayoutExtractor()
    mineru_strategy = MinerUExtractor()

    def docling_first() -> Optional[ExtractionStrategy]:
        if docling_strategy.can_handle(profile):
            return docling_strategy
        if mineru_strategy.can_handle(profile):
            return mineru_strategy
        return None

    def mineru_first() -> Optional[ExtractionStrategy]:
        if mineru_strategy.can_handle(profile):
            return mineru_strategy
        if docling_strategy.can_handle(profile):
            return docling_strategy
        return None

    if engine == "mineru":
        return mineru_first()
    # Default / unknown engine → Docling preferred
    return docling_first()


__all__ = ["select_layout_strategy", "DEFAULT_RULES_PATH"]

