from __future__ import annotations

from .rules import DEFAULT_RULES

try:
    from ftlangdetect import detect as ft_detect
except Exception:
    ft_detect = None


def detect_language(text: str) -> tuple[str, float]:
    cleaned = (text or "").strip()
    if not cleaned:
        return "und", 0.0

    if ft_detect is not None:
        try:
            result = ft_detect(text=cleaned, low_memory=True)
            return result.get("lang", "und"), float(result.get("score", 0.0))
        except Exception:
            pass

    # Lightweight fallback heuristic for Ge'ez script detection.
    geez_chars = sum(1 for ch in cleaned if "\u1200" <= ch <= "\u137F")
    ratio = geez_chars / max(len(cleaned), 1)
    if ratio > 0.2:
        return "am", 0.55

    latin_chars = sum(1 for ch in cleaned if ch.isascii() and ch.isalpha())
    latin_ratio = latin_chars / max(len(cleaned), 1)
    if latin_ratio > 0.3:
        return "en", 0.4

    return "und", 0.1
