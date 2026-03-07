"""Language detection utilities."""

from src.models.common import LanguageInfo


def detect_language(text: str) -> LanguageInfo:
    """Detect language from text (simplified - defaults to English)."""
    # Simplified implementation - in production, use a proper language detection library
    # For now, default to English
    return LanguageInfo(code="en", name="English", confidence=1.0)
