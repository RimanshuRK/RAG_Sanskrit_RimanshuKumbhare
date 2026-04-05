import re
import unicodedata
from typing import List


class SanskritTextUtils:
    """Reusable utility methods for Sanskrit text."""

    # Devanagari Unicode range
    DEVANAGARI_RANGE = r"[\u0900-\u097F]"

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize Unicode using NFC form for Devanagari text."""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def remove_control_characters(text: str) -> str:
        """Remove non-printable and control characters."""
        return "".join(ch for ch in text if not unicodedata.category(ch).startswith("C"))

    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Normalize repeated spaces and newlines."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    @staticmethod
    def remove_pdf_artifacts(text: str) -> str:
        """Remove common artifacts introduced during PDF extraction."""
        # Patterns that look like standalone page numbers
        text = re.sub(r"\n\d+\n", "\n", text)
        # Hyphenated line breaks
        text = re.sub(r"-\n", "", text)
        return text

    @staticmethod
    def is_meaningful_chunk(text: str, min_chars: int = 50) -> bool:
        """Check whether a chunk contains meaningful content."""
        stripped = text.strip()
        return len(stripped) >= min_chars and any(c.isalpha() for c in stripped)

    @classmethod
    def full_clean(cls, text: str) -> str:
        """Apply the full text-cleaning pipeline in order."""
        text = cls.normalize_unicode(text)
        text = cls.remove_control_characters(text)
        text = cls.remove_pdf_artifacts(text)
        text = cls.clean_whitespace(text)
        return text

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split Sanskrit/Hindi text into sentences while handling common delimiters."""
        sentences = re.split(r"[।॥\.\?\!]+", text)
        return [s.strip() for s in sentences if s.strip()]
