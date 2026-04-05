import re
import uuid
from typing import List
from app.models.domain import DocumentChunk
from app.utils.text_utils import SanskritTextUtils
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Service for cleaning and chunking Sanskrit text.

    Strategy:
    1. First, split stories/sections using `** heading **`
    2. If a section is too large, use a sliding window
    3. Keep small sections as-is so the story stays intact
    """

    # `** Bold headings **` format commonly produced by DOCX exports
    HEADING_PATTERN = re.compile(r'\*{1,2}[^*]+\*{1,2}')

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.utils = SanskritTextUtils()

    def process(self, filename: str, raw_text: str) -> List[DocumentChunk]:
        """
        Take a document's raw text and return cleaned chunks.
        Main entry point.
        """
        logger.info(f"Processing: {filename}")

        if not raw_text.strip():
            logger.warning(f"{filename}: Empty after cleaning.")
            return []

        # Step 1: Split sections using headings
        sections = self._split_by_headings(raw_text)
        logger.info(f"{filename}: {len(sections)} sections found by headings")

        doc_chunks = []
        chunk_index = 0

        for section_title, section_text in sections:
            cleaned = self.utils.full_clean(section_text)
            if not cleaned.strip():
                continue

            # Step 2: If the section is small, keep it as a single chunk
            if len(cleaned) <= self.chunk_size * 1.5:
                if self.utils.is_meaningful_chunk(cleaned):
                    content = f"{section_title}\n{cleaned}".strip() if section_title else cleaned
                    doc_chunks.append(
                        DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            source_file=filename,
                            content=content,
                            metadata={
                                "chunk_index": chunk_index,
                                "source": filename,
                                "section": section_title,
                            },
                        )
                    )
                    chunk_index += 1
            else:
                # Step 3: If the section is large, apply a sliding window
                sub_chunks = self._sliding_window_chunk(cleaned)
                for i, chunk_text in enumerate(sub_chunks):
                    if self.utils.is_meaningful_chunk(chunk_text):
                        # Add the heading only to the first chunk
                        content = f"{section_title}\n{chunk_text}".strip() if (section_title and i == 0) else chunk_text
                        doc_chunks.append(
                            DocumentChunk(
                                chunk_id=str(uuid.uuid4()),
                                source_file=filename,
                                content=content,
                                metadata={
                                    "chunk_index": chunk_index,
                                    "source": filename,
                                    "section": section_title,
                                    "sub_chunk": i,
                                },
                            )
                        )
                        chunk_index += 1

        logger.info(f"{filename}: {len(doc_chunks)} valid chunks created")
        return doc_chunks

    def _split_by_headings(self, text: str) -> List[tuple]:
        """
        Split text into sections using the `** heading **` pattern.

        Returns: List of (heading, content) tuples

        Example:
          Input:  "** मूर्खभृत्यस्य ** ...story... ** कालीदास ** ...story2..."
          Output: [("मूर्खभृत्यस्य", "...story..."), ("कालीदास", "...story2...")]
        """
        # Find headings and their positions
        matches = list(self.HEADING_PATTERN.finditer(text))

        if not matches:
            # No heading found, so treat the entire text as one section
            logger.info("No headings found, treating as single section")
            return [("", text)]

        sections = []

        # Text before the first heading, if present
        if matches[0].start() > 0:
            pre_text = text[:matches[0].start()].strip()
            if pre_text:
                sections.append(("", pre_text))

        # Content that follows each heading
        for i, match in enumerate(matches):
            heading = match.group().strip("* ").strip()
            content_start = match.end()
            content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[content_start:content_end].strip()
            sections.append((heading, content))

        return sections

    def _sliding_window_chunk(self, text: str) -> List[str]:
        """
        Sliding window chunking:
        - A `chunk_size` character window
        - A `chunk_overlap` character overlap
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            if end < text_len:
                boundary = self._find_sentence_boundary(text, end)
                if boundary > start:
                    end = boundary

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_len:
                break

            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = start + 1

            start = next_start

        return chunks

    def _find_sentence_boundary(self, text: str, pos: int, search_back: int = 100) -> int:
        """
        Snap the chunk end to the nearest sentence boundary.
        In Sanskrit, `।`, `॥`, and `.` are all valid boundaries.
        """
        search_start = max(0, pos - search_back)
        segment = text[search_start:pos]

        for i in range(len(segment) - 1, -1, -1):
            if segment[i] in ("।", "॥", ".", "?", "!"):
                return search_start + i + 1

        return pos
