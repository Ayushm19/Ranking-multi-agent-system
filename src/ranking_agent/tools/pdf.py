"""Document text extraction (deterministic tool, no LLM)."""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")


def clean_text(text: str) -> str:
    """Normalise whitespace without destroying line structure.

    Line structure has to survive: evidence quoting and section detection both
    depend on lines, so collapsing everything into one paragraph would break
    groundedness checks downstream.
    """
    out = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    out = out.replace("\u00a0", " ")
    out = _TRAILING_WS_RE.sub("\n", out)
    out = _MULTI_BLANK_RE.sub("\n\n", out)
    return out.strip()


def extract_pdf_text(raw: bytes) -> str:
    """Extract text from PDF bytes."""
    reader = PdfReader(io.BytesIO(raw))
    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - one bad page must not fail the file
            logger.warning("pdf page %d extraction failed: %s", i, exc)
    return clean_text("\n".join(pages))


def extract_text(filename: str, raw: bytes) -> str:
    """Dispatch on file extension. Callers validate the extension first."""
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(raw)
    return clean_text(raw.decode("utf-8", errors="replace"))
