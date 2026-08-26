"""Prompt-injection detection (pattern-based, severity 1–3)."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

# (name, compiled pattern, severity 1-3)
_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}"
            r"\b(?:previous|prior|above|earlier|all)\b[^.\n]{0,20}"
            r"\b(?:instruction|prompt|rule|direction)",
            re.IGNORECASE,
        ),
        3,
    ),
    (
        "role_hijack",
        re.compile(
            r"\byou\s+are\s+now\b|\bact\s+as\s+(?:a\s+)?(?:system|admin|developer)\b"
            r"|\bnew\s+system\s+prompt\b|\bswitch\s+to\s+developer\s+mode\b",
            re.IGNORECASE,
        ),
        3,
    ),
    (
        "score_coercion",
        re.compile(
            r"\b(?:give|assign|set|return|output|rate)\b[^.\n]{0,30}"
            r"(?:100|10\s*/\s*10|max(?:imum)?\s+score|highest\s+score|perfect\s+score)"
            r"|\brank\s+(?:this|me)\s+(?:first|#?1|top)\b"
            r"|\b(?:must|always)\s+(?:select|hire|shortlist|recommend)\s+(?:this|me)\b",
            re.IGNORECASE,
        ),
        3,
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(?:reveal|print|repeat|show|output)\b[^.\n]{0,30}"
            r"\b(?:system\s+prompt|instructions|your\s+prompt|hidden\s+rules)\b",
            re.IGNORECASE,
        ),
        2,
    ),
    (
        "delimiter_injection",
        re.compile(
            r"<\s*/?\s*(?:system|assistant|user)\s*>|\[/?\s*(?:INST|SYS)\s*\]"
            r"|^\s*###\s*(?:system|instruction)s?\b",
            re.IGNORECASE | re.MULTILINE,
        ),
        2,
    ),
    (
        "hidden_text_marker",
        re.compile(
            r"font-size\s*:\s*0|color\s*:\s*#?fff(?:fff)?\b|opacity\s*:\s*0"
            r"|display\s*:\s*none|visibility\s*:\s*hidden",
            re.IGNORECASE,
        ),
        2,
    ),
    (
        "tool_or_exfil_attempt",
        re.compile(
            r"\b(?:curl|wget|fetch)\s+https?://|\bexec\s*\(|\bos\.system\b"
            r"|\bsend\s+(?:the\s+)?(?:data|resume|results)\s+to\s+https?://",
            re.IGNORECASE,
        ),
        3,
    ),
]

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")


class InjectionScan(BaseModel):
    """Result of scanning one document."""

    detected: bool = False
    severity: int = 0
    patterns: list[str] = Field(default_factory=list)
    spans: list[str] = Field(default_factory=list)
    sanitized_text: str = ""

    @property
    def should_reject(self) -> bool:
        return self.severity >= 3


def scan_for_injection(text: str, *, strip: bool = True) -> InjectionScan:
    """Detect and optionally strip injection attempts in ``text``.

    Zero-width and bidi control characters are always removed: they are the usual
    vehicle for hiding an instruction inside otherwise innocent-looking text.
    """
    original = text or ""
    cleaned = _ZERO_WIDTH_RE.sub("", original)

    hits: list[str] = []
    spans: list[str] = []
    severity = 0

    for name, pattern, sev in _PATTERNS:
        for match in pattern.finditer(cleaned):
            hits.append(name)
            spans.append(match.group(0)[:160])
            severity = max(severity, sev)

    if strip and spans:
        for name, pattern, _sev in _PATTERNS:
            cleaned = pattern.sub(f"[REMOVED:{name}]", cleaned)

    return InjectionScan(
        detected=bool(hits),
        severity=severity,
        patterns=sorted(set(hits)),
        spans=spans[:10],
        sanitized_text=cleaned,
    )
