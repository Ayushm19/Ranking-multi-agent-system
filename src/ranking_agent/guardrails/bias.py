"""Protected-attribute screening on generated reasoning."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "age",
        re.compile(
            r"\b(?:too\s+(?:old|young)|age(?:d)?\s+\d{2}|elderly|youthful"
            r"|born\s+in\s+(?:19|20)\d{2}|over\s+the\s+hill"
            r"|(?:younger|older)\s+candidate|generation(?:al)?\s+fit)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "gender",
        re.compile(
            r"\b(?:he|she|his|her)\s+(?:would|might|may)\s+(?:not\s+)?(?:fit|struggle)"
            r"|\b(?:male|female)\s+candidate\b|\bgender\b|\bmaternity\b|\bpaternity\b"
            r"|\bpregnan\w+",
            re.IGNORECASE,
        ),
    ),
    (
        "nationality_origin",
        re.compile(
            r"\b(?:foreign(?:er)?|immigrant|visa\s+risk|not\s+a\s+native"
            r"|accent|ethnic(?:ity)?|race|racial|caste|nationality\s+concern)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "religion",
        re.compile(
            r"\b(?:religio\w+|muslim|hindu|christian|jewish|sikh|buddhist"
            r"|church|mosque|temple)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "family_status",
        re.compile(
            r"\b(?:married|unmarried|single\s+parent|has\s+(?:kids|children)"
            r"|marital\s+status|family\s+commitments)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "disability_health",
        re.compile(
            r"\b(?:disab(?:led|ility)|handicap\w*|medical\s+condition|mental\s+health"
            r"|chronic\s+illness)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prestige_proxy",
        re.compile(
            r"\b(?:tier[\s\-]?\d\s+college|non[\s\-]?(?:iit|ivy|tier[\s\-]?1)"
            r"|not\s+from\s+a\s+(?:top|reputed|premier)\s+(?:college|university|school)"
            r"|only\s+(?:iit|nit|ivy))\b",
            re.IGNORECASE,
        ),
    ),
]


class BiasScan(BaseModel):
    flagged: bool = False
    categories: list[str] = Field(default_factory=list)
    spans: list[str] = Field(default_factory=list)
    sanitized_text: str = ""


def scan_for_bias(text: str) -> BiasScan:
    """Flag protected-attribute reasoning in one piece of text."""
    src = text or ""
    categories: list[str] = []
    spans: list[str] = []

    for name, pattern in _PATTERNS:
        for match in pattern.finditer(src):
            categories.append(name)
            spans.append(match.group(0)[:120])

    sanitized = src
    if spans:
        for _name, pattern in _PATTERNS:
            sanitized = pattern.sub("[REDACTED_BIAS]", sanitized)

    return BiasScan(
        flagged=bool(categories),
        categories=sorted(set(categories)),
        spans=spans[:10],
        sanitized_text=sanitized,
    )


def scan_many(texts: list[str]) -> BiasScan:
    """Aggregate scan over several rationales (one per dimension)."""
    categories: list[str] = []
    spans: list[str] = []
    for text in texts:
        result = scan_for_bias(text)
        categories.extend(result.categories)
        spans.extend(result.spans)
    return BiasScan(
        flagged=bool(categories),
        categories=sorted(set(categories)),
        spans=spans[:10],
        sanitized_text="",
    )
