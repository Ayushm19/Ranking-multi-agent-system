"""Evidence verification: exact then fuzzy quote match against the resume."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from ranking_agent.models.scoring import DimensionOpinion, Evidence, VerificationReport

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_MIN_QUOTE_CHARS = 12  # shorter spans ignored as evidence


def _normalize(text: str) -> str:
    lowered = (text or "").lower()
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", lowered)).strip()


def verify_quote(quote: str, source: str, *, threshold: int = 85) -> tuple[bool, int]:
    """Return ``(verified, similarity)`` for one quote against the source text."""
    q_norm = _normalize(quote)
    if len(q_norm) < _MIN_QUOTE_CHARS:
        return False, 0

    s_norm = _normalize(source)
    if not s_norm:
        return False, 0
    if q_norm in s_norm:
        return True, 100

    score = int(fuzz.partial_ratio(q_norm, s_norm))
    return score >= threshold, score


def verify_opinion(
    opinion: DimensionOpinion, source: str, *, threshold: int = 85
) -> DimensionOpinion:
    """Annotate each piece of evidence on one opinion in place."""
    for item in opinion.evidence:
        verified, score = verify_quote(item.quote, source, threshold=threshold)
        item.verified = verified
        item.match_score = score
    return opinion


def build_report(
    opinions: list[DimensionOpinion], source: str, *, threshold: int = 85
) -> VerificationReport:
    """Verify every opinion and summarise groundedness across the run."""
    unverified: list[Evidence] = []
    total = 0
    verified_count = 0
    notes: list[str] = []

    for opinion in opinions:
        verify_opinion(opinion, source, threshold=threshold)
        for item in opinion.evidence:
            total += 1
            if item.verified:
                verified_count += 1
            else:
                unverified.append(item)
        if opinion.evidence and opinion.groundedness < 0.5:
            notes.append(
                f"{opinion.dimension}: only {opinion.groundedness:.0%} of quotes "
                "were found in the resume"
            )
        if not opinion.evidence and not opinion.degraded:
            notes.append(f"{opinion.dimension}: scored without citing any evidence")

    groundedness = 1.0 if total == 0 else verified_count / total
    return VerificationReport(
        total_claims=total,
        verified_claims=verified_count,
        unverified=unverified[:20],
        groundedness=round(groundedness, 4),
        notes=notes,
    )
