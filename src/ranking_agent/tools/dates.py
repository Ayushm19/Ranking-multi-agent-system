"""Experience-duration maths. Total years = union of intervals, not sum."""

from __future__ import annotations

import re
from datetime import date

from ranking_agent.models.resume import ExperienceEntry

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_CURRENT_WORDS = ("present", "current", "till date", "todate", "now", "ongoing")

_ISO_RE = re.compile(r"^(\d{4})[-/](\d{1,2})$")
_YEAR_RE = re.compile(r"^(\d{4})$")
_MON_YEAR_RE = re.compile(r"([a-z]{3,9})[\s\-.,']*(\d{2,4})", re.IGNORECASE)


def parse_month(token: str | None, *, default_month: int = 1) -> tuple[int, int] | None:
    """Parse a loose date token into ``(year, month)``."""
    if not token:
        return None
    text = token.strip().lower()
    if not text:
        return None
    if any(word in text for word in _CURRENT_WORDS):
        today = date.today()
        return today.year, today.month

    if m := _ISO_RE.match(text):
        month = max(1, min(12, int(m.group(2))))
        return int(m.group(1)), month
    if m := _YEAR_RE.match(text):
        return int(m.group(1)), default_month
    if m := _MON_YEAR_RE.search(text):
        month = _MONTHS.get(m.group(1)[:4].rstrip(".").lower()[:4]) or _MONTHS.get(
            m.group(1)[:3].lower()
        )
        year = int(m.group(2))
        if year < 100:  # two-digit year
            year += 2000 if year <= (date.today().year % 100) else 1900
        if month:
            return year, month
        return year, default_month
    return None


def _to_months(ym: tuple[int, int]) -> int:
    return ym[0] * 12 + (ym[1] - 1)


def interval_of(entry: ExperienceEntry) -> tuple[int, int] | None:
    """Month-index interval ``[start, end)`` for one role."""
    start = parse_month(entry.start_date, default_month=1)
    if start is None:
        return None

    if entry.is_current or (
        entry.end_date and any(w in entry.end_date.lower() for w in _CURRENT_WORDS)
    ):
        today = date.today()
        end: tuple[int, int] | None = (today.year, today.month)
    else:
        end = parse_month(entry.end_date, default_month=12)

    if end is None:
        return None

    s, e = _to_months(start), _to_months(end)
    if e < s:
        return None
    return s, max(e, s + 1)  # single-month roles still count as 1 month


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of possibly overlapping intervals."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def total_years(entries: list[ExperienceEntry]) -> float | None:
    """Non-overlapping total years across all roles."""
    intervals = [iv for e in entries if (iv := interval_of(e)) is not None]
    if not intervals:
        return None
    months = sum(end - start for start, end in merge_intervals(intervals))
    return round(months / 12.0, 1)


def infer_experience_level(years: float | None) -> str:
    """Map years onto the weight-table buckets (7+/3-7/0-3)."""
    if years is None:
        return "MID"
    if years >= 7:
        return "SENIOR"
    if years >= 3:
        return "MID"
    return "FRESHER"
