"""PII redaction before prompting, with deterministic contact recovery."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ranking_agent.models.resume import ContactInfo

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"(?:(?:\+|\()\d{1,4}\)?[\s\-.]?)?"
    r"(?:\d{3}[\s\-.]?\d{3}[\s\-.]?\d{4}|\d{10,13})\b"
)
_LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/(?:in|pub)/[A-Za-z0-9_\-%]+/?",
    re.IGNORECASE,
)
_GITHUB_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_\-.]+/?", re.IGNORECASE
)
_URL_RE = re.compile(r"https?://[^\s<>\")]+", re.IGNORECASE)


class RedactionResult(BaseModel):
    """Redacted text plus the contact details recovered from the original."""

    text: str
    contact: ContactInfo = Field(default_factory=ContactInfo)
    redactions: int = 0
    placeholders: dict[str, str] = Field(default_factory=dict)


def extract_contact(text: str) -> ContactInfo:
    """Deterministically pull contact fields out of raw resume text."""
    src = text or ""
    emails = _EMAIL_RE.findall(src)
    linkedin = _LINKEDIN_RE.search(src)
    github = _GITHUB_RE.search(src)

    phone: str | None = None
    for match in _PHONE_RE.finditer(src):
        digits = re.sub(r"\D", "", match.group(0))
        # 10-13 digits filters out years, zip codes, and bare metric numbers.
        if 10 <= len(digits) <= 13:
            phone = match.group(0).strip()
            break

    return ContactInfo(
        name=_guess_name(src),
        email=emails[0] if emails else None,
        phone=phone,
        linkedin=linkedin.group(0) if linkedin else None,
        github=github.group(0) if github else None,
    )


def _guess_name(text: str) -> str | None:
    """First plausible person-name line in the header region."""
    for line in (text or "").splitlines()[:8]:
        s = line.strip().strip("|-–—•*")
        if not 3 < len(s) < 48:
            continue
        if _EMAIL_RE.search(s) or _URL_RE.search(s) or any(ch.isdigit() for ch in s):
            continue
        words = s.split()
        if not 1 < len(words) <= 4:
            continue
        if all(w[:1].isupper() for w in words if w):
            lowered = s.lower()
            if any(k in lowered for k in ("resume", "curriculum", "vitae", "profile")):
                continue
            return s
    return None


def redact(text: str, *, enabled: bool = True) -> RedactionResult:
    """Replace contact PII with typed placeholders."""
    contact = extract_contact(text)
    if not enabled:
        return RedactionResult(text=text or "", contact=contact, redactions=0)

    out = text or ""
    placeholders: dict[str, str] = {}
    count = 0

    def _sub(pattern: re.Pattern[str], label: str, body: str) -> str:
        nonlocal count
        seen: dict[str, str] = {}

        def repl(match: re.Match[str]) -> str:
            nonlocal count
            value = match.group(0)
            if value not in seen:
                seen[value] = f"[{label}_{len(seen) + 1}]"
                placeholders[seen[value]] = value
                count += 1
            return seen[value]

        return pattern.sub(repl, body)

    # Order matters: social URLs before the generic URL rule.
    out = _sub(_EMAIL_RE, "EMAIL", out)
    out = _sub(_LINKEDIN_RE, "LINKEDIN", out)
    out = _sub(_GITHUB_RE, "GITHUB", out)
    out = _sub(_PHONE_RE, "PHONE", out)

    return RedactionResult(
        text=out, contact=contact, redactions=count, placeholders=placeholders
    )
