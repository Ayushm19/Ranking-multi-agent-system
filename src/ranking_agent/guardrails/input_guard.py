"""Input validation for JD text and candidate documents."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ranking_agent.config import Settings, get_settings
from ranking_agent.guardrails.injection import InjectionScan, scan_for_injection

_RESUME_SIGNALS = ("@", "linkedin.com/in/", "github.com/", "curriculum vitae")


class GuardOutcome(BaseModel):
    """Whether the input may proceed, and the text to use if it may."""

    ok: bool = True
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    text: str = ""
    injection: InjectionScan | None = None

    @property
    def sanitized(self) -> bool:
        return bool(self.injection and self.injection.detected)


def check_jd(jd_text: str, settings: Settings | None = None) -> GuardOutcome:
    """Validate and sanitise a job description."""
    s = settings or get_settings()
    text = (jd_text or "").strip()
    failures: list[str] = []
    warnings: list[str] = []

    if not text:
        failures.append("jd_text is empty")
    elif len(text) < s.min_jd_chars:
        failures.append(
            f"jd_text too short ({len(text)} chars, minimum {s.min_jd_chars})"
        )
    if len(text) > s.max_jd_chars:
        warnings.append(f"jd_text truncated from {len(text)} to {s.max_jd_chars} chars")
        text = text[: s.max_jd_chars]

    # Detect resume pasted into the JD field.
    lowered = text.lower()
    if sum(1 for sig in _RESUME_SIGNALS if sig in lowered) >= 2:
        warnings.append("jd_text looks like a resume (contact-detail signals present)")

    scan = scan_for_injection(text)
    if scan.detected:
        warnings.append(
            f"prompt-injection patterns in jd_text: {', '.join(scan.patterns)}"
        )
        text = scan.sanitized_text
        if scan.severity >= s.injection_reject_severity:
            failures.append(
                f"jd_text rejected: injection severity {scan.severity} "
                f"({', '.join(scan.patterns)})"
            )

    return GuardOutcome(
        ok=not failures, failures=failures, warnings=warnings, text=text, injection=scan
    )


def check_document(
    filename: str,
    raw_bytes: bytes | None,
    settings: Settings | None = None,
) -> GuardOutcome:
    """Validate a candidate file before it is parsed."""
    s = settings or get_settings()
    failures: list[str] = []

    suffix = Path(filename or "").suffix.lower()
    if suffix not in s.allowed_suffixes:
        failures.append(
            f"unsupported file type {suffix or '(none)'}; "
            f"allowed: {', '.join(s.allowed_suffixes)}"
        )
    if raw_bytes is not None:
        if not raw_bytes:
            failures.append("file is empty")
        elif len(raw_bytes) > s.max_file_bytes:
            failures.append(
                f"file too large ({len(raw_bytes)} bytes, max {s.max_file_bytes})"
            )

    return GuardOutcome(ok=not failures, failures=failures)


def check_resume_text(text: str, settings: Settings | None = None) -> GuardOutcome:
    """Validate and sanitise extracted resume text."""
    s = settings or get_settings()
    body = (text or "").strip()
    failures: list[str] = []
    warnings: list[str] = []

    if len(body) < s.min_resume_chars:
        failures.append(
            f"extracted text too short ({len(body)} chars, minimum "
            f"{s.min_resume_chars}) — likely a scanned or empty document"
        )
    if len(body) > s.max_resume_chars:
        warnings.append(
            f"resume truncated from {len(body)} to {s.max_resume_chars} chars"
        )
        body = body[: s.max_resume_chars]

    scan = scan_for_injection(body)
    if scan.detected:
        warnings.append(
            f"prompt-injection patterns in resume: {', '.join(scan.patterns)}"
        )
        body = scan.sanitized_text
        if scan.severity >= s.injection_reject_severity:
            failures.append(
                f"resume rejected: injection severity {scan.severity} "
                f"({', '.join(scan.patterns)})"
            )

    return GuardOutcome(
        ok=not failures, failures=failures, warnings=warnings, text=body, injection=scan
    )
