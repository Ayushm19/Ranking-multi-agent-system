"""Deterministic offline LLM provider (hash + keyword overlap; no API key)."""

from __future__ import annotations

import hashlib
import json
import re

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.\-]{1,}")
_STOP = {
    "the", "and", "for", "with", "you", "our", "will", "are", "have", "this",
    "that", "from", "your", "who", "all", "not", "但", "job", "role", "work",
    "team", "years", "year", "experience", "candidate", "resume", "jd",
}


def _tokens(text: str) -> set[str]:
    return {
        w.lower()
        for w in _WORD_RE.findall(text or "")
        if len(w) > 2 and w.lower() not in _STOP
    }


def _stable_unit(seed: str) -> float:
    """Deterministic float in [0, 1) from a string."""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _overlap_score(user_prompt: str) -> tuple[float, list[str]]:
    """Keyword-overlap score from ``[JOB]`` / ``[RESUME]`` sections."""
    jd_part = ""
    resume_part = ""
    lowered = user_prompt.lower()
    if "[job]" in lowered and "[resume]" in lowered:
        idx_job = lowered.index("[job]")
        idx_res = lowered.index("[resume]")
        if idx_job < idx_res:
            jd_part = user_prompt[idx_job:idx_res]
            resume_part = user_prompt[idx_res:]
        else:
            resume_part = user_prompt[idx_res:idx_job]
            jd_part = user_prompt[idx_job:]

    jd_tokens = _tokens(jd_part)
    res_tokens = _tokens(resume_part)
    if not jd_tokens or not res_tokens:
        return 40.0 + _stable_unit(user_prompt) * 40.0, []

    shared = sorted(jd_tokens & res_tokens)
    ratio = len(shared) / max(len(jd_tokens), 1)
    # Map coverage onto a 25–95 band.
    score = 25.0 + min(ratio, 0.8) / 0.8 * 70.0
    return round(min(score, 95.0), 1), shared[:12]


def _find_quote(user_prompt: str, term: str) -> str:
    """Return a prompt line containing ``term`` (must be a real substring)."""
    for line in user_prompt.splitlines():
        stripped = line.strip()
        if len(stripped) > 15 and term.lower() in stripped.lower():
            return stripped[:160]
    return ""


def _evidence(user_prompt: str, terms: list[str], limit: int = 3) -> list[dict]:
    out: list[dict] = []
    for term in terms:
        quote = _find_quote(user_prompt, term)
        if quote:
            out.append({"quote": quote, "claim": f"Candidate shows {term}"})
        if len(out) >= limit:
            break
    # Prefer real resume lines over invented quotes.
    if len(out) < limit:
        resume_part = user_prompt
        lowered = user_prompt.lower()
        if "[resume]" in lowered:
            resume_part = user_prompt[lowered.index("[resume]"):]
        for line in resume_part.splitlines():
            stripped = line.strip()
            if len(stripped) < 20 or stripped.startswith("["):
                continue
            if any(e["quote"] == stripped[:160] for e in out):
                continue
            out.append({"quote": stripped[:160], "claim": "Evidence from resume"})
            if len(out) >= limit:
                break
    return out


def _dimension_payload(user_prompt: str, dimension: str) -> dict:
    base, shared = _overlap_score(user_prompt)
    # Per-dimension offset so scores are not identical.
    offset = (_stable_unit(dimension + user_prompt[:200]) - 0.5) * 12
    score = round(max(0.0, min(100.0, base + offset)), 1)
    return {
        "raw_score": score,
        "confidence": round(0.55 + _stable_unit(dimension) * 0.35, 2),
        "reasoning": (
            f"Mock evaluation of {dimension}: matched {len(shared)} JD terms "
            f"in the candidate profile ({', '.join(shared[:6]) or 'no direct overlap'})."
        ),
        "evidence": _evidence(user_prompt, shared or ["experience"]),
    }


def _jd_payload(user_prompt: str) -> dict:
    """Extract a usable job profile from JD text.

    Prefer known skill vocabulary over raw token frequency — otherwise the mock
    treats stopwords like "ability" and "across" as must-have skills and the
    Missing Must-Have penalty becomes noise.
    """
    from ranking_agent.tools.skills import _ALIASES, normalize_skill

    years = None
    m = re.search(
        r"(\d{1,2})\s*(?:\+|plus)?\s*(?:-|to|–|—)?\s*(\d{1,2})?\s*\+?\s*(?:years|yrs)",
        user_prompt,
        re.IGNORECASE,
    )
    if m:
        years = float(m.group(1))

    title = None
    for line in user_prompt.splitlines():
        s = line.strip()
        if 8 < len(s) < 80 and not s.startswith("[") and "hiring" not in s.lower():
            title = s
            break

    lowered = user_prompt.lower()
    found: list[str] = []
    for canon, aliases in _ALIASES.items():
        needles = (canon.lower(), *aliases)
        if any(re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", lowered) for a in needles):
            found.append(canon)

    # Hyphen/bullet skill-looking lines.
    for line in user_prompt.splitlines():
        s = line.strip(" -\t•*")
        if not s or len(s) > 60:
            continue
        if any(s.lower().startswith(p) for p in ("must", "nice", "you will", "we are")):
            continue
        if normalize_skill(s) and normalize_skill(s) not in found:
            if len(s.split()) <= 4 and not s.endswith("."):
                found.append(normalize_skill(s) or s)

    # Dedupe, preserve order.
    seen: set[str] = set()
    caps: list[str] = []
    for c in found:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            caps.append(c)
    if not caps:
        caps = ["software engineering"]

    seniority = "senior" if (years or 0) >= 7 else "mid" if (years or 0) >= 3 else "junior"
    return {
        "role_title": title,
        "role_family": "engineering",
        "seniority": seniority,
        "required_years_min": years,
        "required_years_max": None,
        "domain": None,
        "industry": None,
        "responsibilities": [],
        "capabilities": [
            {
                "name": c,
                "normalized": normalize_skill(c) or c,
                "importance": "must_have" if i < min(6, len(caps)) else "nice_to_have",
                "category": None,
            }
            for i, c in enumerate(caps[:12])
        ],
        "scale_signals": [],
    }


def _resume_payload(user_prompt: str) -> dict:
    tokens = sorted(_tokens(user_prompt))
    lines = [ln.strip() for ln in user_prompt.splitlines() if ln.strip()]
    return {
        "name": None,
        "summary": next((ln for ln in lines if len(ln) > 60), None),
        "skills": tokens[:20],
        "experiences": [
            {
                "title": None,
                "company": None,
                "start_date": None,
                "end_date": None,
                "is_current": False,
                "description": "\n".join(lines[:8]),
                "skills": tokens[:8],
            }
        ],
        "educations": [],
        "projects": [],
        "certifications": [],
        "others": [],
    }


def _critic_payload(user_prompt: str) -> dict:
    unit = _stable_unit(user_prompt[:400])
    return {
        "verdict": "accept" if unit > 0.15 else "revise",
        "confidence": round(0.6 + unit * 0.3, 2),
        "issues": [] if unit > 0.15 else ["Evidence coverage is thin for one dimension"],
        "suggested_dimension_revisions": {},
        "reasoning": "Mock critic: dimension spread and evidence look internally consistent.",
    }


def _judge_payload(user_prompt: str) -> dict:
    unit = _stable_unit(user_prompt[:400])
    verdict = "correct" if unit > 0.4 else "partially_correct" if unit > 0.1 else "incorrect"
    return {
        "verdict": verdict,
        "score_error_estimate": round((unit - 0.5) * 10, 1),
        "dimension_disagreements": [],
        "reasoning": "Mock judge: score is broadly defensible given the stated evidence.",
    }


def _skill_payload(user_prompt: str) -> dict:
    payload = _dimension_payload(user_prompt, "skill_match")
    _, shared = _overlap_score(user_prompt)
    payload["skill_matches"] = [
        {
            "skill": s,
            "match_type": "exact",
            "score": 100.0,
            "resume_evidence": _find_quote(user_prompt, s),
        }
        for s in shared[:8]
    ]
    return payload


_HANDLERS = {
    "jd_analyst": _jd_payload,
    "resume_parser": _resume_payload,
    "skill_match": _skill_payload,
    "critic": _critic_payload,
    "judge": _judge_payload,
}


def mock_completion(task: str, user_prompt: str) -> str:
    """Return a deterministic JSON string for ``task``."""
    handler = _HANDLERS.get(task)
    if handler is not None:
        return json.dumps(handler(user_prompt))
    # Any evaluator dimension (experience_quality, domain_context, ...).
    return json.dumps(_dimension_payload(user_prompt, task))
