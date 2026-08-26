"""Skill normalisation and deterministic matching."""

from __future__ import annotations

import re

from rapidfuzz import fuzz, process

from ranking_agent.models.scoring import SkillMatch

# Canonical form -> aliases.
_ALIASES: dict[str, tuple[str, ...]] = {
    "Node.js": ("node", "nodejs", "node js", "node.js"),
    "JavaScript": ("js", "javascript", "ecmascript"),
    "TypeScript": ("ts", "typescript"),
    "Python": ("python", "python3", "py"),
    "PostgreSQL": ("postgres", "postgresql", "psql", "post gres"),
    "MongoDB": ("mongo", "mongodb"),
    "Kubernetes": ("k8s", "kubernetes", "kube"),
    "Docker": ("docker", "containerization", "containerisation"),
    "AWS": ("aws", "amazon web services"),
    "Azure": ("azure", "microsoft azure"),
    "GCP": ("gcp", "google cloud", "google cloud platform"),
    "CI/CD": ("ci/cd", "cicd", "ci cd", "continuous integration"),
    "React": ("react", "reactjs", "react.js"),
    "Express": ("express", "expressjs", "express.js"),
    "Next.js": ("next", "nextjs", "next.js"),
    "NestJS": ("nestjs", "nest js", "nest.js"),
    "FastAPI": ("fastapi", "fast api"),
    "Django": ("django",),
    "Flask": ("flask",),
    "Spring Boot": ("spring boot", "springboot", "spring-boot"),
    "Java": ("java", "core java"),
    "Go": ("go", "golang"),
    "SQL": ("sql", "t-sql", "ansi sql"),
    "REST APIs": (
        "rest",
        "rest api",
        "rest apis",
        "restful",
        "restful apis",
        "restful api",
        "api development",
        "apis",
    ),
    "Backend Development": (
        "backend",
        "backend development",
        "backend engineering",
        "backend engineer",
        "server side",
        "server-side",
        "server side development",
    ),
    "GraphQL": ("graphql", "graph ql"),
    "Kafka": ("kafka", "apache kafka"),
    "Redis": ("redis",),
    "Terraform": ("terraform", "hcl"),
    "Ansible": ("ansible",),
    "Machine Learning": ("ml", "machine learning"),
    "GenAI": (
        "genai",
        "gen ai",
        "generative ai",
        "llm",
        "llms",
        "large language model",
        "large language models",
        "llm integrations",
        "genai / llm integrations",
    ),
    "PyTorch": ("pytorch", "torch"),
    "TensorFlow": ("tensorflow", "tf"),
    "Selenium": ("selenium", "selenium webdriver"),
    "Microservices": ("microservices", "micro services", "microservice"),
}

_ALIAS_TO_CANON: dict[str, str] = {
    alias: canon for canon, aliases in _ALIASES.items() for alias in aliases
}

# Transferability families (same family ≠ exact match).
_FAMILIES: dict[str, set[str]] = {
    "cloud": {"AWS", "Azure", "GCP"},
    "container_orchestration": {"Kubernetes", "Docker"},
    "relational_db": {"PostgreSQL", "SQL"},
    "python_web": {"Python", "Django", "Flask", "FastAPI"},
    "jvm": {"Java", "Spring Boot"},
    "js_runtime": {
        "JavaScript",
        "TypeScript",
        "Node.js",
        "Express",
        "Next.js",
        "NestJS",
        "React",
    },
    "ml_framework": {"PyTorch", "TensorFlow", "Machine Learning", "GenAI"},
    "api_style": {"REST APIs", "GraphQL"},
    "backend": {
        "Backend Development",
        "REST APIs",
        "Microservices",
        "FastAPI",
        "Django",
        "Flask",
        "Express",
        "NestJS",
        "Spring Boot",
        "Node.js",
        "Python",
        "Java",
        "Go",
    },
}

# Umbrella JD skill → concrete resume skills that cover it.
_COVERED_BY: dict[str, frozenset[str]] = {
    "REST APIs": frozenset(
        {
            "FastAPI",
            "Django",
            "Flask",
            "Express",
            "NestJS",
            "Spring Boot",
            "Node.js",
            "Next.js",
            "GraphQL",
        }
    ),
    "Backend Development": frozenset(
        {
            "FastAPI",
            "Django",
            "Flask",
            "Express",
            "NestJS",
            "Spring Boot",
            "Node.js",
            "Python",
            "Java",
            "Go",
            "Microservices",
        }
    ),
    "Microservices": frozenset(
        {
            "Kubernetes",
            "Docker",
            "Kafka",
            "FastAPI",
            "Spring Boot",
            "Express",
            "NestJS",
            "Node.js",
        }
    ),
    # Framework implies language.
    "Python": frozenset({"FastAPI", "Django", "Flask", "PyTorch", "TensorFlow"}),
    "Java": frozenset({"Spring Boot"}),
    "JavaScript": frozenset({"TypeScript", "Node.js", "React", "Next.js", "Express", "NestJS"}),
    "TypeScript": frozenset({"Next.js", "NestJS"}),
    "SQL": frozenset({"PostgreSQL"}),
}

_PUNCT_RE = re.compile(r"[^\w\s+#./\-]")
_WS_RE = re.compile(r"\s+")
# Split OR-alternatives in one JD cell: "JS/TS", "AWS or GCP".
_ALT_SPLIT_RE = re.compile(r"\s*(?:/|\bor\b|\||&)\s*", re.IGNORECASE)
_PAREN_ALTS_RE = re.compile(r"\(([^)]+)\)")


def normalize_skill(raw: str) -> str:
    """Map a raw skill mention onto its canonical form."""
    cleaned = _WS_RE.sub(" ", _PUNCT_RE.sub(" ", (raw or "").lower())).strip()
    if not cleaned:
        return ""
    if cleaned in _ALIAS_TO_CANON:
        return _ALIAS_TO_CANON[cleaned]

    # Fuzzy alias (≥90) for typos / spacing.
    best = process.extractOne(cleaned, list(_ALIAS_TO_CANON), scorer=fuzz.ratio)
    if best and best[1] >= 90:
        return _ALIAS_TO_CANON[best[0]]

    return " ".join(w.capitalize() if w.islower() else w for w in cleaned.split())


def normalize_all(skills: list[str]) -> list[str]:
    """Normalise and de-duplicate, preserving first-seen order."""
    seen: dict[str, None] = {}
    for skill in skills:
        canon = normalize_skill(skill)
        if canon:
            seen.setdefault(canon, None)
    return list(seen)


def requirement_alternatives(raw: str) -> list[str]:
    """Expand a JD skill cell into OR-alternatives."""
    text = (raw or "").strip()
    if not text:
        return []

    parts: list[str] = []
    paren = _PAREN_ALTS_RE.search(text)
    if paren and _ALT_SPLIT_RE.search(paren.group(1)):
        parts = [p for p in _ALT_SPLIT_RE.split(paren.group(1)) if p.strip()]
    elif _ALT_SPLIT_RE.search(text):
        parts = [
            p for p in _ALT_SPLIT_RE.split(_PAREN_ALTS_RE.sub(" ", text)) if p.strip()
        ]

    if len(parts) >= 2:
        alts = normalize_all(parts)
        return alts or [normalize_skill(text)]

    canon = normalize_skill(text)
    return [canon] if canon else []


def _family_of(skill: str) -> str | None:
    return next((fam for fam, members in _FAMILIES.items() if skill in members), None)


def _label_for_requirement(raw: str, alts: list[str]) -> str:
    """Stable display name for a (possibly compound) requirement."""
    if len(alts) == 1:
        return alts[0]
    whole = normalize_skill(raw)
    if whole and whole in _ALIASES:
        return whole
    return " / ".join(alts)


def match_skills(
    required: list[str],
    candidate: list[str],
    *,
    fuzzy_threshold: int = 88,
) -> list[SkillMatch]:
    """Classify each required skill: exact / transferable / partial / gap."""
    cand_canon = normalize_all(candidate)
    cand_set = set(cand_canon)
    matches: list[SkillMatch] = []

    for raw in required:
        alts = requirement_alternatives(raw)
        if not alts:
            continue
        label = _label_for_requirement(raw, alts)

        hit = next((a for a in alts if a in cand_set), None)
        if hit:
            matches.append(
                SkillMatch(
                    skill=label,
                    match_type="exact",
                    score=100.0,
                    resume_evidence=hit,
                )
            )
            continue

        # Covered by related concrete skill on the resume.
        covered_by = None
        for alt in alts:
            evidence = next(
                (c for c in cand_canon if c in _COVERED_BY.get(alt, frozenset())),
                None,
            )
            if evidence:
                covered_by = (alt, evidence)
                break
        if covered_by:
            alt, evidence = covered_by
            matches.append(
                SkillMatch(
                    skill=label,
                    match_type="exact",
                    score=100.0,
                    resume_evidence=f"{evidence} (covers {alt})",
                )
            )
            continue

        family_hit = None
        for alt in alts:
            family = _family_of(alt)
            if not family:
                continue
            sibling = next(
                (c for c in cand_canon if c != alt and _family_of(c) == family),
                None,
            )
            if sibling:
                family_hit = (alt, sibling, family)
                break
        if family_hit:
            alt, sibling, family = family_hit
            matches.append(
                SkillMatch(
                    skill=label,
                    match_type="transferable",
                    score=70.0,
                    resume_evidence=f"{sibling} (same {family.replace('_', ' ')})",
                )
            )
            continue

        if cand_canon:
            best = None
            best_score = -1.0
            for alt in alts:
                found = process.extractOne(
                    alt, cand_canon, scorer=fuzz.token_set_ratio
                )
                if found and found[1] > best_score:
                    best = found
                    best_score = found[1]
            if best and best_score >= fuzzy_threshold:
                matches.append(
                    SkillMatch(
                        skill=label,
                        match_type="partial",
                        score=45.0,
                        resume_evidence=best[0],
                    )
                )
                continue

        matches.append(SkillMatch(skill=label, match_type="gap", score=0.0))

    return matches


def coverage_score(matches: list[SkillMatch]) -> float:
    """Weighted coverage of required skills, 0-100."""
    if not matches:
        return 50.0  # no stated requirements is not evidence of a bad candidate
    return round(sum(m.score for m in matches) / len(matches), 1)
