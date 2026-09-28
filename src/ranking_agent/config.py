"""Runtime configuration from environment / ``.env``."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["mock", "openai", "azure", "gemini"]
ExperienceLevel = Literal["SENIOR", "MID", "FRESHER"]

DEFAULT_MODELS: dict[str, str] = {
    "mock": "mock",
    "openai": "gpt-4o-mini",
    "azure": "gpt-4o-mini",
    "gemini": "gemini-3.5-flash-lite",
}

DEFAULT_JUDGE_MODELS: dict[str, str] = {
    "mock": "mock",
    "openai": "gpt-4o",
    "azure": "gpt-4o",
    "gemini": "gemini-3.5-flash",
}

_GEMINI_3X_RE = re.compile(r"^gemini-3", re.IGNORECASE)


def omit_temperature(model: str) -> bool:
    """True for Gemini 3.x — omit temperature; provider default is required."""
    return bool(_GEMINI_3X_RE.match(model or ""))

# Per experience level; each row sums to 1.0.
DIMENSION_WEIGHTS: dict[str, dict[str, float]] = {
    "SENIOR": {
        "role_fit": 0.15,
        "capability_match": 0.20,
        "skill_match": 0.20,
        "experience_quality": 0.20,
        "scale_impact": 0.15,
        "domain_context": 0.10,
    },
    "MID": {
        "role_fit": 0.15,
        "capability_match": 0.30,
        "skill_match": 0.20,
        "experience_quality": 0.20,
        "scale_impact": 0.05,
        "domain_context": 0.10,
    },
    "FRESHER": {
        "role_fit": 0.15,
        "capability_match": 0.15,
        "skill_match": 0.35,
        "experience_quality": 0.05,
        "scale_impact": 0.05,
        "domain_context": 0.25,
    },
}

DIMENSION_LABELS: dict[str, str] = {
    "role_fit": "Role Fit",
    "capability_match": "Core Capability Match",
    "skill_match": "Skill Match",
    "experience_quality": "Experience Quality",
    "scale_impact": "Scale & Impact",
    "domain_context": "Domain & Context",
}

# USD per 1M tokens (input, output). Unknown models → (0, 0).
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "mock": (0.0, 0.0),
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    # Gemini — documented
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),
    # Gemini — estimated (same tier)
    "gemini-3.5-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash": (0.50, 3.00),
    "gemini-3.6-flash": (0.50, 3.00),
    "gemini-3.7-flash": (0.50, 3.00),
    # Gemini — older free-tier fallback chain (separate quota pools)
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
}

# Tried in order after the configured model hits 429/503/404; each Gemini
# model has its own free-tier quota, so a different model often still has
# headroom. Gemini 2.x/1.5 are sunset for new API keys as of this account —
# stick to the gemini-3.x family that's actually reachable.
DEFAULT_GEMINI_MODEL_FALLBACKS: tuple[str, ...] = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
)


from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Environment-driven settings."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE.is_file() else ".env",
        env_prefix="RANKING_",
        extra="ignore",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    # ── LLM ──
    provider: Provider = "gemini"
    model: str = ""  # blank → DEFAULT_MODELS[provider]
    judge_model: str = ""
    temperature: float = 0.0
    max_output_tokens: int = 4096
    request_timeout_s: float = 90.0

    gemini_api_key: str | None = None
    gemini_api_key_fallback: str | None = None  # rotate on 429
    gemini_model_fallbacks: str = ""  # comma-separated; blank → DEFAULT_GEMINI_MODEL_FALLBACKS
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    azure_api_key: str | None = None
    azure_endpoint: str | None = None
    azure_api_version: str = "2024-12-01-preview"

    # ── Reliability ──
    max_retries: int = 3
    retry_base_delay_s: float = 0.5
    batch_concurrency: int = 4
    agent_concurrency: int = 6
    max_repair_loops: int = 1
    run_budget_usd: float = 1.0

    # ── Guardrails ──
    max_file_bytes: int = 10 * 1024 * 1024
    min_resume_chars: int = 200
    max_resume_chars: int = 120_000
    min_jd_chars: int = 80
    max_jd_chars: int = 60_000
    allowed_suffixes: tuple[str, ...] = (".pdf", ".txt", ".md")
    max_candidates_per_batch: int = 30
    injection_reject_severity: int = 3
    enable_pii_redaction: bool = True
    enable_bias_screen: bool = True

    # ── Evidence / groundedness ──
    groundedness_fuzzy_threshold: int = 85
    groundedness_warn_below: float = 0.80
    groundedness_review_below: float = 0.50
    max_groundedness_penalty: float = 12.0

    # ── Evaluation ──
    consistency_runs: int = 3
    consistency_unstable_stdev: float = 6.0
    enable_critic: bool = True
    enable_judge: bool = False

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be within [0, 2]")
        return v

    def model_post_init(self, _context: object) -> None:
        """Fill in provider-appropriate model names when none were configured."""
        if not self.model:
            self.model = DEFAULT_MODELS.get(self.provider, "gpt-4o-mini")
        if not self.judge_model:
            self.judge_model = DEFAULT_JUDGE_MODELS.get(self.provider, self.model)

    @property
    def api_key(self) -> str | None:
        return {
            "openai": self.openai_api_key,
            "azure": self.azure_api_key,
            "gemini": self.gemini_api_key,
            "mock": "mock",
        }.get(self.provider)

    @property
    def gemini_api_keys(self) -> list[str]:
        """Primary + fallback Gemini keys, de-duplicated, primary first."""
        keys: list[str] = []
        for raw in (self.gemini_api_key, self.gemini_api_key_fallback):
            key = (raw or "").strip()
            if key and key not in keys:
                keys.append(key)
        return keys

    @property
    def gemini_fallback_models(self) -> tuple[str, ...]:
        """Models to try, in order, after the requested model is exhausted."""
        if not self.gemini_model_fallbacks.strip():
            return DEFAULT_GEMINI_MODEL_FALLBACKS
        return tuple(
            m.strip() for m in self.gemini_model_fallbacks.split(",") if m.strip()
        )

    def weights_for(self, level: str) -> dict[str, float]:
        return DIMENSION_WEIGHTS.get(level, DIMENSION_WEIGHTS["MID"])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings — used by tests that patch the environment."""
    get_settings.cache_clear()


def validate_weights() -> None:
    """Raise if a weight table does not sum to 1.0."""
    for level, table in DIMENSION_WEIGHTS.items():
        total = sum(table.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"dimension weights for {level} sum to {total}, expected 1.0"
            )
    missing = {
        level: set(DIMENSION_LABELS) - set(table)
        for level, table in DIMENSION_WEIGHTS.items()
        if set(DIMENSION_LABELS) - set(table)
    }
    if missing:
        raise ValueError(f"weight tables missing dimensions: {missing}")
