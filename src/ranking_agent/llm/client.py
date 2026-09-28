"""Provider-agnostic LLM client (OpenAI, Azure, Gemini, mock)."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Any

import httpx

from ranking_agent.config import Settings, get_settings, omit_temperature
from ranking_agent.llm.cost import CostTracker, LLMUsage
from ranking_agent.llm.mock import mock_completion

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMError(RuntimeError):
    """Non-retryable provider failure."""


class LLMTransientError(RuntimeError):
    """Retryable provider failure (timeout, 429, 5xx)."""


def extract_json(raw: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Models wrap JSON in prose or fences often enough that treating that as a hard
    failure would burn a retry on a response that is actually usable.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty response")

    candidates: list[str] = []
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
    raise ValueError(f"no JSON object found in response: {text[:200]!r}")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class LLMClient:
    """Async JSON-mode client with retries, budget enforcement, and usage tally."""

    def __init__(
        self,
        settings: Settings | None = None,
        cost: CostTracker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cost = cost or CostTracker(budget_usd=self.settings.run_budget_usd)
        self._gemini_key_index = 0  # advanced on 429

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        task: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], LLMUsage]:
        """Call the model and return ``(parsed_json, usage)``.

        On 429/503 (quota/overload), rotates through Gemini API keys first;
        once a model's keys are exhausted (or the model itself 404s / is
        rejected outright), falls back to the next configured model — free
        models get pulled without notice and free-tier quota is per-model, so
        a different model often still has headroom.
        """
        self.cost.check_budget()
        primary = model or self.settings.model
        temp = self.settings.temperature if temperature is None else temperature
        if self.settings.provider == "gemini":
            models = [primary, *[m for m in self.settings.gemini_fallback_models if m != primary]]
        else:
            models = [primary]

        last_error: Exception | None = None
        for mdl in models:
            try:
                return await self._complete_with_model(system, user, task, mdl, temp)
            except (LLMError, LLMTransientError, ValueError, httpx.HTTPError) as exc:
                last_error = exc
                if mdl != models[-1]:
                    logger.warning(
                        "model=%s failed for task=%s (%s) — falling back to next model",
                        mdl, task, exc,
                    )

        raise LLMError(
            f"task={task} failed after exhausting models {models}: {last_error}"
        ) from last_error

    async def _complete_with_model(
        self, system: str, user: str, task: str, mdl: str, temp: float
    ) -> tuple[dict[str, Any], LLMUsage]:
        """Retry a single model across its configured Gemini keys."""
        self._gemini_key_index = 0
        last_error: Exception | None = None

        for attempt in range(1, self.settings.max_retries + 1):
            try:
                raw, usage = await self._dispatch(system, user, task, mdl, temp)
                self.cost.add(usage)
                return extract_json(raw), usage
            except LLMError:
                raise  # non-retryable for this model — let the caller try the next one
            except (LLMTransientError, ValueError, httpx.HTTPError) as exc:
                last_error = exc
                rotated = isinstance(exc, LLMTransientError) and "429" in str(exc) \
                    and self._rotate_gemini_key_on_quota()
                if attempt >= self.settings.max_retries:
                    raise
                delay = self.settings.retry_base_delay_s * (2 ** (attempt - 1))
                delay += random.uniform(0, delay * 0.25)
                if rotated:
                    delay = min(delay, 0.35)  # fresh key → short backoff
                logger.warning(
                    "llm attempt %d/%d failed for task=%s model=%s: %s (retry in %.2fs)",
                    attempt, self.settings.max_retries, task, mdl, exc, delay,
                )
                await asyncio.sleep(delay)

        raise LLMError(
            f"task={task} model={mdl} failed after {self.settings.max_retries} attempts: {last_error}"
        ) from last_error

    def _rotate_gemini_key_on_quota(self) -> bool:
        """Move to the next configured Gemini key. Returns True if switched."""
        if self.settings.provider != "gemini":
            return False
        keys = self.settings.gemini_api_keys
        if len(keys) < 2:
            return False
        prev = self._gemini_key_index % len(keys)
        self._gemini_key_index = (prev + 1) % len(keys)
        logger.warning(
            "gemini quota/429 — switching API key (%d → %d of %d)",
            prev + 1,
            self._gemini_key_index + 1,
            len(keys),
        )
        return True

    def _current_gemini_key(self) -> str:
        keys = self.settings.gemini_api_keys
        if not keys:
            raise LLMError(
                "gemini provider requires RANKING_GEMINI_API_KEY "
                "(optional RANKING_GEMINI_API_KEY_FALLBACK for quota failover)"
            )
        return keys[self._gemini_key_index % len(keys)]

    async def _dispatch(
        self, system: str, user: str, task: str, model: str, temperature: float
    ) -> tuple[str, LLMUsage]:
        provider = self.settings.provider
        if provider == "mock":
            return self._mock(task, user)
        if provider == "gemini":
            return await self._gemini(system, user, model, temperature)
        return await self._openai_compatible(system, user, model, temperature)

    def _mock(self, task: str, user: str) -> tuple[str, LLMUsage]:
        raw = mock_completion(task, user)
        return raw, LLMUsage.priced("mock", _estimate_tokens(user), _estimate_tokens(raw))

    async def _openai_compatible(
        self, system: str, user: str, model: str, temperature: float
    ) -> tuple[str, LLMUsage]:
        s = self.settings
        if s.provider == "azure":
            if not (s.azure_endpoint and s.azure_api_key):
                raise LLMError("azure provider requires endpoint and api key")
            url = (
                f"{s.azure_endpoint.rstrip('/')}/openai/deployments/{model}"
                f"/chat/completions?api-version={s.azure_api_version}"
            )
            headers = {"api-key": s.azure_api_key}
        else:
            if not s.openai_api_key:
                raise LLMError("openai provider requires RANKING_OPENAI_API_KEY")
            url = f"{s.openai_base_url.rstrip('/')}/chat/completions"
            headers = {"Authorization": f"Bearer {s.openai_api_key}"}

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": s.max_output_tokens,
            "response_format": {"type": "json_object"},
        }

        async with httpx.AsyncClient(timeout=s.request_timeout_s) as client:
            resp = await client.post(url, headers=headers, json=payload)
            self._raise_for_status(resp)
            data = resp.json()

        content = (data["choices"][0]["message"].get("content") or "").strip()
        usage_obj = data.get("usage") or {}
        return content, LLMUsage.priced(
            model,
            int(usage_obj.get("prompt_tokens") or _estimate_tokens(system + user)),
            int(usage_obj.get("completion_tokens") or _estimate_tokens(content)),
        )

    async def _gemini(
        self, system: str, user: str, model: str, temperature: float
    ) -> tuple[str, LLMUsage]:
        s = self.settings
        api_key = self._current_gemini_key()

        # API key in header (not query string).
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        headers = {"x-goog-api-key": api_key}

        generation_config: dict[str, Any] = {
            "maxOutputTokens": s.max_output_tokens,
            "responseMimeType": "application/json",
        }
        if not omit_temperature(model):
            generation_config["temperature"] = temperature

        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation_config,
        }
        async with httpx.AsyncClient(timeout=s.request_timeout_s) as client:
            resp = await client.post(url, headers=headers, json=payload)
            self._raise_for_status(resp)
            data = resp.json()

        candidates = data.get("candidates") or []
        if not candidates:
            # Safety block / empty candidates — non-retryable.
            block = (data.get("promptFeedback") or {}).get("blockReason")
            suffix = f" (blocked: {block})" if block else ""
            raise LLMError(f"gemini returned no candidates{suffix}")

        first = candidates[0]
        parts = (first.get("content") or {}).get("parts") or []
        content = "".join(p.get("text", "") for p in parts).strip()
        if not content and first.get("finishReason") == "MAX_TOKENS":
            raise LLMTransientError(
                "gemini hit MAX_TOKENS before emitting any content; "
                "raise RANKING_MAX_OUTPUT_TOKENS"
            )
        meta = data.get("usageMetadata") or {}
        return content, LLMUsage.priced(
            model,
            int(meta.get("promptTokenCount") or _estimate_tokens(system + user)),
            int(meta.get("candidatesTokenCount") or _estimate_tokens(content)),
        )

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if resp.status_code == 429 or resp.status_code >= 500:
            raise LLMTransientError(f"provider {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise LLMError(f"provider {resp.status_code}: {resp.text[:300]}")
