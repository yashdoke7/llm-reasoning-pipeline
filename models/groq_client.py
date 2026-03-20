"""
models/groq_client.py
Unified Groq API client with exponential backoff, rate limiting,
structured output parsing, and token usage tracking.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from groq import Groq
from groq import RateLimitError, APIStatusError, APIConnectionError
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────

class ModelUnavailableError(Exception):
    """Raised when a model is not available on Groq (404)."""
    pass


class DailyLimitExhaustedError(Exception):
    """Raised when the Groq daily token quota (TPD) is exhausted."""
    pass


# ── Response models ───────────────────────────────────────────

class LLMResponse(BaseModel):
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float


@dataclass
class TokenBudget:
    """Simple sliding-window rate limiter for Groq free tier."""
    tpm_limit: int = 14400
    rpm_limit: int = 30
    _requests: list[float] = field(default_factory=list)
    _tokens: list[tuple[float, int]] = field(default_factory=list)
    _last_request: float = 0.0

    def _clean(self, window: float = 60.0) -> None:
        now = time.time()
        self._requests = [t for t in self._requests if now - t < window]
        self._tokens = [(t, n) for t, n in self._tokens if now - t < window]

    def wait_if_needed(self, estimated_tokens: int = 200) -> None:
        # Small gap to avoid burst — 0.3s = ~200 RPM effective (well under 300 limit)
        now = time.time()
        since_last = now - self._last_request
        min_gap = 0.3
        if since_last < min_gap:
            time.sleep(min_gap - since_last)

        self._clean()
        # Trigger at 80% of RPM limit
        rpm_threshold = int(self.rpm_limit * 0.8)
        if len(self._requests) >= rpm_threshold:
            sleep = 62 - (time.time() - self._requests[0])
            if sleep > 0:
                logger.info(f"Rate limit: sleeping {sleep:.1f}s (RPM {len(self._requests)}/{self.rpm_limit})")
                time.sleep(sleep)
            self._clean()
        # Trigger at 80% of TPM limit
        used = sum(n for _, n in self._tokens)
        tpm_threshold = int(self.tpm_limit * 0.8)
        if used + estimated_tokens >= tpm_threshold:
            if self._tokens:
                sleep = 62 - (time.time() - self._tokens[0][0])
                if sleep > 0:
                    logger.info(f"Rate limit: sleeping {sleep:.1f}s (TPM {used}/{self.tpm_limit})")
                    time.sleep(sleep)
            self._clean()

    def record(self, tokens: int) -> None:
        now = time.time()
        self._last_request = now
        self._requests.append(now)
        self._tokens.append((now, tokens))


# ── Client ────────────────────────────────────────────────────

class GroqClient:
    """
    Production-grade Groq API wrapper.

    Usage:
        client = GroqClient()
        resp = client.complete("llama-3.1-8b-instant", "What is 2+2?")
        print(resp.content)
        print(client.get_usage_summary())
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        tpm_limit: int = 120_000,
        rpm_limit: int = 300,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: float = 30.0,
    ) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Run: export GROQ_API_KEY=your_key"
            )
        self._client = Groq(api_key=key, timeout=timeout)
        self._budget = TokenBudget(tpm_limit=tpm_limit, rpm_limit=rpm_limit)
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        # Cumulative token usage tracking
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_requests = 0
        self._unavailable_models: set[str] = set()

    def complete(
        self,
        model: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Send a completion request. Blocks on rate limits, retries on errors.

        Args:
            model:          Groq model ID (e.g. "llama-3.1-8b-instant")
            user_prompt:    User message content
            system_prompt:  Optional system message
            temperature:    Sampling temperature (0.0-1.0)
            max_tokens:     Max tokens to generate
            json_mode:      If True, sets response_format to JSON

        Returns:
            LLMResponse with content and token usage

        Raises:
            ModelUnavailableError: If model returned 404 previously
        """
        if model in self._unavailable_models:
            raise ModelUnavailableError(f"Model '{model}' was previously unavailable (404). Skipping.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        estimated = len(user_prompt.split()) * 1.3 + max_tokens
        self._budget.wait_if_needed(int(estimated))

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.time()
        response = self._complete_with_retry(**kwargs)
        latency_ms = (time.time() - start) * 1000

        usage = response.usage
        self._budget.record(usage.total_tokens)
        self._total_prompt_tokens += usage.prompt_tokens
        self._total_completion_tokens += usage.completion_tokens
        self._total_requests += 1

        logger.debug(
            f"[{model}] tokens={usage.total_tokens} latency={latency_ms:.0f}ms"
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=latency_ms,
        )

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _complete_with_retry(self, **kwargs: Any) -> Any:
        try:
            return self._client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            # Detect daily token limit (TPD) — no point retrying for hours
            err_msg = str(e).lower()
            if "tokens per day" in err_msg or "tpd" in err_msg:
                logger.error(
                    "Daily token limit (TPD) exhausted on Groq free tier. "
                    "Stopping — retry tomorrow or upgrade at console.groq.com/settings/billing"
                )
                raise DailyLimitExhaustedError(
                    "Groq daily token quota (500K TPD) exhausted. "
                    "Wait until reset or upgrade to Dev Tier."
                ) from e
            logger.warning("Groq rate limit hit — backing off")
            raise
        except APIConnectionError:
            logger.warning("Groq connection error — retrying")
            raise
        except APIStatusError as e:
            if e.status_code == 404:
                model = kwargs.get("model", "unknown")
                self._unavailable_models.add(model)
                raise ModelUnavailableError(
                    f"Model '{model}' not found on Groq (404). "
                    "It may have been deprecated. Check available models at console.groq.com"
                )
            logger.error(f"Groq API error {e.status_code}: {e.message}")
            raise

    def complete_json(
        self,
        model: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.05,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """
        Convenience wrapper that returns parsed JSON dict.
        Always uses json_mode=True and parses the response.
        """
        resp = self.complete(
            model=model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        try:
            return json.loads(resp.content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}\nRaw: {resp.content[:200]}")
            # Attempt recovery: strip markdown code fences
            cleaned = resp.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(cleaned)

    def get_usage_summary(self) -> dict:
        """Return cumulative token usage and estimated cost."""
        return {
            "total_requests": self._total_requests,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "unavailable_models": list(self._unavailable_models),
        }

    def is_model_available(self, model: str) -> bool:
        """Check if a model has been marked as unavailable."""
        return model not in self._unavailable_models


# ── Singleton factory ─────────────────────────────────────────

_client_instance: Optional[GroqClient] = None


def get_client(cfg: Optional[dict] = None) -> GroqClient:
    """Return a cached GroqClient instance."""
    global _client_instance
    if _client_instance is None:
        params = cfg.get("groq", {}) if cfg else {}
        _client_instance = GroqClient(
            tpm_limit=params.get("tpm_limit", 120_000),
            rpm_limit=params.get("rpm_limit", 300),
            max_retries=params.get("max_retries", 5),
            base_delay=params.get("base_delay", 1.0),
            max_delay=params.get("max_delay", 60.0),
            timeout=params.get("timeout", 30),
        )
    return _client_instance
