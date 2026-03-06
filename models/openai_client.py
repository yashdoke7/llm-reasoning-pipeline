"""
models/openai_client.py
OpenAI API client — drop-in replacement for GroqClient/OllamaClient.

Primarily used as an independent JUDGE for step-level evaluation,
not as the solver. This separates solver and judge to eliminate
the self-evaluation bias.

Setup:
    1. Get API key: https://platform.openai.com/api-keys
    2. Add to .env: OPENAI_API_KEY=sk-...
    3. Use as judge: --judge-provider openai --judge-model gpt-4o-mini

Cost estimate (gpt-4o-mini as judge):
    ~$0.15/1M input tokens, ~$0.60/1M output tokens
    33 tasks × ~600 tokens = ~20K tokens = ~$0.003 total
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

from models.groq_client import LLMResponse

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    OpenAI API wrapper. Same interface as GroqClient/OllamaClient.

    Usage:
        client = OpenAIClient()
        resp = client.complete("gpt-4o-mini", "What is 2+2?")
        print(resp.content)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY not set. Add to .env or export OPENAI_API_KEY=sk-..."
            )

        from openai import OpenAI
        self._client = OpenAI(api_key=key, timeout=timeout)
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_requests = 0

    def complete(
        self,
        model: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.time()
        response = self._client.chat.completions.create(**kwargs)
        latency_ms = (time.time() - start) * 1000

        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0

        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_requests += 1

        content = response.choices[0].message.content or ""
        logger.debug(
            f"[{model}] tokens={prompt_tokens + completion_tokens} "
            f"latency={latency_ms:.0f}ms"
        )

        return LLMResponse(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
        )

    def complete_json(
        self,
        model: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
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
            cleaned = (
                resp.content.strip()
                .lstrip("```json").lstrip("```").rstrip("```")
                .strip()
            )
            return json.loads(cleaned)

    def get_usage_summary(self) -> dict:
        return {
            "total_requests": self._total_requests,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "provider": "openai",
        }


# ── Singleton factory ─────────────────────────────────────────

_openai_instance: Optional[OpenAIClient] = None


def get_openai_client(cfg: Optional[dict] = None) -> OpenAIClient:
    """Return a cached OpenAIClient instance."""
    global _openai_instance
    if _openai_instance is None:
        params = cfg.get("openai", {}) if cfg else {}
        _openai_instance = OpenAIClient(
            api_key=params.get("api_key") or os.environ.get("OPENAI_API_KEY"),
            timeout=params.get("timeout", 60.0),
        )
    return _openai_instance
