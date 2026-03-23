"""
models/ollama_client.py
Local Ollama API client — unlimited inference with no rate limits.
Drop-in replacement for GroqClient.

Setup:
    1. Install Ollama: https://ollama.com/download
    2. Pull a model: ollama pull llama3.2:3b
    3. Set provider: "ollama" in configs/config.yaml
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from models.groq_client import LLMResponse

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Local Ollama API wrapper. Same interface as GroqClient.

    Usage:
        client = OllamaClient()
        resp = client.complete("llama3.2:3b", "What is 2+2?")
        print(resp.content)
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        try:
            import ollama as _ollama
            self._ollama = _ollama
            self._client = _ollama.Client(host=host, timeout=timeout)
        except ImportError:
            raise ImportError(
                "ollama package not installed. Run: pip install ollama\n"
                "Also install Ollama itself: https://ollama.com/download"
            )
        self._host = host
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
        keep_alive: Optional[str | int] = None,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            kwargs["format"] = "json"
        
        if keep_alive is not None:
            kwargs["keep_alive"] = keep_alive

        start = time.time()
        response = self._client.chat(**kwargs)
        latency_ms = (time.time() - start) * 1000

        prompt_tokens = response.get("prompt_eval_count", 0) or 0
        completion_tokens = response.get("eval_count", 0) or 0

        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_requests += 1

        content = response["message"]["content"]
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
        temperature: float = 0.05,
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
            "provider": "ollama",
            "host": self._host,
        }

    def is_model_available(self, model: str) -> bool:
        try:
            self._client.show(model)
            return True
        except Exception:
            return False


# ── Singleton factory ─────────────────────────────────────────

_ollama_instance: Optional[OllamaClient] = None


def get_ollama_client(cfg: Optional[dict] = None) -> OllamaClient:
    """Return a cached OllamaClient instance."""
    global _ollama_instance
    if _ollama_instance is None:
        params = cfg.get("ollama", {}) if cfg else {}
        _ollama_instance = OllamaClient(
            host=params.get("host", "http://localhost:11434"),
            timeout=params.get("timeout", 120.0),
        )
    return _ollama_instance
