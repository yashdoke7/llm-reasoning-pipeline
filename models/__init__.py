"""
models/__init__.py
Unified client factory — returns Groq, Ollama, or OpenAI client based on provider config.

Providers:
    "groq"   — Groq cloud API (fast, free tier, rate-limited)
    "ollama" — Local Ollama (free, unlimited, requires GPU/CPU)
    "openai" — OpenAI API (paid, use as judge for objective evaluation)

Solver vs Judge pattern:
    solver_client = get_client(cfg, provider="ollama")   # generates reasoning
    judge_client  = get_client(cfg, provider="openai")   # grades each step
"""
from models.groq_client import get_client as _get_groq_client, LLMResponse


def get_client(cfg=None, provider=None):
    """
    Factory: returns a GroqClient, OllamaClient, or OpenAIClient.

    Args:
        cfg:      Full config dict (reads cfg["provider"] if provider not given)
        provider: "groq" | "ollama" | "openai" (overrides config)
    """
    if provider is None and cfg:
        provider = cfg.get("provider", "groq")
    provider = provider or "groq"

    if provider == "ollama":
        from models.ollama_client import get_ollama_client
        return get_ollama_client(cfg)

    if provider == "openai":
        from models.openai_client import get_openai_client
        return get_openai_client(cfg)

    return _get_groq_client(cfg)