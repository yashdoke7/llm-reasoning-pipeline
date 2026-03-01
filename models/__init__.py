"""
models/__init__.py
Unified client factory — returns Groq or Ollama client based on provider config.
"""
from models.groq_client import get_client as _get_groq_client, LLMResponse


def get_client(cfg=None, provider=None):
    """
    Factory: returns a GroqClient or OllamaClient based on provider.

    Args:
        cfg:      Full config dict (reads cfg["provider"] if provider not given)
        provider: "groq" or "ollama" (overrides config)
    """
    if provider is None and cfg:
        provider = cfg.get("provider", "groq")
    provider = provider or "groq"

    if provider == "ollama":
        from models.ollama_client import get_ollama_client
        return get_ollama_client(cfg)
    return _get_groq_client(cfg)