"""
LLM Factory — uses LangChain's init_chat_model for easy provider switching.

Usage:
    from utils.llm_factory import get_llm

    config = {"llm_provider": "google", "llm_model": "gemini-2.5-flash", ...}
    llm = get_llm(config)

To switch providers, change config:
    config["llm_provider"] = "openai"
    config["llm_model"] = "gpt-4o"
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from config.default_config import LLM_CONFIG


# Map our config provider names to LangChain init_chat_model provider keys
_PROVIDER_MAP = {
    "openai": "openai",
    "google": "google_genai",  # Gemini API (GOOGLE_API_KEY)
    "anthropic": "anthropic",
}


def get_llm(
    config: Optional[Dict[str, Any]] = None,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    """
    Create a LangChain ChatModel using init_chat_model.

    Config keys:
        llm_provider: "openai" | "google" | "anthropic"
        llm_model: model name (e.g. "gpt-4o", "gemini-2.0-flash", "claude-3-5-sonnet-20241022")
        llm_temperature: 0.0–1.0
        llm_max_tokens: max output tokens
        llm_api_base: optional custom API base URL
        llm_api_key: optional override for API key (otherwise uses env)

    Environment variables (per provider):
        OpenAI: OPENAI_API_KEY
        Google: GOOGLE_API_KEY
        Anthropic: ANTHROPIC_API_KEY
    """
    from langchain.chat_models import init_chat_model
    cfg = config or {}
    prov = provider or cfg.get("llm_provider", "openai")
    preset = LLM_CONFIG.get(prov, LLM_CONFIG["google"])
    mdl = model or cfg.get("llm_model") or preset.get("model", "gpt-4o")
    temp = temperature if temperature is not None else cfg.get("llm_temperature", preset.get("temperature", 0.1))
    tokens = max_tokens if max_tokens is not None else cfg.get("llm_max_tokens", preset.get("max_tokens", 4000))
    base = api_base or cfg.get("llm_api_base") or preset.get("api_base")
    key = api_key or cfg.get("llm_api_key")

    # Map our provider names to LangChain's init_chat_model provider keys
    lc_provider = _PROVIDER_MAP.get(prov, "openai")

    kwargs: Dict[str, Any] = {
        "model": mdl,
        "model_provider": lc_provider,
        "temperature": temp,
        "max_tokens": tokens,
    }
    # Provider-specific params (init_chat_model passes **kwargs to underlying model)
    if lc_provider == "openai":
        if key:
            kwargs["api_key"] = key
        if base:
            kwargs["base_url"] = base.rstrip("/")
    elif lc_provider == "google_genai" and key:
        kwargs["google_api_key"] = key
    elif lc_provider == "anthropic" and key:
        kwargs["api_key"] = key
    if base and lc_provider == "anthropic":
        kwargs["anthropic_api_url"] = base.rstrip("/")

    return init_chat_model(**kwargs)


def get_available_providers() -> Dict[str, str]:
    """Return provider -> description for docs / CLI."""
    return {
        "openai": "OpenAI (GPT-4o, GPT-4, etc.). Set OPENAI_API_KEY.",
        "google": "Google (Gemini). Set GOOGLE_API_KEY. Requires: pip install langchain-google-genai",
        "anthropic": "Anthropic (Claude). Set ANTHROPIC_API_KEY. Requires: pip install langchain-anthropic",
    }
