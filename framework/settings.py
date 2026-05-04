"""
Central configuration for Groq (OpenAI-compatible chat completions).
Loaded from environment; supports .env via load_dotenv at process entry.
"""
from __future__ import annotations
import os
from functools import lru_cache


DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_AGENT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_JUDGE_MODEL = "llama-3.3-70b-versatile"


@lru_cache
def groq_base_url() -> str:
    return (os.environ.get("GROQ_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


@lru_cache
def groq_agent_model() -> str:
    return os.environ.get("GROQ_AGENT_MODEL") or DEFAULT_AGENT_MODEL


@lru_cache
def groq_judge_model() -> str:
    return os.environ.get("GROQ_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL


def groq_api_key() -> str:
    key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not key:
        raise EnvironmentError("GROQ_API_KEY is not set in environment.")
    return key


def reload_settings() -> None:
    """Call after changing os.environ in tests."""
    groq_base_url.cache_clear()
    groq_agent_model.cache_clear()
    groq_judge_model.cache_clear()


def llm_max_retries() -> int:
    return max(1, int(os.environ.get("GROQ_MAX_RETRIES", "10")))


def llm_retry_base_seconds() -> float:
    return float(os.environ.get("GROQ_RETRY_BASE_SEC", "2.0"))


def llm_request_delay_seconds() -> float:
    """Optional pause before each Groq API call (agent + judge). Milliseconds env."""
    ms = float(os.environ.get("GROQ_REQUEST_DELAY_MS", "0"))
    return max(0.0, ms / 1000.0)
