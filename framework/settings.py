"""
Central configuration for NVIDIA NIM (OpenAI-compatible) endpoints.
Loaded from environment; supports .env via load_dotenv() at process entry.
"""
from __future__ import annotations
import os
from functools import lru_cache


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_AGENT_MODEL = "meta/llama-3.3-70b-instruct"
DEFAULT_JUDGE_MODEL = "meta/llama-3.3-70b-instruct"


@lru_cache
def nvidia_base_url() -> str:
    return (os.environ.get("NVIDIA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


@lru_cache
def nvidia_agent_model() -> str:
    return os.environ.get("NVIDIA_AGENT_MODEL") or DEFAULT_AGENT_MODEL


@lru_cache
def nvidia_judge_model() -> str:
    return os.environ.get("NVIDIA_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL


def reload_settings() -> None:
    """Call after changing os.environ in tests."""
    nvidia_base_url.cache_clear()
    nvidia_agent_model.cache_clear()
    nvidia_judge_model.cache_clear()
