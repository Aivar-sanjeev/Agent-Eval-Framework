"""
Shared chat.completions.create with retries on 429 / connection errors.
Used by the research agent and LLM-as-judge so both survive rate limits.
"""
from __future__ import annotations
import logging
import random
import time
from typing import Any

from openai import APIConnectionError, OpenAI, RateLimitError
from rich.console import Console

from framework.settings import llm_max_retries, llm_request_delay_seconds, llm_retry_base_seconds

logger = logging.getLogger(__name__)
_notify = Console(stderr=True, soft_wrap=True)


def chat_completions_create(
    client: OpenAI,
    *,
    notify_label: str = "Groq",
    **kwargs: Any,
) -> Any:
    """
    Call client.chat.completions.create with exponential backoff on 429.
    """
    delay_between = llm_request_delay_seconds()
    max_retries = llm_max_retries()
    base = llm_retry_base_seconds()

    response = None
    for attempt in range(max_retries):
        if delay_between > 0:
            time.sleep(delay_between)
        try:
            response = client.chat.completions.create(**kwargs)
            break
        except RateLimitError:
            if attempt >= max_retries - 1:
                raise
            backoff = base * (2**attempt) + random.uniform(0, 1.0)
            logger.warning(
                "%s rate limited (429); retry %s/%s in %.1fs",
                notify_label,
                attempt + 1,
                max_retries,
                backoff,
            )
            _notify.print(
                f"[dim yellow]{notify_label} 429 — backing off {backoff:.1f}s "
                f"(attempt {attempt + 2}/{max_retries})[/dim yellow]"
            )
            time.sleep(backoff)
        except APIConnectionError as e:
            if attempt >= max_retries - 1:
                raise
            backoff = base * (2**attempt) + random.uniform(0, 0.5)
            logger.warning(
                "%s connection error; retry %s/%s in %.1fs: %s",
                notify_label,
                attempt + 1,
                max_retries,
                backoff,
                e,
            )
            _notify.print(
                f"[dim yellow]{notify_label} connection error — retry in {backoff:.1f}s "
                f"({attempt + 2}/{max_retries})[/dim yellow]"
            )
            time.sleep(backoff)

    if response is None:
        raise RuntimeError(f"{notify_label}: chat completion failed after retries")
    return response
