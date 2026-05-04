"""
llm_judge.py — Groq-powered LLM-as-judge (OpenAI-compatible API).
All evaluators that need a model call go through this module.
"""
from __future__ import annotations
import json

from openai import OpenAI

from framework.openai_retry import chat_completions_create
from framework.settings import groq_api_key, groq_base_url, groq_judge_model


def get_client() -> OpenAI:
    return OpenAI(base_url=groq_base_url(), api_key=groq_api_key())


def judge(
    system_prompt: str,
    user_prompt: str,
    expect_json: bool = True,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> dict | str:
    """
    Call the judge model and return parsed JSON (or raw string).
    """
    client = get_client()

    system = system_prompt
    if expect_json:
        system += (
            "\n\nIMPORTANT: Respond ONLY with a valid JSON object. "
            "No preamble, no markdown fences, no explanation outside the JSON."
        )

    response = chat_completions_create(
        client,
        notify_label="Groq judge",
        model=groq_judge_model(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    raw = response.choices[0].message.content.strip()

    if not expect_json:
        return raw

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except Exception:
                pass
        return {"error": "parse_failed", "raw": raw, "passed": False, "score": 0.0, "reason": raw}
