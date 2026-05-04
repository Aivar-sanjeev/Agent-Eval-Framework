"""
llm_judge.py — NVIDIA NIM-powered LLM-as-judge.
All evaluators that need a model call go through this module.
"""
from __future__ import annotations
import json
import os
from typing import Optional
from openai import OpenAI

from framework.settings import nvidia_base_url, nvidia_judge_model


def get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY is not set in environment.")
    return OpenAI(base_url=nvidia_base_url(), api_key=api_key)


def judge(
    system_prompt: str,
    user_prompt: str,
    expect_json: bool = True,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> dict | str:
    """
    Call the NIM judge model and return parsed JSON (or raw string).
    Always instructs the model to respond in JSON when expect_json=True.
    """
    client = get_client()

    system = system_prompt
    if expect_json:
        system += (
            "\n\nIMPORTANT: Respond ONLY with a valid JSON object. "
            "No preamble, no markdown fences, no explanation outside the JSON."
        )

    response = client.chat.completions.create(
        model=nvidia_judge_model(),
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

    # Strip markdown fences if model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to extract first {...} block
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except Exception:
                pass
        return {"error": "parse_failed", "raw": raw, "passed": False, "score": 0.0, "reason": raw}
