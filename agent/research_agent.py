"""
research_agent.py — A multi-tool research agent powered by NVIDIA NIM.
Instruments every step with the framework tracer for full observability.

Tools available:
  • web_search   — simulated (returns realistic mock data)
  • calculator   — real expression evaluator (uses Python eval safely)
  • weather_lookup — simulated
  • unit_converter — real conversion
"""
from __future__ import annotations
import json
import math
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI

from framework.schema import (
    FinalAnswerSpan, InterpretationSpan, ReasoningSpan, ToolCallSpan, Trace,
)
from framework.tracer import Tracer
from framework.settings import nvidia_agent_model, nvidia_base_url



# ── Tool implementations ───────────────────────────────────────────────────────

MOCK_SEARCH_DB: Dict[str, str] = {
    "climate change": "Global temperatures have risen ~1.1°C since pre-industrial times. The IPCC warns of 1.5°C threshold risks. CO2 levels hit 421 ppm in 2023.",
    "python": "Python 3.12 was released in October 2023. It introduced faster startup, improved error messages, and new syntax for type parameters.",
    "nvidia": "NVIDIA reported Q3 FY2024 revenue of $18.1 billion, up 206% year-over-year, driven by AI chip demand. H100 GPU is the flagship data center product.",
    "llm": "Large Language Models (LLMs) are neural networks trained on vast text corpora. GPT-4, Claude, Gemini, and Llama are prominent examples. Transformer architecture is foundational.",
    "bitcoin": "Bitcoin reached an all-time high near $69,000 in November 2021. It operates on a proof-of-work blockchain with a 21M coin supply cap.",
    "default": "Search results: Found 847 results. Top sources indicate mixed evidence on the topic. Consensus leans toward moderate confidence in the primary claim.",
}

WEATHER_DB: Dict[str, Dict] = {
    "london": {"temp_c": 12, "condition": "Partly cloudy", "humidity": 78},
    "new york": {"temp_c": 22, "condition": "Sunny", "humidity": 55},
    "tokyo": {"temp_c": 18, "condition": "Overcast", "humidity": 65},
    "coimbatore": {"temp_c": 29, "condition": "Hot and humid", "humidity": 72},
    "default": {"temp_c": 20, "condition": "Clear", "humidity": 60},
}

UNIT_CONVERSIONS: Dict[Tuple[str, str], float] = {
    ("km", "miles"): 0.621371,
    ("miles", "km"): 1.60934,
    ("kg", "lbs"): 2.20462,
    ("lbs", "kg"): 0.453592,
    ("celsius", "fahrenheit"): None,   # special case
    ("fahrenheit", "celsius"): None,
    ("meters", "feet"): 3.28084,
    ("feet", "meters"): 0.3048,
}


def tool_web_search(query: str) -> str:
    q = query.lower()
    for key, result in MOCK_SEARCH_DB.items():
        if key in q:
            return result
    return MOCK_SEARCH_DB["default"]


def tool_calculator(expression: str) -> str:
    try:
        # Safe eval: only math operations
        safe_env = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, safe_env)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


def tool_weather_lookup(location: str) -> str:
    loc = location.lower()
    data = WEATHER_DB.get(loc, WEATHER_DB["default"])
    return (
        f"Weather in {location}: {data['condition']}, "
        f"{data['temp_c']}°C, Humidity {data['humidity']}%"
    )


def tool_unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    fu, tu = from_unit.lower(), to_unit.lower()
    if (fu, tu) == ("celsius", "fahrenheit"):
        return f"{value}°C = {value * 9/5 + 32:.2f}°F"
    if (fu, tu) == ("fahrenheit", "celsius"):
        return f"{value}°F = {(value - 32) * 5/9:.2f}°C"
    factor = UNIT_CONVERSIONS.get((fu, tu))
    if factor:
        return f"{value} {from_unit} = {value * factor:.4f} {to_unit}"
    return f"Conversion from {from_unit} to {to_unit} not supported."


TOOLS_REGISTRY = {
    "web_search": tool_web_search,
    "calculator": tool_calculator,
    "weather_lookup": tool_weather_lookup,
    "unit_converter": tool_unit_converter,
}

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information on a topic.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Math expression to evaluate, e.g. '2 ** 10' or 'sqrt(144)'"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather_lookup",
            "description": "Get current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": "Convert a value from one unit to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
]


# ── Agent class ────────────────────────────────────────────────────────────────

class ResearchAgent:
    """
    Multi-step research agent backed by NVIDIA NIM.
    Every reasoning step and tool call is captured in a structured Trace.
    """

    def __init__(self, version: str, system_prompt: Optional[str] = None, tracer: Optional[Tracer] = None):
        self.version = version
        self.tracer = tracer or Tracer()
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise EnvironmentError("NVIDIA_API_KEY not set")
        self.client = OpenAI(base_url=nvidia_base_url(), api_key=api_key)
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.available_tools = list(TOOLS_REGISTRY.keys())

    def _default_system_prompt(self) -> str:
        return (
            "You are a precise research assistant. Use the available tools to answer questions accurately.\n\n"
            "Tool routing (pick exactly one best tool per information need):\n"
            "• web_search — facts, news, company metrics, science, definitions, anything needing external knowledge.\n"
            "• calculator — numeric expressions, compound interest, unit math that is pure calculation.\n"
            "• weather_lookup — current weather for a named city/location only.\n"
            "• unit_converter — convert measurements (km↔miles, °C↔°F, kg↔lbs, etc.).\n\n"
            "If the question mixes topics (e.g. weather + conversion), handle each with the appropriate tool in sequence.\n"
            "Always reason briefly about which tool fits the user's intent. "
            "Ground your final answer in tool outputs; do not invent numbers not returned by tools."
        )

    def _call_llm(self, messages: List[Dict], use_tools: bool = True) -> Any:
        kwargs = dict(
            model=nvidia_agent_model(),
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        if use_tools:
            kwargs["tools"] = TOOL_SPECS
            kwargs["tool_choice"] = "auto"
        return self.client.chat.completions.create(**kwargs)

    def _execute_tool(self, tool_name: str, params: Dict) -> Any:
        fn = TOOLS_REGISTRY.get(tool_name)
        if not fn:
            return f"Error: tool '{tool_name}' not found."
        try:
            return fn(**params)
        except Exception as e:
            return f"Tool error: {e}"

    def run(self, query: str) -> Trace:
        """Execute the agent on a query and return a fully populated Trace."""
        trace = Trace(agent_version=self.version, query=query)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        step = 0
        max_steps = 8

        while step < max_steps:
            step += 1

            # ── LLM call ──
            response = self._call_llm(messages)
            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # ── Reasoning span ──
            reasoning_text = msg.content or ""
            tool_name_selected = None
            if msg.tool_calls:
                tool_name_selected = msg.tool_calls[0].function.name

            reasoning_span = ReasoningSpan(
                step_index=step,
                reasoning_text=reasoning_text,
                identified_intent=query[:100],
                selected_tool=tool_name_selected,
                available_tools=self.available_tools,
                raw_llm_output=reasoning_text,
            )
            trace.add_span(reasoning_span)

            # ── No tool call → final answer ──
            if finish_reason == "stop" or not msg.tool_calls:
                final_text = reasoning_text or "I was unable to determine an answer."
                final_span = FinalAnswerSpan(
                    answer_text=final_text,
                    total_steps=step,
                    total_tool_calls=sum(1 for s in trace.spans if s.get("span_type") == "tool_call"),
                )
                trace.add_span(final_span)
                trace.close(final_text)
                break

            # ── Tool calls ──
            messages.append({"role": "assistant", "content": reasoning_text, "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]})

            tool_results_for_interp = []

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    params = json.loads(tc.function.arguments)
                except Exception:
                    params = {}

                t0 = time.time()
                result = self._execute_tool(tool_name, params)
                latency = (time.time() - t0) * 1000

                tool_span = ToolCallSpan(
                    step_index=step,
                    tool_name=tool_name,
                    tool_parameters=params,
                    tool_result=result,
                    latency_ms=round(latency, 2),
                )
                trace.add_span(tool_span)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })
                tool_results_for_interp.append((tool_name, result))

            # ── Interpretation span — ask the model to reflect ──
            interp_prompt = messages + [{
                "role": "user",
                "content": (
                    "Briefly summarize what you just learned from the tool result(s). "
                    "Then state: do you need more information (continue), "
                    "call another tool (tool_call), or are you ready to give the final answer (final_answer)? "
                    "Reply in this JSON format only: "
                    "{\"summary\": \"...\", \"next_action\": \"continue|tool_call|final_answer\", \"reasoning\": \"...\"}"
                ),
            }]
            interp_response = self._call_llm(interp_prompt, use_tools=False)
            interp_raw = interp_response.choices[0].message.content or ""

            # Parse interpretation JSON
            try:
                clean = interp_raw.strip().lstrip("```json").rstrip("```").strip()
                interp_data = json.loads(clean)
            except Exception:
                interp_data = {
                    "summary": interp_raw[:200],
                    "next_action": "continue",
                    "reasoning": "",
                }

            next_action = interp_data.get("next_action", "continue")
            if next_action not in ("continue", "tool_call", "final_answer"):
                next_action = "continue"

            interp_span = InterpretationSpan(
                step_index=step,
                tool_result_summary=interp_data.get("summary", ""),
                next_action=next_action,
                reasoning_for_next=interp_data.get("reasoning", ""),
                raw_llm_output=interp_raw,
            )
            trace.add_span(interp_span)

            if next_action == "final_answer":
                # Trigger a final answer generation
                messages.append({
                    "role": "user",
                    "content": "Please now provide your final, complete answer to the original question.",
                })
                final_response = self._call_llm(messages, use_tools=False)
                final_text = final_response.choices[0].message.content or ""
                final_span = FinalAnswerSpan(
                    answer_text=final_text,
                    total_steps=step,
                    total_tool_calls=sum(1 for s in trace.spans if s.get("span_type") == "tool_call"),
                )
                trace.add_span(final_span)
                trace.close(final_text)
                break
        else:
            # Hit max steps
            trace.close("Max steps reached without final answer.")

        self.tracer.save_trace(trace)
        return trace
