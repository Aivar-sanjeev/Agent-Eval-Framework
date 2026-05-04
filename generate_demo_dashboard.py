"""
generate_demo_dashboard.py
Populates realistic synthetic trace + eval data and renders the dashboard.
Run this WITHOUT an API key to preview the full dashboard.
"""
import sys, json, uuid, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from framework.schema import (
    Trace, ReasoningSpan, ToolCallSpan, InterpretationSpan,
    FinalAnswerSpan, EvalResult, Dataset, DatasetEntry,
)
from framework.tracer import Tracer
from framework.dataset import DatasetManager
from framework.dashboard import generate_dashboard

random.seed(42)
tracer = Tracer()
dm = DatasetManager()

QUERIES = [
    "What is the current CO2 level in the atmosphere and how has it changed?",
    "If I invest $5000 at 7% annual compound interest for 10 years, how much will I have?",
    "What's the weather in Tokyo and what is 25°C in Fahrenheit?",
    "How far is a marathon in miles, and how long to run at 6 min/km?",
    "What is NVIDIA's current revenue trend and AI demand outlook?",
]

TOOLS = ["web_search", "calculator", "weather_lookup", "unit_converter"]

# Realistic score profiles: v1 is weaker, v2 is improved
SCORE_PROFILES = {
    "v1.0": {
        "intent_clarity":         (0.72, 0.10),
        "tool_selection":         (0.68, 0.12),
        "pre_call_hallucination": (0.70, 0.10),
        "param_schema_validation":(0.88, 0.08),
        "param_value_quality":    (0.65, 0.12),
        "result_interpretation":  (0.67, 0.11),
        "next_step_decision":     (0.63, 0.13),
        "goal_completion":        (0.62, 0.14),
        "efficiency":             (0.55, 0.15),
        "final_hallucination":    (0.66, 0.12),
    },
    "v2.0": {
        "intent_clarity":         (0.89, 0.06),
        "tool_selection":         (0.87, 0.07),
        "pre_call_hallucination": (0.85, 0.07),
        "param_schema_validation":(0.97, 0.03),
        "param_value_quality":    (0.83, 0.08),
        "result_interpretation":  (0.84, 0.07),
        "next_step_decision":     (0.81, 0.09),
        "goal_completion":        (0.79, 0.10),
        "efficiency":             (0.76, 0.10),
        "final_hallucination":    (0.82, 0.08),
    },
}

EVAL_LAYERS = {
    "intent_clarity":          "pre_tool",
    "tool_selection":          "pre_tool",
    "pre_call_hallucination":  "pre_tool",
    "param_schema_validation": "post_tool",
    "param_value_quality":     "post_tool",
    "result_interpretation":   "post_tool",
    "next_step_decision":      "post_tool",
    "goal_completion":         "e2e",
    "efficiency":              "e2e",
    "final_hallucination":     "e2e",
}

EVAL_TYPES = {
    "param_schema_validation": "deterministic",
    "efficiency":              "deterministic",
}

def clamp(x): return max(0.0, min(1.0, x))

def make_trace(version, query):
    tool = random.choice(TOOLS)
    trace = Trace(agent_version=version, query=query)

    # Reasoning span
    rs = ReasoningSpan(
        step_index=1,
        reasoning_text=f"The user is asking about: {query[:60]}. I should use {tool}.",
        identified_intent=query[:80],
        selected_tool=tool,
        available_tools=TOOLS,
        raw_llm_output=f"I need to call {tool}."
    )
    trace.add_span(rs)

    # Tool call span
    params = {"query": query[:40]} if tool == "web_search" else {"expression": "5000 * 1.07**10"}
    tcs = ToolCallSpan(
        step_index=1,
        tool_name=tool,
        tool_parameters=params,
        tool_result="Result: 9835.76" if tool == "calculator" else "Found relevant information.",
        latency_ms=round(random.uniform(80, 350), 1),
    )
    trace.add_span(tcs)

    # Interpretation span
    ints = InterpretationSpan(
        step_index=1,
        tool_result_summary="Extracted the key data from the tool result.",
        next_action="final_answer",
        reasoning_for_next="I have enough information to answer.",
        raw_llm_output="I have the answer."
    )
    trace.add_span(ints)

    # Final answer
    fas = FinalAnswerSpan(
        answer_text=f"Based on the information retrieved: {query[:40]}... [answer]",
        total_steps=1,
        total_tool_calls=1,
    )
    trace.add_span(fas)
    trace.close(fas.answer_text)
    tracer.save_trace(trace)
    return trace

def make_eval_results(version, trace_id, query):
    profile = SCORE_PROFILES[version]
    results = []
    for eval_name, (mean, std) in profile.items():
        score = clamp(random.gauss(mean, std))
        threshold_map = {
            "intent_clarity": 0.80, "tool_selection": 0.80,
            "pre_call_hallucination": 0.80, "param_schema_validation": 0.95,
            "param_value_quality": 0.75, "result_interpretation": 0.75,
            "next_step_decision": 0.75, "goal_completion": 0.70,
            "efficiency": 0.65, "final_hallucination": 0.75,
        }
        threshold = threshold_map.get(eval_name, 0.75)
        results.append(EvalResult(
            eval_id=str(uuid.uuid4())[:8],
            eval_layer=EVAL_LAYERS[eval_name],
            eval_name=eval_name,
            trace_id=trace_id,
            passed=score >= threshold,
            score=round(score, 3),
            reason="Synthetic evaluation result for demo purposes.",
            eval_type=EVAL_TYPES.get(eval_name, "llm_judge"),
        ))
    return results

# Generate data for both versions
for version in ["v1.0", "v2.0"]:
    all_results = []
    entries = []
    for query in QUERIES:
        trace = make_trace(version, query)
        results = make_eval_results(version, trace.trace_id, query)
        tracer.save_eval_results(results)
        entries.append(DatasetEntry(
            trace_id=trace.trace_id,
            query=query,
            agent_version=version,
            golden_answer="[demo golden answer]",
            expected_tools=["web_search"],
            labels={"optimal_tool_calls": 1},
            eval_results=[r.model_dump() for r in results],
        ))
    dataset = Dataset(version=version, description=f"Demo dataset {version}", entries=entries)
    dm.save(dataset)

# Generate dashboard
path = generate_dashboard(versions=["v1.0", "v2.0"], tracer=tracer)
print(f"\nDashboard generated: {path}")
