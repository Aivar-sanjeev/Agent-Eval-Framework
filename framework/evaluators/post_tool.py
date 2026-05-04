"""
post_tool.py — Layer 2 evaluators: run AT and AFTER a tool call.
Evaluates parameter correctness, result interpretation, and next-step decisions.
"""
from __future__ import annotations
import json
import uuid
from typing import Any, Dict, List, Optional
from framework.schema import EvalResult, ToolCallSpan, InterpretationSpan
from framework.llm_judge import judge


# ── Eval 4: Tool Parameter Schema Validation (deterministic) ─────────────────

TOOL_SCHEMAS: Dict[str, Dict] = {
    "web_search": {
        "required": ["query"],
        "types": {"query": str},
    },
    "calculator": {
        "required": ["expression"],
        "types": {"expression": str},
    },
    "weather_lookup": {
        "required": ["location"],
        "types": {"location": str},
    },
    "unit_converter": {
        "required": ["value", "from_unit", "to_unit"],
        "types": {"value": (int, float), "from_unit": str, "to_unit": str},
    },
}


def eval_param_schema(
    trace_id: str,
    span: ToolCallSpan,
) -> EvalResult:
    """
    Deterministic check: do tool parameters match the expected schema?
    """
    schema = TOOL_SCHEMAS.get(span.tool_name)
    if schema is None:
        # Unknown tool — pass with note
        return EvalResult(
            eval_id=str(uuid.uuid4())[:8],
            eval_layer="post_tool",
            eval_name="param_schema_validation",
            trace_id=trace_id,
            span_id=span.span_id,
            passed=True,
            score=1.0,
            reason=f"No schema defined for tool '{span.tool_name}' — skipped.",
            eval_type="deterministic",
        )

    params = span.tool_parameters
    missing = [k for k in schema["required"] if k not in params]
    type_errors = []
    for field, expected_type in schema.get("types", {}).items():
        if field in params and not isinstance(params[field], expected_type):
            type_errors.append(
                f"'{field}' should be {expected_type.__name__ if hasattr(expected_type, '__name__') else expected_type}, "
                f"got {type(params[field]).__name__}"
            )

    passed = not missing and not type_errors
    if passed:
        reason = "All required parameters present with correct types."
        score = 1.0
    else:
        parts = []
        if missing:
            parts.append(f"Missing fields: {missing}")
        if type_errors:
            parts.append(f"Type errors: {type_errors}")
        reason = "; ".join(parts)
        score = 0.0

    return EvalResult(
        eval_id=str(uuid.uuid4())[:8],
        eval_layer="post_tool",
        eval_name="param_schema_validation",
        trace_id=trace_id,
        span_id=span.span_id,
        passed=passed,
        score=score,
        reason=reason,
        eval_type="deterministic",
    )


# ── Eval 5: Tool Parameter Value Quality (LLM-judge) ─────────────────────────

def eval_param_quality(
    trace_id: str,
    span: ToolCallSpan,
    query: str,
) -> EvalResult:
    """
    Are the parameter values sensible and precise given the user's query?
    E.g. a web_search with query="foo" for a question about climate change is bad.
    """
    result = judge(
        system_prompt=(
            "You are an expert evaluator judging the quality of tool call parameters "
            "made by an AI agent. Focus on whether the parameter VALUES are appropriate "
            "and well-formed for the user's actual question."
        ),
        user_prompt=(
            f"User query: {query}\n"
            f"Tool called: {span.tool_name}\n"
            f"Parameters used: {json.dumps(span.tool_parameters, indent=2)}\n\n"
            "Are the parameter values precise, relevant, and likely to produce a useful result? "
            "Consider: specificity, accuracy, alignment with the query.\n"
            "Return JSON: {\"passed\": bool, \"score\": 0.0-1.0, \"reason\": \"one sentence\"}"
        ),
    )
    return EvalResult(
        eval_id=str(uuid.uuid4())[:8],
        eval_layer="post_tool",
        eval_name="param_value_quality",
        trace_id=trace_id,
        span_id=span.span_id,
        passed=bool(result.get("passed", False)),
        score=float(result.get("score", 0.0)),
        reason=result.get("reason", "judge parse error"),
        eval_type="llm_judge",
    )


# ── Eval 6: Tool Result Interpretation (LLM-judge) ───────────────────────────

def eval_result_interpretation(
    trace_id: str,
    tool_span: ToolCallSpan,
    interp_span: InterpretationSpan,
    query: str,
) -> EvalResult:
    """
    Did the agent correctly interpret the tool result?
    """
    tool_result_str = str(tool_span.tool_result)[:1000]  # truncate for prompt
    result = judge(
        system_prompt=(
            "You are an expert evaluator. Assess whether an AI agent correctly "
            "understood and summarized a tool's output."
        ),
        user_prompt=(
            f"User query: {query}\n"
            f"Tool used: {tool_span.tool_name}\n"
            f"Raw tool result: {tool_result_str}\n\n"
            f"Agent's interpretation: {interp_span.tool_result_summary}\n\n"
            "Did the agent correctly extract the key information from the tool result? "
            "Did it miss important details or misinterpret the data?\n"
            "Return JSON: {\"passed\": bool, \"score\": 0.0-1.0, \"reason\": \"one sentence\"}"
        ),
    )
    return EvalResult(
        eval_id=str(uuid.uuid4())[:8],
        eval_layer="post_tool",
        eval_name="result_interpretation",
        trace_id=trace_id,
        span_id=interp_span.span_id,
        passed=bool(result.get("passed", False)),
        score=float(result.get("score", 0.0)),
        reason=result.get("reason", "judge parse error"),
        eval_type="llm_judge",
    )


# ── Eval 7: Next-Step Decision Quality (LLM-judge) ───────────────────────────

def eval_next_step_decision(
    trace_id: str,
    interp_span: InterpretationSpan,
    query: str,
    step_index: int,
    total_steps: int,
) -> EvalResult:
    """
    Was the agent's decision on what to do next (continue, call another tool, or answer) correct?
    """
    result = judge(
        system_prompt=(
            "You are an expert evaluator judging AI agent decision-making. "
            "Assess whether the agent made the right decision about what to do next "
            "after receiving a tool result."
        ),
        user_prompt=(
            f"User query: {query}\n"
            f"Step {step_index} of ~{total_steps} total steps.\n"
            f"Agent's interpretation of result: {interp_span.tool_result_summary}\n"
            f"Agent decided to: {interp_span.next_action}\n"
            f"Reasoning: {interp_span.reasoning_for_next}\n\n"
            "Was this the right next action? Should the agent have stopped, continued, "
            "or taken a different approach?\n"
            "Return JSON: {\"passed\": bool, \"score\": 0.0-1.0, \"reason\": \"one sentence\"}"
        ),
    )
    return EvalResult(
        eval_id=str(uuid.uuid4())[:8],
        eval_layer="post_tool",
        eval_name="next_step_decision",
        trace_id=trace_id,
        span_id=interp_span.span_id,
        passed=bool(result.get("passed", False)),
        score=float(result.get("score", 0.0)),
        reason=result.get("reason", "judge parse error"),
        eval_type="llm_judge",
    )


# ── Runner: collect all post-tool evals for a trace ──────────────────────────

def run_post_tool_evals(
    trace_id: str,
    query: str,
    spans: List[Dict],
) -> List[EvalResult]:
    results = []

    # Index spans by step for pairing tool_call + interpretation
    tool_spans: Dict[int, ToolCallSpan] = {}
    interp_spans: Dict[int, InterpretationSpan] = {}

    for s in spans:
        if s.get("span_type") == "tool_call":
            ts = ToolCallSpan(**s)
            tool_spans[ts.step_index] = ts
        elif s.get("span_type") == "interpretation":
            ints = InterpretationSpan(**s)
            interp_spans[ints.step_index] = ints

    total_steps = max(
        list(tool_spans.keys()) + list(interp_spans.keys()) + [0]
    )

    for step, tool_span in tool_spans.items():
        # Deterministic schema check
        results.append(eval_param_schema(trace_id, tool_span))
        # LLM param quality check
        results.append(eval_param_quality(trace_id, tool_span, query))

        # Pair with interpretation span if available
        if step in interp_spans:
            interp_span = interp_spans[step]
            results.append(eval_result_interpretation(trace_id, tool_span, interp_span, query))
            results.append(eval_next_step_decision(trace_id, interp_span, query, step, total_steps))

    return results
