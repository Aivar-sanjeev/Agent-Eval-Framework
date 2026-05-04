"""
pre_tool.py — Layer 1 evaluators: run BEFORE a tool call.
Evaluates reasoning quality and tool selection.
"""
from __future__ import annotations
import uuid
from typing import Any, Dict, List
from framework.schema import EvalResult, ReasoningSpan
from framework.llm_judge import judge


# ── Eval 1: Intent Clarity (LLM-judge) ────────────────────────────────────────

def eval_intent_clarity(
    trace_id: str,
    span: ReasoningSpan,
    query: str,
) -> EvalResult:
    """
    Does the agent's reasoning correctly identify the user's intent?
    Score: 0.0 = completely wrong intent, 1.0 = perfectly identified.
    """
    result = judge(
        system_prompt=(
            "You are an expert AI evaluation judge. "
            "Your job is to assess whether an AI agent correctly identified "
            "the user's intent from their query before making a tool call."
        ),
        user_prompt=(
            f"User query: {query}\n\n"
            f"Agent's reasoning before tool call:\n{span.reasoning_text}\n\n"
            f"Agent identified intent as: {span.identified_intent or 'not stated'}\n\n"
            "Evaluate: Did the agent correctly understand what the user wants?\n"
            "Return JSON: {\"passed\": bool, \"score\": 0.0-1.0, \"reason\": \"one sentence\"}"
        ),
    )
    return EvalResult(
        eval_id=str(uuid.uuid4())[:8],
        eval_layer="pre_tool",
        eval_name="intent_clarity",
        trace_id=trace_id,
        span_id=span.span_id,
        passed=bool(result.get("passed", False)),
        score=float(result.get("score", 0.0)),
        reason=result.get("reason", "judge parse error"),
        eval_type="llm_judge",
    )


# ── Eval 2: Tool Selection Appropriateness (LLM-judge) ────────────────────────

def eval_tool_selection(
    trace_id: str,
    span: ReasoningSpan,
    query: str,
) -> EvalResult:
    """
    Did the agent pick the right tool given the available options and query?
    """
    tools_str = ", ".join(span.available_tools) if span.available_tools else "unknown"
    result = judge(
        system_prompt=(
            "You are an expert AI evaluation judge specialising in tool-use agents. "
            "Evaluate whether the agent chose the most appropriate tool."
        ),
        user_prompt=(
            f"User query: {query}\n\n"
            f"Available tools: {tools_str}\n"
            f"Agent selected tool: {span.selected_tool or 'none'}\n"
            f"Agent's reasoning: {span.reasoning_text}\n\n"
            "Was this the right tool choice? Consider: is the tool capable of answering "
            "the query? Is there a clearly better tool available?\n"
            "Return JSON: {\"passed\": bool, \"score\": 0.0-1.0, \"reason\": \"one sentence\"}"
        ),
    )
    return EvalResult(
        eval_id=str(uuid.uuid4())[:8],
        eval_layer="pre_tool",
        eval_name="tool_selection",
        trace_id=trace_id,
        span_id=span.span_id,
        passed=bool(result.get("passed", False)),
        score=float(result.get("score", 0.0)),
        reason=result.get("reason", "judge parse error"),
        eval_type="llm_judge",
    )


# ── Eval 3: Pre-call Hallucination Check (LLM-judge) ─────────────────────────

def eval_pre_call_hallucination(
    trace_id: str,
    span: ReasoningSpan,
    query: str,
) -> EvalResult:
    """
    Is the agent inventing facts or constraints that aren't in the user's query?
    score=1.0 means NO hallucination (good), score=0.0 means hallucinating.
    """
    result = judge(
        system_prompt=(
            "You are an expert evaluator checking for hallucination in AI agent reasoning. "
            "Hallucination here means the agent assumes facts, constraints, or context "
            "that are NOT present in the user's query."
        ),
        user_prompt=(
            f"User query: {query}\n\n"
            f"Agent's reasoning before tool call:\n{span.reasoning_text}\n\n"
            "Does the agent assume anything that isn't grounded in the user's query? "
            "Score 1.0 = no hallucination (clean reasoning), 0.0 = significant hallucination.\n"
            "Return JSON: {\"passed\": bool, \"score\": 0.0-1.0, \"reason\": \"one sentence\"}"
        ),
    )
    return EvalResult(
        eval_id=str(uuid.uuid4())[:8],
        eval_layer="pre_tool",
        eval_name="pre_call_hallucination",
        trace_id=trace_id,
        span_id=span.span_id,
        passed=bool(result.get("passed", False)),
        score=float(result.get("score", 0.0)),
        reason=result.get("reason", "judge parse error"),
        eval_type="llm_judge",
    )


# ── Runner: collect all pre-tool evals for a trace ───────────────────────────

def run_pre_tool_evals(trace_id: str, query: str, spans: List[Dict]) -> List[EvalResult]:
    results = []
    for span_dict in spans:
        if span_dict.get("span_type") != "reasoning":
            continue
        span = ReasoningSpan(**span_dict)
        results.append(eval_intent_clarity(trace_id, span, query))
        if span.selected_tool:
            results.append(eval_tool_selection(trace_id, span, query))
        results.append(eval_pre_call_hallucination(trace_id, span, query))
    return results
