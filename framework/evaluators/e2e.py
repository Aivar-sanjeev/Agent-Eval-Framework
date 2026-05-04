"""
e2e.py — Layer 3 evaluators: end-to-end quality of the full agent run.
Evaluates goal completion, efficiency, and hallucination in final answer.
"""
from __future__ import annotations
import uuid
from typing import Dict, List, Optional
from framework.schema import EvalResult, Trace
from framework.llm_judge import judge


# ── Eval 8: Goal Completion (LLM-judge) ──────────────────────────────────────

def eval_goal_completion(
    trace: Trace,
    golden_answer: Optional[str] = None,
) -> EvalResult:
    """
    Did the agent fully accomplish what the user asked?
    If a golden answer is provided, comparison is made against it.
    Otherwise, the judge evaluates on reasonableness.
    """
    golden_part = ""
    if golden_answer:
        golden_part = f"\nExpected answer (golden): {golden_answer}\n"

    result = judge(
        system_prompt=(
            "You are an expert evaluator assessing whether an AI agent successfully "
            "completed the user's request. Consider: Does the final answer address "
            "the query? Is it complete? Is it accurate?"
        ),
        user_prompt=(
            f"User query: {trace.query}\n"
            f"{golden_part}"
            f"Agent's final answer:\n{trace.final_answer}\n\n"
            f"The agent used {sum(1 for s in trace.spans if s.get('span_type') == 'tool_call')} tool calls "
            f"across {len(trace.spans)} total spans.\n\n"
            "Did the agent fully and correctly answer the user's query?\n"
            "Return JSON: {\"passed\": bool, \"score\": 0.0-1.0, \"reason\": \"one sentence\"}"
        ),
    )
    return EvalResult(
        eval_id=str(uuid.uuid4())[:8],
        eval_layer="e2e",
        eval_name="goal_completion",
        trace_id=trace.trace_id,
        passed=bool(result.get("passed", False)),
        score=float(result.get("score", 0.0)),
        reason=result.get("reason", "judge parse error"),
        eval_type="llm_judge",
    )


# ── Eval 9: Efficiency (deterministic) ───────────────────────────────────────

def eval_efficiency(
    trace: Trace,
    optimal_tool_calls: Optional[int] = None,
) -> EvalResult:
    """
    How efficient was the agent? Penalizes unnecessary tool calls.
    optimal_tool_calls: the minimum number of tool calls needed (from dataset label).
    If not provided, uses a heuristic (≤3 tool calls = efficient for most queries).
    """
    tool_call_count = sum(
        1 for s in trace.spans if s.get("span_type") == "tool_call"
    )
    optimal = optimal_tool_calls or 2  # default heuristic

    if tool_call_count == 0:
        score = 0.0
        passed = False
        reason = "Agent made no tool calls — cannot have solved the task."
    elif tool_call_count <= optimal:
        score = 1.0
        passed = True
        reason = f"Used {tool_call_count} tool calls (optimal ≤ {optimal})."
    elif tool_call_count <= optimal + 1:
        score = 0.7
        passed = True
        reason = f"Used {tool_call_count} tool calls, 1 extra beyond optimal {optimal}."
    elif tool_call_count <= optimal + 2:
        score = 0.4
        passed = False
        reason = f"Used {tool_call_count} tool calls, {tool_call_count - optimal} extra beyond optimal {optimal}."
    else:
        score = 0.1
        passed = False
        reason = f"Used {tool_call_count} tool calls, far exceeding optimal {optimal} — likely looping."

    return EvalResult(
        eval_id=str(uuid.uuid4())[:8],
        eval_layer="e2e",
        eval_name="efficiency",
        trace_id=trace.trace_id,
        passed=passed,
        score=score,
        reason=reason,
        eval_type="deterministic",
        metadata={"tool_call_count": tool_call_count, "optimal": optimal},
    )


# ── Eval 10: Final Answer Hallucination (LLM-judge) ──────────────────────────

def eval_final_hallucination(trace: Trace) -> EvalResult:
    """
    Does the final answer contain claims not supported by tool results?
    score=1.0 = no hallucination (grounded), score=0.0 = fabricated content.
    """
    # Collect all tool results as grounding context
    tool_results = []
    for s in trace.spans:
        if s.get("span_type") == "tool_call" and s.get("tool_result"):
            tool_results.append(
                f"[{s['tool_name']}] → {str(s['tool_result'])[:500]}"
            )

    grounding = "\n".join(tool_results) if tool_results else "No tool results available."

    result = judge(
        system_prompt=(
            "You are an expert hallucination detector for AI agents. "
            "Your job is to check whether the agent's final answer contains claims "
            "that are NOT supported by the tool results it received. "
            "Focus on factual claims, numbers, names, and statements of fact."
        ),
        user_prompt=(
            f"User query: {trace.query}\n\n"
            f"Tool results (grounding evidence):\n{grounding}\n\n"
            f"Agent's final answer:\n{trace.final_answer}\n\n"
            "Are there any factual claims in the final answer that cannot be "
            "traced back to the tool results? "
            "Score 1.0 = fully grounded (good), 0.0 = heavily hallucinated.\n"
            "Return JSON: {\"passed\": bool, \"score\": 0.0-1.0, \"reason\": \"one sentence\"}"
        ),
    )
    return EvalResult(
        eval_id=str(uuid.uuid4())[:8],
        eval_layer="e2e",
        eval_name="final_hallucination",
        trace_id=trace.trace_id,
        passed=bool(result.get("passed", False)),
        score=float(result.get("score", 0.0)),
        reason=result.get("reason", "judge parse error"),
        eval_type="llm_judge",
    )


# ── Runner: collect all e2e evals for a trace ────────────────────────────────

def run_e2e_evals(
    trace: Trace,
    golden_answer: Optional[str] = None,
    optimal_tool_calls: Optional[int] = None,
) -> List[EvalResult]:
    return [
        eval_goal_completion(trace, golden_answer),
        eval_efficiency(trace, optimal_tool_calls),
        eval_final_hallucination(trace),
    ]
