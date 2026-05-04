"""
Braintrust-inspired row shape for offline scoring / spreadsheet import.

Braintrust (https://www.braintrust.dev) centers experiments around datasets with
inputs, outputs, and scores. These helpers map our traces into comparable rows
without requiring their SDK or API key — useful for CSV/JSONL interchange.

See also: Braintrust dataset format docs when syncing with their cloud product.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from framework.schema import EvalResult, Trace


def trace_to_braintrust_row(
    trace: Trace,
    eval_results: Optional[List[EvalResult]] = None,
    golden_answer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Single row: input = user query, output = final answer, scores = eval aggregates.
    """
    scores_map: Dict[str, float] = {}
    if eval_results:
        for er in eval_results:
            scores_map[er.eval_name] = er.score

    tool_calls = [
        {
            "tool": s.get("tool_name"),
            "params": s.get("tool_parameters"),
            "result_preview": str(s.get("tool_result"))[:500],
        }
        for s in trace.spans
        if s.get("span_type") == "tool_call"
    ]

    return {
        "id": trace.trace_id,
        "input": trace.query,
        "output": trace.final_answer,
        "expected": golden_answer,
        "metadata": {
            "agent_version": trace.agent_version,
            "tool_calls": tool_calls,
            "span_count": len(trace.spans),
        },
        "scores": scores_map,
    }
