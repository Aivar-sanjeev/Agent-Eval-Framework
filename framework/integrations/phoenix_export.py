"""
Export traces to OpenInference-style JSON Lines for Arize Phoenix / similar tools.

Phoenix and related stacks ingest spans with OpenTelemetry semantic conventions.
This exporter emits simplified span documents you can batch-upload or inspect locally.

References:
  - OpenInference: https://github.com/Arize-ai/openinference
  - Arize Phoenix: https://docs.arize.com/phoenix

Each line is a JSON object with trace_id, span_id, parent_span_id, span_kind,
attributes (tool metadata), and eval-friendly context — suitable for comparing
with PromptFoo/RAGAS-style offline workflows when imported into Phoenix UI.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from framework.schema import Trace


def trace_to_openinference_records(trace: Trace) -> List[Dict[str, Any]]:
    """
    Flatten a Trace into span-shaped dicts (OpenInference-inspired attributes).
    """
    records: List[Dict[str, Any]] = []
    parent_by_step: Dict[int, str] = {}

    for s in trace.spans:
        span_type = s.get("span_type", "unknown")
        sid = s.get("span_id") or ""

        if span_type == "reasoning":
            kind = "CHAIN"
            name = "reasoning"
            attrs = {
                "input.value": trace.query[:2000],
                "output.value": (s.get("reasoning_text") or "")[:4000],
                "llm.available_tools": ",".join(s.get("available_tools") or []),
                "tool.selected": s.get("selected_tool") or "",
            }
        elif span_type == "tool_call":
            kind = "TOOL"
            name = s.get("tool_name") or "tool"
            attrs = {
                "tool.name": name,
                "input.value": json.dumps(s.get("tool_parameters") or {}),
                "output.value": str(s.get("tool_result"))[:8000],
                "tool.latency_ms": s.get("latency_ms"),
            }
        elif span_type == "interpretation":
            kind = "CHAIN"
            name = "interpretation"
            attrs = {
                "output.value": json.dumps(
                    {
                        "summary": s.get("tool_result_summary"),
                        "next_action": s.get("next_action"),
                        "reasoning": s.get("reasoning_for_next"),
                    }
                ),
            }
        elif span_type == "final_answer":
            kind = "CHAIN"
            name = "final_answer"
            attrs = {
                "output.value": (s.get("answer_text") or "")[:8000],
            }
        else:
            kind = "CHAIN"
            name = span_type
            attrs = {"output.value": json.dumps(s)}

        step = s.get("step_index")
        parent_span_id = parent_by_step.get(int(step) - 1, "") if step else ""

        rec = {
            "trace_id": trace.trace_id,
            "span_id": sid,
            "parent_span_id": parent_span_id,
            "span_kind": kind,
            "span_name": name,
            "agent_version": trace.agent_version,
            "attributes": attrs,
            "metadata": trace.metadata,
        }
        records.append(rec)
        if step is not None:
            parent_by_step[int(step)] = sid

    return records


def export_traces_jsonl(traces: Iterable[Trace], out_path: Path) -> int:
    """Write one JSON object per line (NDJSON). Returns line count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for trace in traces:
            for row in trace_to_openinference_records(trace):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    return n


def iter_export_file(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                yield json.loads(line)
