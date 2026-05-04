"""
tracer.py — Captures agent traces and persists them to JSON files.
Traces are structured and queryable; each run appends to traces.json.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Optional
from framework.schema import Trace, EvalResult

try:
    from framework.trace_index import TraceIndex
except ImportError:
    TraceIndex = None  # type: ignore

DATA_DIR = Path(__file__).parent.parent / "data"
TRACES_FILE = DATA_DIR / "traces.json"
EVAL_RESULTS_FILE = DATA_DIR / "eval_results.json"


def _load_json(path: Path) -> list:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_json(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class Tracer:
    """Lightweight tracer that writes structured traces to JSON."""

    def __init__(self, data_dir: Optional[Path] = None, use_sqlite_index: bool = True):
        self.data_dir = data_dir or DATA_DIR
        self.traces_file = self.data_dir / "traces.json"
        self.eval_results_file = self.data_dir / "eval_results.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._index: Optional["TraceIndex"] = None
        if use_sqlite_index and TraceIndex is not None:
            self._index = TraceIndex(self.data_dir / "traces_index.db")

    # ── Trace CRUD ─────────────────────────────

    def save_trace(self, trace: Trace) -> None:
        traces = _load_json(self.traces_file)
        # Update if exists, else append
        for i, t in enumerate(traces):
            if t["trace_id"] == trace.trace_id:
                traces[i] = trace.model_dump()
                _save_json(self.traces_file, traces)
                if self._index is not None:
                    self._index.upsert(trace)
                return
        traces.append(trace.model_dump())
        _save_json(self.traces_file, traces)
        if self._index is not None:
            self._index.upsert(trace)

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        for t in _load_json(self.traces_file):
            if t["trace_id"] == trace_id:
                return Trace(**t)
        return None

    def get_traces_by_version(self, agent_version: str) -> List[Trace]:
        return [
            Trace(**t) for t in _load_json(self.traces_file)
            if t.get("agent_version") == agent_version
        ]

    def get_all_traces(self) -> List[Trace]:
        return [Trace(**t) for t in _load_json(self.traces_file)]

    def reindex_sqlite_from_json(self) -> int:
        """Rebuild SQLite from traces.json (recovery / migration)."""
        if self._index is None:
            return 0
        return self._index.reindex_from_json(self.traces_file)

    def query_traces(self, **kwargs):
        """Delegate to SQLite index (tool_name, span_type, agent_version, limit)."""
        if self._index is None:
            return []
        return self._index.query_traces(**kwargs)

    # ── EvalResult CRUD ────────────────────────

    def save_eval_result(self, result: EvalResult) -> None:
        results = _load_json(self.eval_results_file)
        results.append(result.model_dump())
        _save_json(self.eval_results_file, results)

    def save_eval_results(self, results: List[EvalResult]) -> None:
        existing = _load_json(self.eval_results_file)
        existing.extend([r.model_dump() for r in results])
        _save_json(self.eval_results_file, existing)

    def get_eval_results(
        self,
        trace_id: Optional[str] = None,
        agent_version: Optional[str] = None,
        eval_layer: Optional[str] = None,
    ) -> List[EvalResult]:
        all_results = _load_json(self.eval_results_file)
        # Join with traces if filtering by version
        trace_ids_for_version: Optional[set] = None
        if agent_version:
            traces = _load_json(self.traces_file)
            trace_ids_for_version = {
                t["trace_id"] for t in traces
                if t.get("agent_version") == agent_version
            }

        filtered = []
        for r in all_results:
            if trace_id and r["trace_id"] != trace_id:
                continue
            if trace_ids_for_version and r["trace_id"] not in trace_ids_for_version:
                continue
            if eval_layer and r["eval_layer"] != eval_layer:
                continue
            filtered.append(EvalResult(**r))
        return filtered

    def get_all_eval_results(self) -> List[EvalResult]:
        return [EvalResult(**r) for r in _load_json(self.eval_results_file)]

    # ── Scoring helpers ────────────────────────

    def compute_version_scores(self, agent_version: str) -> dict:
        """Aggregate scores per eval across all traces for a given version."""
        results = self.get_eval_results(agent_version=agent_version)
        if not results:
            return {}

        by_eval: dict = {}
        for r in results:
            key = r.eval_name
            if key not in by_eval:
                by_eval[key] = {"scores": [], "layer": r.eval_layer, "passed": []}
            by_eval[key]["scores"].append(r.score)
            by_eval[key]["passed"].append(r.passed)

        summary = {}
        for eval_name, data in by_eval.items():
            scores = data["scores"]
            summary[eval_name] = {
                "layer": data["layer"],
                "avg_score": round(sum(scores) / len(scores), 3),
                "pass_rate": round(sum(data["passed"]) / len(data["passed"]), 3),
                "n": len(scores),
            }
        return summary
