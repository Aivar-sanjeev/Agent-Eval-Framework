"""
SQLite index for structured, queryable traces (supplements JSON files).

Use this for filtering by tool name, agent version, span type, or full-text-like
conditions without scanning large JSON arrays in Python.
"""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from framework.schema import Trace

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_PATH = DATA_DIR / "traces_index.db"


class TraceIndex:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or INDEX_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    agent_version TEXT NOT NULL,
                    query TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    final_answer TEXT,
                    span_count INTEGER DEFAULT 0,
                    tool_call_count INTEGER DEFAULT 0,
                    raw_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_traces_version ON traces(agent_version);

                CREATE TABLE IF NOT EXISTS spans (
                    span_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL REFERENCES traces(trace_id) ON DELETE CASCADE,
                    step_index INTEGER,
                    span_type TEXT NOT NULL,
                    tool_name TEXT,
                    selected_tool TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
                CREATE INDEX IF NOT EXISTS idx_spans_tool ON spans(tool_name);
                CREATE INDEX IF NOT EXISTS idx_spans_type ON spans(span_type);
                """
            )

    def upsert(self, trace: Trace) -> None:
        raw = json.dumps(trace.model_dump(), ensure_ascii=False)
        tool_calls = sum(1 for s in trace.spans if s.get("span_type") == "tool_call")
        with self._connect() as c:
            c.execute("DELETE FROM spans WHERE trace_id = ?", (trace.trace_id,))
            c.execute(
                """
                INSERT OR REPLACE INTO traces (
                    trace_id, agent_version, query, started_at, finished_at,
                    final_answer, span_count, tool_call_count, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.trace_id,
                    trace.agent_version,
                    trace.query,
                    trace.started_at,
                    trace.finished_at,
                    trace.final_answer,
                    len(trace.spans),
                    tool_calls,
                    raw,
                ),
            )
            for s in trace.spans:
                sid = s.get("span_id") or ""
                stype = s.get("span_type") or "unknown"
                tool_name = s.get("tool_name")
                selected = s.get("selected_tool")
                c.execute(
                    """
                    INSERT INTO spans (span_id, trace_id, step_index, span_type, tool_name, selected_tool)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sid or f"{trace.trace_id}-{stype}-{s.get('step_index')}",
                        trace.trace_id,
                        s.get("step_index"),
                        stype,
                        tool_name,
                        selected,
                    ),
                )

    def reindex_from_json(self, traces_file: Path) -> int:
        """Rebuild SQLite from a traces.json array file."""
        if not traces_file.exists():
            return 0
        with open(traces_file, encoding="utf-8") as f:
            rows = json.load(f)
        n = 0
        for row in rows:
            upsert_trace = Trace(**row)
            self.upsert(upsert_trace)
            n += 1
        return n

    def query_traces(
        self,
        *,
        agent_version: Optional[str] = None,
        tool_name: Optional[str] = None,
        span_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Return trace rows matching optional filters (AND semantics).
        """
        clauses: List[str] = []
        params: List[Any] = []

        if agent_version:
            clauses.append("t.agent_version = ?")
            params.append(agent_version)

        join_spans = tool_name or span_type
        base_from = "traces t"
        if join_spans:
            base_from += " INNER JOIN spans s ON s.trace_id = t.trace_id"
            if tool_name:
                clauses.append("s.tool_name = ?")
                params.append(tool_name)
            if span_type:
                clauses.append("s.span_type = ?")
                params.append(span_type)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT DISTINCT t.trace_id, t.agent_version, t.query, t.tool_call_count, t.started_at
            FROM {base_from}
            {where}
            ORDER BY t.started_at DESC
            LIMIT ?
        """
        params.append(limit)

        with self._connect() as c:
            cur = c.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def get_trace_json(self, trace_id: str) -> Optional[dict]:
        with self._connect() as c:
            row = c.execute(
                "SELECT raw_json FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            if not row:
                return None
            return json.loads(row["raw_json"])
