"""
Append-only score history for tracking agent quality across runs and releases.
Written as JSON Lines: one record per eval suite run.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).parent.parent / "data"
HISTORY_FILE = DATA_DIR / "score_history.jsonl"


def append_run(
    *,
    agent_version: str,
    scores: Dict[str, Any],
    gate_passed: bool,
    dataset_version: Optional[str] = None,
    notes: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "agent_version": agent_version,
        "dataset_version": dataset_version,
        "scores": scores,
        "gate_passed": gate_passed,
        "notes": notes,
        "metadata": metadata or {},
    }
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_history(limit_runs: int = 200) -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    lines = HISTORY_FILE.read_text(encoding="utf-8").strip().splitlines()
    rows = []
    for line in lines[-limit_runs:]:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def runs_for_version(agent_version: str) -> List[Dict[str, Any]]:
    return [r for r in load_history(500) if r.get("agent_version") == agent_version]
