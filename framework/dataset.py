"""
dataset.py — Versioned dataset management.
Datasets are stored as JSON files: data/datasets/v{version}.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional
from framework.schema import Dataset, DatasetEntry

DATA_DIR = Path(__file__).parent.parent / "data" / "datasets"


class DatasetManager:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, version: str) -> Path:
        return self.data_dir / f"v{version}.json"

    def save(self, dataset: Dataset) -> None:
        with open(self._path(dataset.version), "w", encoding="utf-8") as f:
            json.dump(dataset.model_dump(), f, indent=2, ensure_ascii=False)
        print(f"  Dataset v{dataset.version} saved ({len(dataset.entries)} entries)")

    def load(self, version: str) -> Optional[Dataset]:
        path = self._path(version)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return Dataset(**json.load(f))

    def list_versions(self) -> List[str]:
        return sorted(
            [p.stem[1:] for p in self.data_dir.glob("v*.json")]
        )

    def add_eval_results_to_dataset(
        self,
        version: str,
        trace_id: str,
        eval_results: List[dict],
    ) -> None:
        """Attach eval results to a dataset entry by trace_id."""
        dataset = self.load(version)
        if not dataset:
            return
        for entry in dataset.entries:
            if entry.trace_id == trace_id:
                entry.eval_results = eval_results
                break
        self.save(dataset)

    def build_from_traces(
        self,
        version: str,
        description: str,
        trace_dicts: List[dict],
        labels: Optional[Dict[str, dict]] = None,
    ) -> Dataset:
        """
        Build and save a versioned dataset from a list of trace dicts.
        labels: {trace_id: {"golden_answer": ..., "optimal_tool_calls": int, ...}}
        """
        labels = labels or {}
        entries = []
        for t in trace_dicts:
            label = labels.get(t["trace_id"], {})
            entries.append(DatasetEntry(
                trace_id=t["trace_id"],
                query=t["query"],
                agent_version=t["agent_version"],
                golden_answer=label.get("golden_answer"),
                expected_tools=label.get("expected_tools", []),
                labels=label,
            ))
        dataset = Dataset(version=version, description=description, entries=entries)
        self.save(dataset)
        return dataset
