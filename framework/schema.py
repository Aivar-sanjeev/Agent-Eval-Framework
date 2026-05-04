"""
schema.py — Pydantic models for structured traces, spans, eval results, and datasets.
Every agent run produces a Trace containing ordered Spans; evals produce EvalResults.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
import uuid


# ──────────────────────────────────────────────
# Span models (one per step in the agent loop)
# ──────────────────────────────────────────────

class ReasoningSpan(BaseModel):
    """Captures the agent's reasoning before deciding on a tool call."""
    span_type: Literal["reasoning"] = "reasoning"
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    step_index: int
    reasoning_text: str
    identified_intent: Optional[str] = None
    selected_tool: Optional[str] = None
    available_tools: List[str] = []
    raw_llm_output: str = ""


class ToolCallSpan(BaseModel):
    """Captures a single tool invocation — parameters and raw result."""
    span_type: Literal["tool_call"] = "tool_call"
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    step_index: int
    tool_name: str
    tool_parameters: Dict[str, Any] = {}
    tool_result: Any = None
    tool_error: Optional[str] = None
    latency_ms: Optional[float] = None


class InterpretationSpan(BaseModel):
    """Captures the agent's interpretation of a tool result and next-step decision."""
    span_type: Literal["interpretation"] = "interpretation"
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    step_index: int
    tool_result_summary: str
    next_action: Literal["continue", "tool_call", "final_answer"]
    reasoning_for_next: str = ""
    raw_llm_output: str = ""


class FinalAnswerSpan(BaseModel):
    """The agent's final response to the user."""
    span_type: Literal["final_answer"] = "final_answer"
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    answer_text: str
    total_steps: int
    total_tool_calls: int


# Union type for spans
Span = ReasoningSpan | ToolCallSpan | InterpretationSpan | FinalAnswerSpan


# ──────────────────────────────────────────────
# Trace — one complete agent run
# ──────────────────────────────────────────────

class Trace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_version: str
    query: str
    spans: List[Dict[str, Any]] = []   # serialised spans
    final_answer: Optional[str] = None
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    finished_at: Optional[str] = None
    metadata: Dict[str, Any] = {}

    def add_span(self, span: BaseModel) -> None:
        self.spans.append(span.model_dump())

    def close(self, final_answer: str) -> None:
        self.final_answer = final_answer
        self.finished_at = datetime.utcnow().isoformat()


# ──────────────────────────────────────────────
# Eval result models
# ──────────────────────────────────────────────

class EvalResult(BaseModel):
    eval_id: str
    eval_layer: Literal["pre_tool", "post_tool", "e2e"]
    eval_name: str
    trace_id: str
    span_id: Optional[str] = None
    passed: bool
    score: float                        # 0.0 – 1.0
    reason: str
    eval_type: Literal["deterministic", "llm_judge"]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = {}


# ──────────────────────────────────────────────
# Dataset — versioned collection of traces + labels
# ──────────────────────────────────────────────

class DatasetEntry(BaseModel):
    trace_id: str
    query: str
    agent_version: str
    golden_answer: Optional[str] = None
    expected_tools: List[str] = []
    labels: Dict[str, Any] = {}        # human labels keyed by eval_id
    eval_results: List[Dict[str, Any]] = []


class Dataset(BaseModel):
    dataset_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    version: str
    description: str = ""
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    entries: List[DatasetEntry] = []
