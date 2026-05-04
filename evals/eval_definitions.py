"""
eval_definitions.py — All eval specifications in one place.
New evals can be added here without touching the agent code.
This file is the authoritative registry of what gets evaluated and why.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvalSpec:
    """Specification for a single evaluator."""
    eval_name: str
    layer: str           # pre_tool | post_tool | e2e
    eval_type: str       # deterministic | llm_judge
    description: str
    passing_threshold: float
    what_it_catches: str
    tags: List[str] = field(default_factory=list)


# ── Registry of all evals ─────────────────────────────────────────────────────

EVAL_REGISTRY: List[EvalSpec] = [

    # ── Layer 1: Pre-tool ──────────────────────────────────────────────────────
    EvalSpec(
        eval_name="intent_clarity",
        layer="pre_tool",
        eval_type="llm_judge",
        description=(
            "Checks that the agent correctly identified the user's goal from the query "
            "before deciding to use a tool."
        ),
        passing_threshold=0.80,
        what_it_catches="Misunderstood queries that lead the agent down wrong paths.",
        tags=["reasoning", "intent"],
    ),
    EvalSpec(
        eval_name="tool_selection",
        layer="pre_tool",
        eval_type="llm_judge",
        description=(
            "Verifies the agent chose the most appropriate tool from the available set "
            "given the user's query."
        ),
        passing_threshold=0.80,
        what_it_catches="Wrong tool selection (e.g. using calculator for a factual lookup).",
        tags=["tool_use", "reasoning"],
    ),
    EvalSpec(
        eval_name="pre_call_hallucination",
        layer="pre_tool",
        eval_type="llm_judge",
        description=(
            "Checks the agent isn't inventing constraints, facts, or context not present "
            "in the user's query before the first tool call."
        ),
        passing_threshold=0.80,
        what_it_catches="Fabricated assumptions that corrupt the tool call parameters.",
        tags=["hallucination", "reasoning"],
    ),

    # ── Layer 2: Post-tool ─────────────────────────────────────────────────────
    EvalSpec(
        eval_name="param_schema_validation",
        layer="post_tool",
        eval_type="deterministic",
        description=(
            "Deterministic check that tool call parameters match the expected JSON schema: "
            "required fields present, correct types."
        ),
        passing_threshold=0.95,
        what_it_catches="Structural parameter errors that would cause tool failures.",
        tags=["parameters", "schema", "deterministic"],
    ),
    EvalSpec(
        eval_name="param_value_quality",
        layer="post_tool",
        eval_type="llm_judge",
        description=(
            "Checks that parameter VALUES are sensible and precise given the user's query, "
            "not just structurally valid."
        ),
        passing_threshold=0.75,
        what_it_catches="Semantically bad parameters (e.g. vague search queries).",
        tags=["parameters", "quality"],
    ),
    EvalSpec(
        eval_name="result_interpretation",
        layer="post_tool",
        eval_type="llm_judge",
        description=(
            "Checks the agent correctly summarised and extracted key information from "
            "the tool's raw result."
        ),
        passing_threshold=0.75,
        what_it_catches="Misread tool outputs that corrupt subsequent reasoning.",
        tags=["interpretation", "reasoning"],
    ),
    EvalSpec(
        eval_name="next_step_decision",
        layer="post_tool",
        eval_type="llm_judge",
        description=(
            "Evaluates whether the agent's decision after a tool call (continue / "
            "call another tool / answer) was appropriate."
        ),
        passing_threshold=0.75,
        what_it_catches="Premature answers or unnecessary extra tool calls.",
        tags=["decision", "reasoning"],
    ),

    # ── Layer 3: End-to-End ────────────────────────────────────────────────────
    EvalSpec(
        eval_name="goal_completion",
        layer="e2e",
        eval_type="llm_judge",
        description=(
            "Judges whether the agent's final answer fully and correctly addresses "
            "the user's original query. Compared against golden answer if available."
        ),
        passing_threshold=0.70,
        what_it_catches="Incomplete or wrong final answers.",
        tags=["answer_quality", "correctness"],
    ),
    EvalSpec(
        eval_name="efficiency",
        layer="e2e",
        eval_type="deterministic",
        description=(
            "Measures whether the agent used the minimum necessary tool calls. "
            "Penalises excessive or redundant calls."
        ),
        passing_threshold=0.65,
        what_it_catches="Loops, redundant searches, and over-eager tool use.",
        tags=["efficiency", "deterministic"],
    ),
    EvalSpec(
        eval_name="final_hallucination",
        layer="e2e",
        eval_type="llm_judge",
        description=(
            "Checks that the final answer contains no factual claims beyond what "
            "was returned by the tool calls (grounded generation)."
        ),
        passing_threshold=0.75,
        what_it_catches="Fabricated facts, numbers, or names in the final answer.",
        tags=["hallucination", "grounding"],
    ),
]


def get_eval(name: str) -> Optional[EvalSpec]:
    for e in EVAL_REGISTRY:
        if e.eval_name == name:
            return e
    return None


def list_evals(layer: Optional[str] = None) -> List[EvalSpec]:
    if layer:
        return [e for e in EVAL_REGISTRY if e.layer == layer]
    return EVAL_REGISTRY
