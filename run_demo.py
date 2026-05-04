"""
run_demo.py — Full end-to-end demonstration of the Agent Eval Framework.

Runs two agent versions (v1 and v2) on 5 benchmark queries, evaluates all
3 layers, builds versioned datasets, and generates the HTML dashboard.

Usage:
    export NVIDIA_API_KEY=your_key_here
    python run_demo.py
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent))

from agent.research_agent import ResearchAgent
from framework.tracer import Tracer
from framework.dataset import DatasetManager
from framework.runner import EvalRunner
from framework.dashboard import generate_dashboard

console = Console()

# ── Benchmark queries ─────────────────────────────────────────────────────────
BENCHMARK_QUERIES = [
    "What is the current CO2 level in the atmosphere and how has it changed?",
    "If I invest $5000 at 7% annual compound interest for 10 years, how much will I have?",
    "What's the weather in Tokyo and what is 25 degrees Celsius in Fahrenheit?",
    "How far is a marathon in miles, and how long would it take to run at 6 minutes per km?",
    "What is NVIDIA's current revenue trend and what does their stock performance indicate about AI demand?",
]

# ── Human labels (from evals/labels.json) ────────────────────────────────────
LABELS_PATH = Path(__file__).parent / "evals" / "labels.json"

def load_labels() -> list:
    with open(LABELS_PATH) as f:
        return json.load(f)["queries"]


# ── Agent versions ────────────────────────────────────────────────────────────

V1_SYSTEM_PROMPT = """You are a research assistant. Use tools to answer questions.
Be thorough."""

V2_SYSTEM_PROMPT = """You are a precise research assistant with expertise in using tools efficiently.
Before calling a tool:
  1. Clearly identify what specific information you need.
  2. Choose the single most appropriate tool.
  3. Formulate a precise, specific query/parameter — avoid vague inputs.
After getting a tool result:
  4. Extract the key facts directly relevant to the user's question.
  5. Decide if you have enough to answer, or if one more targeted tool call is needed.
  6. Avoid redundant calls — if you have the answer, give it.
Always give a concise, accurate, well-structured final answer grounded in tool results."""


def run_version(
    version: str,
    system_prompt: str,
    tracer: Tracer,
    labels: list,
) -> list:
    """Run all benchmark queries for one agent version. Returns list of traces."""
    console.print(Rule(f"[bold cyan]Agent Version: {version}[/bold cyan]"))
    agent = ResearchAgent(version=version, system_prompt=system_prompt, tracer=tracer)
    traces = []
    for i, query in enumerate(BENCHMARK_QUERIES, 1):
        console.print(f"\n[bold]Query {i}/{len(BENCHMARK_QUERIES)}:[/bold] {query[:80]}...")
        try:
            trace = agent.run(query)
            traces.append(trace)
            spans = len(trace.spans)
            tool_calls = sum(1 for s in trace.spans if s.get("span_type") == "tool_call")
            console.print(f"  [green]✓[/green] Done — {spans} spans, {tool_calls} tool calls")
        except Exception as e:
            console.print(f"  [red]✗ Error:[/red] {e}")
    return traces


def eval_version(
    version: str,
    traces: list,
    labels: list,
    runner: EvalRunner,
    dm: DatasetManager,
) -> dict:
    """Run all evals for a version's traces and build dataset."""
    console.print(Rule(f"[bold yellow]Evaluating Version: {version}[/bold yellow]"))

    # Build label lookup by query text
    label_by_query = {l["query"]: l for l in labels}

    # Build dataset entries
    trace_dicts = [t.model_dump() for t in traces]
    label_map = {}
    for trace in traces:
        lbl = label_by_query.get(trace.query, {})
        label_map[trace.trace_id] = lbl

    dataset = dm.build_from_traces(
        version=version,
        description=f"Benchmark dataset for agent {version} — 5 queries",
        trace_dicts=trace_dicts,
        labels=label_map,
    )

    # Run evals
    trace_lookup = {t.trace_id: t for t in traces}
    scores = runner.run_suite(version, dataset.entries, trace_lookup)

    # Print score table
    runner.print_score_table(version, scores)

    return scores


def main():
    console.print(Panel.fit(
        "[bold white]Agent Evaluation Framework[/bold white]\n"
        "[dim]NVIDIA NIM · JSON storage · 3-layer evaluation[/dim]",
        style="blue",
        padding=(1, 4),
    ))

    # Check API key
    if not os.environ.get("NVIDIA_API_KEY"):
        console.print("[bold red]Error:[/bold red] NVIDIA_API_KEY environment variable not set.")
        console.print("  export NVIDIA_API_KEY=your_key_here")
        sys.exit(1)

    # Setup
    tracer = Tracer()
    runner = EvalRunner(tracer)
    dm = DatasetManager()
    labels = load_labels()

    # ── Version 1: Basic agent ────────────────────────────────────────────────
    v1_traces = run_version("v1.0", V1_SYSTEM_PROMPT, tracer, labels)
    v1_scores = eval_version("v1.0", v1_traces, labels, runner, dm)

    # ── Version 2: Improved agent ─────────────────────────────────────────────
    v2_traces = run_version("v2.0", V2_SYSTEM_PROMPT, tracer, labels)
    v2_scores = eval_version("v2.0", v2_traces, labels, runner, dm)

    # ── Comparison summary ────────────────────────────────────────────────────
    console.print(Rule("[bold green]Version Comparison[/bold green]"))
    all_evals = set(v1_scores.keys()) | set(v2_scores.keys())
    improvements, regressions = [], []
    for name in all_evals:
        s1 = v1_scores.get(name, {}).get("avg_score", 0)
        s2 = v2_scores.get(name, {}).get("avg_score", 0)
        delta = s2 - s1
        if delta > 0.05:
            improvements.append((name, s1, s2, delta))
        elif delta < -0.05:
            regressions.append((name, s1, s2, delta))

    if improvements:
        console.print("\n[green]Improvements in v2.0:[/green]")
        for name, s1, s2, d in sorted(improvements, key=lambda x: -x[3]):
            console.print(f"  {name}: {s1:.2f} → {s2:.2f} (+{d:.2f})")

    if regressions:
        console.print("\n[red]Regressions in v2.0:[/red]")
        for name, s1, s2, d in sorted(regressions, key=lambda x: x[3]):
            console.print(f"  {name}: {s1:.2f} → {s2:.2f} ({d:.2f})")

    if not improvements and not regressions:
        console.print("  No significant changes between versions.")

    # ── Dashboard ─────────────────────────────────────────────────────────────
    console.print(Rule("[bold magenta]Generating Dashboard[/bold magenta]"))
    dashboard_path = generate_dashboard(versions=["v1.0", "v2.0"], tracer=tracer)
    console.print(f"\n[bold green]✅ Dashboard ready:[/bold green] {dashboard_path}")
    console.print("\nOpen the dashboard in your browser to see the full report.\n")


if __name__ == "__main__":
    main()
