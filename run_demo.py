"""
run_demo.py — Full end-to-end demonstration of the Agent Eval Framework.

Runs two agent versions (v1 and v2) on 5 benchmark queries, evaluates all
3 layers, builds versioned datasets, and generates the HTML dashboard.

Usage:
    set GROQ_API_KEY=your_key_here
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

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent))

from agent.research_agent import ResearchAgent, list_tool_catalog
from evals.eval_definitions import EVAL_REGISTRY
from framework.tracer import Tracer
from framework.dataset import DatasetManager
from framework.runner import EvalRunner
from framework.dashboard import generate_dashboard
from framework.settings import groq_agent_model, groq_base_url, groq_judge_model

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
    with open(LABELS_PATH, encoding="utf-8") as f:
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


def _trace_tool_names(trace) -> list[str]:
    return [
        s.get("tool_name", "")
        for s in trace.spans
        if s.get("span_type") == "tool_call" and s.get("tool_name")
    ]


def print_environment_banner() -> None:
    """Endpoint and models (no secrets)."""
    t = Table(title="Groq configuration", box=box.ROUNDED, header_style="bold cyan")
    t.add_column("Setting", style="dim")
    t.add_column("Value", overflow="fold")
    t.add_row("Base URL", groq_base_url())
    t.add_row("Agent model", groq_agent_model())
    t.add_row("Judge model", groq_judge_model())
    console.print(t)
    console.print(
        "[dim]All Groq calls retry on 429 with exponential backoff. "
        "Optional: GROQ_REQUEST_DELAY_MS=50–150 to reduce bursts.[/dim]\n"
    )


def print_available_tools() -> None:
    rows = list_tool_catalog()
    t = Table(title="Available agent tools", box=box.ROUNDED, header_style="bold green", expand=True)
    t.add_column("#", justify="right", style="dim", width=3)
    t.add_column("Tool", style="cyan", no_wrap=True, width=18)
    t.add_column("Description", overflow="fold")
    for i, r in enumerate(rows, 1):
        t.add_row(str(i), r["name"], r["description"])
    console.print(t)
    console.print()


def print_benchmark_schedule() -> None:
    t = Table(title="Benchmark run schedule (this demo)", box=box.SIMPLE, header_style="bold yellow")
    t.add_column("Step", justify="right", style="dim", width=4)
    t.add_column("Query (truncated)", overflow="fold")
    for i, q in enumerate(BENCHMARK_QUERIES, 1):
        t.add_row(str(i), q[:120] + ("…" if len(q) > 120 else ""))
    console.print(t)
    console.print()


def print_eval_registry_summary() -> None:
    t = Table(title="Registered evaluators (all layers)", box=box.MINIMAL, header_style="bold magenta")
    t.add_column("Eval", style="cyan", no_wrap=True, max_width=22)
    t.add_column("Layer", max_width=10)
    t.add_column("Type", max_width=14)
    t.add_column("What it measures", overflow="fold")
    for spec in EVAL_REGISTRY:
        t.add_row(
            spec.eval_name,
            spec.layer,
            spec.eval_type,
            spec.description[:140] + ("…" if len(spec.description) > 140 else ""),
        )
    console.print(t)
    console.print()


def run_version(
    version: str,
    system_prompt: str,
    tracer: Tracer,
    labels: list,
) -> list:
    """Run all benchmark queries for one agent version. Returns list of traces."""
    console.print(Rule(f"[bold cyan]Agent run · version {version}[/bold cyan]"))
    agent = ResearchAgent(version=version, system_prompt=system_prompt, tracer=tracer)
    traces = []
    for i, query in enumerate(BENCHMARK_QUERIES, 1):
        console.print(
            Panel(
                Text(query, overflow="fold"),
                title=f"[bold]Query {i}/{len(BENCHMARK_QUERIES)}[/bold]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        try:
            trace = agent.run(query)
            traces.append(trace)
            spans = len(trace.spans)
            tool_calls = sum(1 for s in trace.spans if s.get("span_type") == "tool_call")
            tools_used = ", ".join(_trace_tool_names(trace)) or "—"
            ans = (trace.final_answer or "")[:350]
            if len(trace.final_answer or "") > 350:
                ans += "…"
            meta = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
            meta.add_row("[dim]Spans[/dim]", str(spans))
            meta.add_row("[dim]Tool calls[/dim]", str(tool_calls))
            meta.add_row("[dim]Tools used[/dim]", tools_used)
            console.print(Panel(meta, title="[green]Trace summary[/green]", border_style="green"))
            console.print(
                Panel(Text(ans, overflow="fold"), title="[dim]Final answer (preview)[/dim]", border_style="dim")
            )
        except Exception as e:
            console.print(f"  [bold red]Error:[/bold red] {e}")
    return traces


def eval_version(
    version: str,
    traces: list,
    labels: list,
    runner: EvalRunner,
    dm: DatasetManager,
) -> dict:
    """Run all evals for a version's traces and build dataset."""
    console.print(Rule(f"[bold yellow]Evaluation · version {version}[/bold yellow]"))

    label_by_query = {l["query"]: l for l in labels}

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

    trace_lookup = {t.trace_id: t for t in traces}
    scores = runner.run_suite(version, dataset.entries, trace_lookup)

    runner.print_score_table(version, scores)

    return scores


def main():
    console.print(
        Panel.fit(
            "[bold white]Agent Evaluation Framework[/bold white]\n"
            "[dim]Groq · structured traces · 3-layer eval · detailed judge reasons[/dim]",
            style="blue",
            padding=(1, 4),
        )
    )

    if not os.environ.get("GROQ_API_KEY"):
        console.print("[bold red]Error:[/bold red] GROQ_API_KEY environment variable not set.")
        console.print("  Add it to .env or set GROQ_API_KEY (see https://console.groq.com/keys ).")
        sys.exit(1)

    print_environment_banner()
    print_available_tools()
    print_benchmark_schedule()
    print_eval_registry_summary()

    tracer = Tracer()
    runner = EvalRunner(tracer, verbose_eval=True)
    dm = DatasetManager()
    labels = load_labels()

    v1_traces = run_version("v1.0", V1_SYSTEM_PROMPT, tracer, labels)
    v1_scores = eval_version("v1.0", v1_traces, labels, runner, dm)

    v2_traces = run_version("v2.0", V2_SYSTEM_PROMPT, tracer, labels)
    v2_scores = eval_version("v2.0", v2_traces, labels, runner, dm)

    console.print(Rule("[bold green]Version comparison (avg scores)[/bold green]"))
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
            console.print(f"  {name}: {s1:.2f} -> {s2:.2f} (+{d:.2f})")

    if regressions:
        console.print("\n[red]Regressions in v2.0:[/red]")
        for name, s1, s2, d in sorted(regressions, key=lambda x: x[3]):
            console.print(f"  {name}: {s1:.2f} -> {s2:.2f} ({d:.2f})")

    if not improvements and not regressions:
        console.print("  No large swings between versions.")

    console.print(Rule("[bold magenta]HTML dashboard[/bold magenta]"))
    dashboard_path = generate_dashboard(versions=["v1.0", "v2.0"], tracer=tracer)
    console.print(f"\n[bold green]Dashboard written:[/bold green] {dashboard_path}")
    console.print("\n[dim]Tip:[/dim] [cyan]streamlit run streamlit_app.py[/cyan] for the browser UI.\n")


if __name__ == "__main__":
    main()
