"""
runner.py — Orchestrates all evals across all three layers.
Also handles deployment gate logic: a failing suite blocks a release.
"""
from __future__ import annotations
from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

from framework.schema import Dataset, DatasetEntry, EvalResult, Trace
from framework.tracer import Tracer
from framework.evaluators.pre_tool import run_pre_tool_evals
from framework.evaluators.post_tool import run_post_tool_evals
from framework.evaluators.e2e import run_e2e_evals
from framework.score_history import append_run

console = Console()


def _emit(silent: bool, *args, **kwargs) -> None:
    if not silent:
        console.print(*args, **kwargs)


# ── Deployment gate thresholds ─────────────────────────────────────────────────
GATE_THRESHOLDS: Dict[str, float] = {
    "intent_clarity":         0.80,
    "tool_selection":         0.80,
    "pre_call_hallucination": 0.80,
    "param_schema_validation":0.95,   # strict — structural correctness
    "param_value_quality":    0.75,
    "result_interpretation":  0.75,
    "next_step_decision":     0.75,
    "goal_completion":        0.70,
    "efficiency":             0.65,
    "final_hallucination":    0.75,
}


def _eval_detail_table(title: str, results: List[EvalResult], silent: bool = False) -> None:
    """Rich table: every eval with score, pass, type, and judge/deterministic reason."""
    if silent or not results:
        return
    table = Table(
        title=title,
        box=box.MINIMAL_DOUBLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        expand=True,
        title_style="bold white",
    )
    table.add_column("Eval", style="cyan", no_wrap=True, max_width=22)
    table.add_column("Layer", justify="center", max_width=10)
    table.add_column("Score", justify="right", max_width=6)
    table.add_column("Pass", justify="center", max_width=5)
    table.add_column("Type", max_width=14)
    table.add_column("Reason / detail", overflow="fold", max_width=72)

    for r in results:
        pstyle = "green" if r.passed else "red"
        score_txt = f"[{pstyle}]{r.score:.2f}[/{pstyle}]"
        pass_txt = "[green]yes[/green]" if r.passed else "[red]no[/red]"
        reason = (r.reason or "").strip()
        if len(reason) > 600:
            reason = reason[:600] + "…"
        table.add_row(
            r.eval_name,
            r.eval_layer,
            score_txt,
            pass_txt,
            r.eval_type,
            reason,
        )
    console.print(table)
    console.print()


class EvalRunner:
    def __init__(
        self,
        tracer: Optional[Tracer] = None,
        verbose_eval: bool = False,
        silent: bool = False,
    ):
        self.tracer = tracer or Tracer()
        self.verbose_eval = verbose_eval
        self.silent = silent

    def run_all_evals(
        self,
        trace: Trace,
        golden_answer: Optional[str] = None,
        optimal_tool_calls: Optional[int] = None,
    ) -> List[EvalResult]:
        """Run all 3 layers of evals for a single trace."""
        all_results: List[EvalResult] = []
        silent = self.silent

        _emit(silent, f"  [cyan]{'──' if self.verbose_eval else '→'} Layer 1 · Pre-tool[/cyan]", end="")
        if not self.verbose_eval:
            _emit(silent, end=" ")
        else:
            _emit(silent, "")
        pre = run_pre_tool_evals(trace.trace_id, trace.query, trace.spans)
        all_results.extend(pre)
        if self.verbose_eval:
            _eval_detail_table("Layer 1 · Pre-tool (reasoning & tool choice)", pre, silent=silent)
        else:
            _emit(silent, f"[green]{len(pre)} evals[/green]")

        _emit(silent, f"  [cyan]{'──' if self.verbose_eval else '→'} Layer 2 · Post-tool[/cyan]", end="")
        if not self.verbose_eval:
            _emit(silent, end=" ")
        else:
            _emit(silent, "")
        post = run_post_tool_evals(trace.trace_id, trace.query, trace.spans)
        all_results.extend(post)
        if self.verbose_eval:
            _eval_detail_table("Layer 2 · Post-tool (params & interpretation)", post, silent=silent)
        else:
            _emit(silent, f"[green]{len(post)} evals[/green]")

        _emit(silent, f"  [cyan]{'──' if self.verbose_eval else '→'} Layer 3 · End-to-end[/cyan]", end="")
        if not self.verbose_eval:
            _emit(silent, end=" ")
        else:
            _emit(silent, "")
        e2e = run_e2e_evals(trace, golden_answer, optimal_tool_calls)
        all_results.extend(e2e)
        if self.verbose_eval:
            _eval_detail_table("Layer 3 · End-to-end (goal & grounding)", e2e, silent=silent)
        else:
            _emit(silent, f"[green]{len(e2e)} evals[/green]")

        # Persist
        self.tracer.save_eval_results(all_results)
        return all_results

    def run_suite(
        self,
        agent_version: str,
        entries: List[DatasetEntry],
        traces: Dict[str, Trace],
    ) -> Dict[str, float]:
        """Run the full eval suite for a dataset version. Returns aggregated scores."""
        silent = self.silent
        _emit(
            silent,
            Panel(
                f"[bold]Eval suite · agent version:[/bold] [yellow]{agent_version}[/yellow]",
                style="blue",
                padding=(1, 2),
            ),
        )
        all_results: List[EvalResult] = []

        for idx, entry in enumerate(entries, 1):
            trace = traces.get(entry.trace_id)
            if not trace:
                _emit(silent, f"  [yellow]⚠ No trace found for {entry.trace_id}[/yellow]")
                continue
            _emit(silent, "")
            _emit(
                silent,
                Panel(
                    Text(entry.query, overflow="fold"),
                    title=f"[bold]Evaluating trace {idx}/{len(entries)}[/bold]",
                    subtitle=f"[dim]{entry.trace_id}[/dim]",
                    border_style="magenta",
                    padding=(1, 2),
                ),
            )
            results = self.run_all_evals(
                trace,
                golden_answer=entry.golden_answer,
                optimal_tool_calls=entry.labels.get("optimal_tool_calls"),
            )
            all_results.extend(results)

        scores = self.tracer.compute_version_scores(agent_version)
        gate_passed, failures = self.check_deployment_gate(scores)
        append_run(
            agent_version=agent_version,
            scores=scores,
            gate_passed=gate_passed,
            dataset_version=agent_version,
            metadata={"gate_failures": failures},
        )
        return scores

    def check_deployment_gate(self, scores: Dict[str, dict]) -> tuple[bool, List[str]]:
        """
        Returns (gate_passed: bool, failures: List[str]).
        gate_passed=False means deployment is BLOCKED.
        """
        failures = []
        for eval_name, threshold in GATE_THRESHOLDS.items():
            if eval_name not in scores:
                continue
            avg_score = scores[eval_name]["avg_score"]
            if avg_score < threshold:
                failures.append(
                    f"{eval_name}: {avg_score:.2f} < {threshold:.2f} (threshold)"
                )
        return len(failures) == 0, failures

    def print_score_table(self, version: str, scores: Dict[str, dict]) -> None:
        if self.silent:
            return
        table = Table(
            title=f"Eval Scores — Agent {version}",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Eval Name", style="cyan", width=28)
        table.add_column("Layer", justify="center")
        table.add_column("Avg Score", justify="right")
        table.add_column("Pass Rate", justify="right")
        table.add_column("N", justify="right")
        table.add_column("Gate", justify="center")

        layer_order = {"pre_tool": 0, "post_tool": 1, "e2e": 2}
        sorted_evals = sorted(scores.items(), key=lambda x: layer_order.get(x[1]["layer"], 9))

        for eval_name, data in sorted_evals:
            threshold = GATE_THRESHOLDS.get(eval_name, 0.0)
            gate_ok = data["avg_score"] >= threshold
            gate_str = "✅" if gate_ok else "❌"
            score_color = "green" if data["avg_score"] >= threshold else "red"
            table.add_row(
                eval_name,
                data["layer"],
                f"[{score_color}]{data['avg_score']:.2f}[/{score_color}]",
                f"{data['pass_rate']:.0%}",
                str(data["n"]),
                gate_str,
            )

        console.print(table)

        gate_passed, failures = self.check_deployment_gate(scores)
        if gate_passed:
            console.print(Panel(
                "[bold green]✅ DEPLOYMENT GATE PASSED[/bold green]\nAll eval thresholds met.",
                style="green"
            ))
        else:
            failure_text = "\n".join(f"  • {f}" for f in failures)
            console.print(Panel(
                f"[bold red]❌ DEPLOYMENT BLOCKED[/bold red]\n\nFailing evals:\n{failure_text}",
                style="red"
            ))
