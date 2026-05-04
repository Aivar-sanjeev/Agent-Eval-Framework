"""
CLI: deployment gate, Phoenix export, trace queries, dashboard regen.

Usage (from repo root):
  python -m framework.cli gate --version v2.0
  python -m framework.cli export-phoenix --output exports/phoenix.jsonl
  python -m framework.cli query --tool web_search --limit 20
  python -m framework.cli dashboard --versions v1.0 v2.0
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass


def cmd_gate(args: argparse.Namespace) -> int:
    from framework.tracer import Tracer
    from framework.runner import EvalRunner

    _load_dotenv()
    tracer = Tracer()
    runner = EvalRunner(tracer)
    scores = tracer.compute_version_scores(args.version)
    if not scores:
        print(f"No eval results found for agent version {args.version}. Run eval suite first.", file=sys.stderr)
        return 2
    gate_passed, failures = runner.check_deployment_gate(scores)
    print(json.dumps({"gate_passed": gate_passed, "scores": scores, "failures": failures}, indent=2))
    return 0 if gate_passed else 1


def cmd_export_phoenix(args: argparse.Namespace) -> int:
    from framework.tracer import Tracer
    from framework.integrations.phoenix_export import export_traces_jsonl

    tracer = Tracer()
    traces = tracer.get_all_traces()
    if args.version:
        traces = [t for t in traces if t.agent_version == args.version]
    out = Path(args.output)
    n = export_traces_jsonl(traces, out)
    print(f"Wrote {n} span rows to {out}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from framework.tracer import Tracer

    tracer = Tracer()
    rows = tracer.query_traces(
        agent_version=args.agent_version,
        tool_name=args.tool,
        span_type=args.span_type,
        limit=args.limit,
    )
    print(json.dumps(rows, indent=2))
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    from framework.tracer import Tracer
    from framework.dashboard import generate_dashboard

    tracer = Tracer()
    path = generate_dashboard(versions=list(args.versions), tracer=tracer)
    print(path)
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    from framework.tracer import Tracer

    n = Tracer().reindex_sqlite_from_json()
    print(f"Indexed {n} traces into SQLite.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Agent Eval Framework CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate", help="Check deployment gate for an agent version (exit 1 if blocked)")
    g.add_argument("--version", required=True, help="Agent version label, e.g. v2.0")
    g.set_defaults(func=cmd_gate)

    e = sub.add_parser("export-phoenix", help="Export traces as OpenInference-style JSONL")
    e.add_argument("--output", "-o", required=True, help="Output .jsonl path")
    e.add_argument("--version", help="Filter by agent_version")
    e.set_defaults(func=cmd_export_phoenix)

    q = sub.add_parser("query", help="SQL-backed trace query (SQLite index)")
    q.add_argument("--agent-version", help="Filter agent_version")
    q.add_argument("--tool", help="Filter spans where tool_name matches")
    q.add_argument("--span-type", help="Filter spans by span_type")
    q.add_argument("--limit", type=int, default=50)
    q.set_defaults(func=cmd_query)

    d = sub.add_parser("dashboard", help="Regenerate HTML dashboard")
    d.add_argument("--versions", nargs="+", required=True)
    d.set_defaults(func=cmd_dashboard)

    r = sub.add_parser("reindex", help="Rebuild SQLite index from data/traces.json")
    r.set_defaults(func=cmd_reindex)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
