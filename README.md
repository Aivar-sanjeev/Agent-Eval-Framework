# Agent Evaluation Framework

Multi-layer evaluation for **tool-using agents**: measure quality **before** tool calls (reasoning + tool choice), **at/after** tool calls (parameters, interpretation, next step), and **end-to-end** (goal completion, efficiency, grounded answers).

The agent implementation is **decoupled** from evaluators: you register evals in `evals/eval_definitions.py` and implement logic under `framework/evaluators/`. The demo agent in `agent/research_agent.py` only emits structured **traces**; it does not import eval code.

Inference and LLM-as-judge calls use **NVIDIA NIM** (OpenAI-compatible `chat/completions` at `integrate.api.nvidia.com`).

## Features

| Capability | How |
|------------|-----|
| Framework-agnostic traces | Pydantic models in `framework/schema.py` — wire any runtime (LangChain, Bedrock, custom) by producing the same JSON shape |
| Deterministic + model evals | Schema checks + NVIDIA-backed judges |
| Versioned datasets | `framework/dataset.py` → `data/datasets/v{version}.json` |
| Queryable traces | JSON source of truth + **SQLite** index (`data/traces_index.db`) with `query_traces()` |
| Score history | Append-only `data/score_history.jsonl` on each suite run — dashboard shows recent runs |
| Deployment gate | Thresholds in `framework/runner.py` → `python -m framework.cli gate --version v2.0` exits **1** if blocked |
| External tool inspiration | **OpenInference-style** JSONL export for **Arize Phoenix**-class UIs; **Braintrust-shaped** rows in `framework/integrations/braintrust_compat.py` |

## Setup

```bash
cd agent_eval_framework
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env   # set NVIDIA_API_KEY
```

Environment variables:

| Variable | Purpose |
|----------|---------|
| `NVIDIA_API_KEY` | Required for the demo agent and LLM judges |
| `NVIDIA_BASE_URL` | Default `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_AGENT_MODEL` | Model id for the research agent (default `meta/llama-3.3-70b-instruct`) |
| `NVIDIA_JUDGE_MODEL` | Model id for eval judges (same default) |

## Quickstart

**Full demo** (runs agent v1.0 and v2.0 on five benchmark queries, evaluates all layers, writes datasets, updates score history, renders HTML):

```bash
python run_demo.py
```

Open `reports/dashboard.html` for scores, deployment gate status, **recent eval suite runs**, and charts.

**Without GPU/API** — synthetic dashboard only:

```bash
python generate_demo_dashboard.py
```

## CLI

Run from this directory (`agent_eval_framework` on `PYTHONPATH`):

```bash
python -m framework.cli gate --version v2.0
python -m framework.cli query --tool web_search --limit 20
python -m framework.cli export-phoenix -o exports/phoenix_spans.jsonl
python -m framework.cli dashboard --versions v1.0 v2.0
python -m framework.cli reindex
```

- **gate** — CI-friendly: exit code **1** if any metric is below `GATE_THRESHOLDS` in `framework/runner.py`.
- **export-phoenix** — NDJSON spans with OpenInference-style fields for Phoenix / offline analysis.
- **query** — SQLite-backed filters (`--agent-version`, `--tool`, `--span-type`).
- **reindex** — Rebuild SQLite from `data/traces.json` if the index is missing or stale.

## Three evaluation layers

1. **Pre-tool** (`framework/evaluators/pre_tool.py`): intent clarity, tool selection (skipped when the model answers without tools), pre-call hallucination.
2. **Post-tool** (`framework/evaluators/post_tool.py`): parameter schema (deterministic), parameter quality, result interpretation, next-step decision.
3. **E2E** (`framework/evaluators/e2e.py`): goal completion, efficiency (deterministic vs `optimal_tool_calls` label), final-answer grounding.

All specs and thresholds are documented in `evals/eval_definitions.py`.

## Datasets and labels

Human labels live in `evals/labels.json` (golden answers, `optimal_tool_calls`, etc.). `DatasetManager.build_from_traces` merges traces with labels into versioned dataset files under `data/datasets/`.

Production sampling workflow (conceptual): export traces → label in UI or spreadsheet → merge into `labels.json` or a new dataset version → re-run `run_suite`.

## Integrations (inspiration)

- **Arize Phoenix / OpenInference**: `framework/integrations/phoenix_export.py` — import NDJSON into Phoenix or adapt to OTLP later.
- **Braintrust-style experiments**: `trace_to_braintrust_row()` for `input` / `output` / `scores` rows.
- **PromptFoo / RAGAS**: use the same trace JSON as offline test cases; judges mirror RAGAS-style rubrics without coupling to those libraries.

## Project layout

```
agent_eval_framework/
  agent/research_agent.py    # NVIDIA NIM demo agent + tool traces
  evals/eval_definitions.py  # Eval registry (documentation + thresholds)
  framework/
    schema.py                # Trace, spans, EvalResult, Dataset
    tracer.py                # JSON persistence + SQLite index
    trace_index.py           # Queryable span index
    runner.py                # Suite orchestration + deployment gate + score_history append
    dataset.py               # Versioned datasets
    llm_judge.py             # NVIDIA NIM judge helper
    dashboard.py             # Static HTML report
    cli.py                   # Gate, export, query, dashboard
    integrations/            # Phoenix + Braintrust-compat exporters
  data/
    traces.json
    traces_index.db
    eval_results.json
    score_history.jsonl
    datasets/v*.json
  reports/dashboard.html
```

## License

Use and modify freely for evaluation workflows internal to your team.
