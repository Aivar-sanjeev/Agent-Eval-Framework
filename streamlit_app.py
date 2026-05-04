"""
Browser UI for the Agent Eval Framework: tools overview, benchmark queries,
interactive agent run + full eval table with judge reasons.

Usage (from agent_eval_framework directory):
    pip install streamlit
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import csv
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import streamlit as st

from agent.research_agent import ResearchAgent, list_tool_catalog
from evals.eval_definitions import EVAL_REGISTRY
from framework.runner import EvalRunner, GATE_THRESHOLDS
from framework.tracer import Tracer

from run_demo import (
    BENCHMARK_QUERIES,
    V1_SYSTEM_PROMPT,
    V2_SYSTEM_PROMPT,
    load_labels,
)


def main():
    st.set_page_config(
        page_title="Agent Eval Framework",
        page_icon="📊",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"] > div:first-child {
            padding-top: 0.5rem;
        }
        .block-container { max-width: 1200px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Agent evaluation framework")
    st.caption("Multi-layer traces · Groq · deterministic + LLM judges")

    api_ok = bool(os.environ.get("GROQ_API_KEY", "").strip())
    if not api_ok:
        st.error(
            "Set **GROQ_API_KEY** in `.env` (see https://console.groq.com/keys ). "
            "Overview tables below still work without it."
        )
    else:
        st.success("Groq API key is set.")

    with st.expander("Available tools", expanded=True):
        st.dataframe(list_tool_catalog(), width="stretch", hide_index=True)

    with st.expander("Benchmark queries (this demo)"):
        st.dataframe(
            {"#": list(range(1, len(BENCHMARK_QUERIES) + 1)), "query": BENCHMARK_QUERIES},
            width="stretch",
            hide_index=True,
        )

    with st.expander("Registered evaluators"):
        rows = [
            {
                "eval": e.eval_name,
                "layer": e.layer,
                "type": e.eval_type,
                "threshold": e.passing_threshold,
                "description": e.description[:180],
            }
            for e in EVAL_REGISTRY
        ]
        st.dataframe(rows, width="stretch", hide_index=True)

    st.divider()

    st.subheader("Interactive run")
    st.write(
        "Pick an agent version and a query. The app runs the research agent, "
        "then all eval layers and shows **scores and judge reasons** in a table."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        version_choice = st.radio(
            "Agent version",
            ("v1.0 — basic prompt", "v2.0 — improved prompt"),
            horizontal=True,
        )
    with col_b:
        query = st.selectbox("Query", BENCHMARK_QUERIES)

    prompt = V1_SYSTEM_PROMPT if version_choice.startswith("v1") else V2_SYSTEM_PROMPT
    version_tag = "v1.0" if version_choice.startswith("v1") else "v2.0"

    label_by = {l["query"]: l for l in load_labels()}
    lbl = label_by.get(query, {})
    golden = lbl.get("golden_answer")
    optimal = lbl.get("optimal_tool_calls")

    if st.button("Run agent + full eval", type="primary", disabled=not api_ok):
        tracer = Tracer()
        agent = ResearchAgent(version=version_tag, system_prompt=prompt, tracer=tracer)
        with st.status("Running agent and eval suite (may take a few minutes)…", expanded=True) as status:
            st.write("Calling Groq (agent)…")
            trace = agent.run(query)
            status.write("Agent finished. Running evaluators (many judge API calls)…")
            runner = EvalRunner(tracer, verbose_eval=False, silent=True)
            results = runner.run_all_evals(
                trace,
                golden_answer=golden,
                optimal_tool_calls=optimal,
            )
            status.update(label="Done", state="complete")

        st.markdown("#### Trace summary")
        tc = sum(1 for s in trace.spans if s.get("span_type") == "tool_call")
        c1, c2, c3 = st.columns(3)
        c1.metric("Spans", len(trace.spans))
        c2.metric("Tool calls", tc)
        c3.metric("Trace ID", trace.trace_id[:12] + "…")

        with st.expander("Final answer", expanded=True):
            st.write(trace.final_answer or "(none)")

        rows_eval = []
        for r in results:
            thr = GATE_THRESHOLDS.get(r.eval_name, 0.0)
            gate_ok = r.score >= thr
            rows_eval.append(
                {
                    "eval": r.eval_name,
                    "layer": r.eval_layer,
                    "score": round(r.score, 4),
                    "pass": r.passed,
                    "meets_gate": gate_ok,
                    "gate_threshold": thr,
                    "type": r.eval_type,
                    "reason": r.reason,
                }
            )
        st.markdown("#### Evaluation details (with reasons)")
        st.dataframe(rows_eval, width="stretch", hide_index=True)

        buf = io.StringIO()
        w = csv.DictWriter(
            buf,
            fieldnames=list(rows_eval[0].keys()) if rows_eval else [],
        )
        if rows_eval:
            w.writeheader()
            w.writerows(rows_eval)
        csv_bytes = buf.getvalue().encode("utf-8")
        st.download_button(
            "Download eval results as CSV",
            data=csv_bytes,
            file_name=f"eval_{version_tag}_{trace.trace_id[:8]}.csv",
            mime="text/csv",
        )

    st.divider()
    st.caption(
        "Rate limits: judges retry on HTTP 429 with backoff. "
        "Optional: `GROQ_REQUEST_DELAY_MS=80` in `.env` if you hit rate limits."
    )


if __name__ == "__main__":
    main()
