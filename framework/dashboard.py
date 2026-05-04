"""
dashboard.py — Generates a self-contained HTML dashboard showing eval scores
across agent versions, layer breakdowns, and deployment gate status.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from framework.tracer import Tracer
from framework.runner import GATE_THRESHOLDS
from framework.score_history import load_history

REPORTS_DIR = Path(__file__).parent.parent / "reports"


def _mean_eval_score(scores_snapshot: Dict[str, Any]) -> float:
    """Average of per-eval avg_score fields from a suite snapshot."""
    vals = []
    for _name, data in scores_snapshot.items():
        if isinstance(data, dict) and "avg_score" in data:
            vals.append(float(data["avg_score"]))
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def generate_dashboard(
    versions: List[str],
    tracer: Optional[Tracer] = None,
    output_path: Optional[Path] = None,
) -> Path:
    tracer = tracer or Tracer()
    output_path = output_path or REPORTS_DIR / "dashboard.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Gather scores per version
    version_scores: Dict[str, Dict] = {}
    for v in versions:
        version_scores[v] = tracer.compute_version_scores(v)

    # All eval names in order
    layer_order = {"pre_tool": 0, "post_tool": 1, "e2e": 2}
    all_evals: List[str] = []
    for v, scores in version_scores.items():
        for name in scores:
            if name not in all_evals:
                all_evals.append(name)
    all_evals.sort(key=lambda n: (
        layer_order.get(
            next((s[n]["layer"] for s in version_scores.values() if n in s), "e2e"), 9
        ), n
    ))

    # Gate status per version
    gate_status: Dict[str, dict] = {}
    for v, scores in version_scores.items():
        failures = []
        for eval_name, threshold in GATE_THRESHOLDS.items():
            if eval_name not in scores:
                continue
            if scores[eval_name]["avg_score"] < threshold:
                failures.append(eval_name)
        gate_status[v] = {"passed": len(failures) == 0, "failures": failures}

    # Traces per version
    traces_count: Dict[str, int] = {}
    for v in versions:
        traces_count[v] = len(tracer.get_traces_by_version(v))

    # Layer avg per version
    layer_avgs: Dict[str, Dict[str, float]] = {}
    for v, scores in version_scores.items():
        avgs: Dict[str, List[float]] = {"pre_tool": [], "post_tool": [], "e2e": []}
        for name, data in scores.items():
            layer = data.get("layer", "e2e")
            if layer in avgs:
                avgs[layer].append(data["avg_score"])
        layer_avgs[v] = {
            layer: round(sum(vals) / len(vals), 3) if vals else 0.0
            for layer, vals in avgs.items()
        }

    history_rows = list(reversed(load_history(80)))

    html = _render_html(
        versions=versions,
        version_scores=version_scores,
        all_evals=all_evals,
        gate_status=gate_status,
        traces_count=traces_count,
        layer_avgs=layer_avgs,
        layer_order=layer_order,
        history_rows=history_rows,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Dashboard saved -> {output_path}")
    return output_path


def _score_color(score: float, threshold: float) -> str:
    if score >= threshold:
        return "#22c55e" if score >= 0.85 else "#84cc16"
    return "#ef4444" if score < threshold - 0.1 else "#f97316"


def _render_html(
    versions,
    version_scores,
    all_evals,
    gate_status,
    traces_count,
    layer_avgs,
    layer_order,
    history_rows,
) -> str:
    layer_colors = {
        "pre_tool": "#6366f1",
        "post_tool": "#0ea5e9",
        "e2e": "#f59e0b",
    }
    layer_labels = {
        "pre_tool": "Layer 1 · Pre-Tool",
        "post_tool": "Layer 2 · Post-Tool",
        "e2e": "Layer 3 · End-to-End",
    }

    # Build chart data for score history
    chart_datasets = []
    colors_cycle = ["#6366f1", "#0ea5e9", "#f59e0b", "#22c55e", "#ef4444", "#ec4899", "#14b8a6", "#a855f7", "#f97316", "#84cc16"]
    for i, eval_name in enumerate(all_evals):
        data_points = []
        for v in versions:
            s = version_scores.get(v, {}).get(eval_name)
            data_points.append(round(s["avg_score"], 3) if s else None)
        chart_datasets.append({
            "label": eval_name,
            "data": data_points,
            "borderColor": colors_cycle[i % len(colors_cycle)],
            "backgroundColor": colors_cycle[i % len(colors_cycle)] + "22",
            "tension": 0.3,
            "pointRadius": 5,
        })

    # Build table rows
    table_rows_html = ""
    current_layer = None
    for eval_name in all_evals:
        # Determine layer
        ev_layer = None
        for v in versions:
            s = version_scores.get(v, {}).get(eval_name)
            if s:
                ev_layer = s.get("layer")
                break
        if ev_layer is None:
            continue

        if ev_layer != current_layer:
            current_layer = ev_layer
            lcolor = layer_colors.get(ev_layer, "#888")
            table_rows_html += f"""
            <tr class="layer-header">
              <td colspan="{2 + len(versions) * 2}" style="background:{lcolor}22; color:{lcolor}; font-weight:700; padding:8px 16px; font-size:0.78rem; letter-spacing:0.08em; text-transform:uppercase;">
                {layer_labels.get(ev_layer, ev_layer)}
              </td>
            </tr>"""

        threshold = GATE_THRESHOLDS.get(eval_name, 0.0)
        cells = ""
        for v in versions:
            s = version_scores.get(v, {}).get(eval_name)
            if s:
                sc = s["avg_score"]
                color = _score_color(sc, threshold)
                bar_w = int(sc * 60)
                cells += f"""
                <td class="score-cell">
                  <div class="score-bar-wrap">
                    <div class="score-bar" style="width:{bar_w}px;background:{color}"></div>
                    <span style="color:{color};font-weight:700">{sc:.2f}</span>
                  </div>
                </td>
                <td class="passrate-cell">{s['pass_rate']:.0%} ({s['n']})</td>"""
            else:
                cells += "<td class='score-cell'>—</td><td class='passrate-cell'>—</td>"

        table_rows_html += f"""
        <tr>
          <td class="eval-name">{eval_name}</td>
          <td class="threshold-cell">{threshold:.2f}</td>
          {cells}
        </tr>"""

    # Gate cards
    gate_cards_html = ""
    for v in versions:
        gs = gate_status.get(v, {})
        passed = gs.get("passed", False)
        failures = gs.get("failures", [])
        trace_n = traces_count.get(v, 0)
        color = "#22c55e" if passed else "#ef4444"
        icon = "✅" if passed else "❌"
        status_label = "DEPLOYMENT READY" if passed else "DEPLOYMENT BLOCKED"
        fail_html = ""
        if failures:
            fail_html = "<ul class='fail-list'>" + "".join(f"<li>{f}</li>" for f in failures) + "</ul>"
        gate_cards_html += f"""
        <div class="gate-card" style="border-color:{color}22;background:{color}08">
          <div class="gate-version">Agent {v}</div>
          <div class="gate-icon">{icon}</div>
          <div class="gate-status" style="color:{color}">{status_label}</div>
          <div class="gate-meta">{trace_n} traces evaluated</div>
          {fail_html}
        </div>"""

    # Layer summary cards per version
    layer_summary_html = ""
    for v in versions:
        avgs = layer_avgs.get(v, {})
        cards = ""
        for layer in ["pre_tool", "post_tool", "e2e"]:
            avg = avgs.get(layer, 0.0)
            lc = layer_colors[layer]
            bar = int(avg * 100)
            cards += f"""
            <div class="layer-mini-card">
              <div class="lmc-label" style="color:{lc}">{layer_labels[layer]}</div>
              <div class="lmc-score" style="color:{lc}">{avg:.2f}</div>
              <div class="lmc-bar-bg"><div class="lmc-bar-fill" style="width:{bar}%;background:{lc}"></div></div>
            </div>"""
        layer_summary_html += f"""
        <div class="version-layer-block">
          <div class="vlb-version">Agent {v}</div>
          {cards}
        </div>"""

    # Chart data JSON
    chart_data_json = json.dumps({
        "labels": versions,
        "datasets": chart_datasets,
    })

    # Layer radar data
    radar_datasets = []
    radar_colors = ["#6366f1", "#0ea5e9", "#f59e0b", "#22c55e"]
    for i, v in enumerate(versions):
        avgs = layer_avgs.get(v, {})
        radar_datasets.append({
            "label": f"Agent {v}",
            "data": [
                avgs.get("pre_tool", 0),
                avgs.get("post_tool", 0),
                avgs.get("e2e", 0),
            ],
            "borderColor": radar_colors[i % len(radar_colors)],
            "backgroundColor": radar_colors[i % len(radar_colors)] + "33",
        })

    radar_data_json = json.dumps({
        "labels": ["Pre-Tool", "Post-Tool", "E2E"],
        "datasets": radar_datasets,
    })

    # Score run history (append-only suite runs)
    if history_rows:
        hist_body = ""
        for r in history_rows[:25]:
            ts = r.get("ts", "")[:19].replace("T", " ")
            ver = r.get("agent_version", "")
            gp = r.get("gate_passed", False)
            gate_html = "<span style='color:#22c55e'>PASS</span>" if gp else "<span style='color:#ef4444'>BLOCK</span>"
            mean_s = _mean_eval_score(r.get("scores") or {})
            hist_body += f"""<tr><td style='font-family:Space Mono,monospace;font-size:0.78rem;color:#64748b'>{ts}</td>
<td>{ver}</td><td>{gate_html}</td><td style='font-family:Space Mono,monospace'>{mean_s:.2f}</td></tr>"""
        history_section_html = f"""
  <div class="section">
    <div class="section-title">Recent Eval Suite Runs (score history)</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Timestamp (UTC)</th><th>Agent version</th><th>Gate</th><th>Mean eval score</th></tr></thead>
        <tbody>{hist_body}</tbody>
      </table>
    </div>
    <p style="margin-top:12px;color:#64748b;font-size:0.78rem;font-family:Space Mono,monospace">Appended on each <code>run_suite</code> / demo — use for release-to-release comparisons.</p>
  </div>"""
    else:
        history_section_html = ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent Eval Framework — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0a0d14;
    --surface: #111827;
    --surface2: #1a2035;
    --border: #1e2d45;
    --text: #e2e8f0;
    --muted: #64748b;
    --accent: #6366f1;
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #f59e0b;
    --cyan: #0ea5e9;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
    font-size: 14px;
  }}
  /* Header */
  .header {{
    background: linear-gradient(135deg, #0f1729 0%, #1a0a2e 50%, #0a1a2e 100%);
    border-bottom: 1px solid var(--border);
    padding: 32px 48px 28px;
    position: relative;
    overflow: hidden;
  }}
  .header::before {{
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, #6366f133, transparent 70%);
    pointer-events: none;
  }}
  .header-top {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
  }}
  .header h1 {{
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #fff;
  }}
  .header h1 span {{ color: var(--accent); }}
  .header-sub {{
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 6px;
    font-family: 'Space Mono', monospace;
  }}
  .badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
  }}
  .badge-versions {{
    background: var(--accent)22;
    color: var(--accent);
    border: 1px solid var(--accent)44;
  }}
  .badge-evals {{
    background: #0ea5e922;
    color: var(--cyan);
    border: 1px solid #0ea5e944;
  }}
  .badges {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }}

  /* Layout */
  .container {{ max-width: 1400px; margin: 0 auto; padding: 0 32px 60px; }}
  .section {{ margin-top: 40px; }}
  .section-title {{
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}

  /* Gate cards */
  .gate-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
  }}
  .gate-card {{
    background: var(--surface);
    border: 1px solid;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
  }}
  .gate-version {{
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}
  .gate-icon {{ font-size: 2.4rem; margin: 12px 0 8px; }}
  .gate-status {{
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.04em;
  }}
  .gate-meta {{ color: var(--muted); font-size: 0.78rem; margin-top: 8px; }}
  .fail-list {{
    text-align: left;
    margin-top: 12px;
    padding-left: 16px;
    color: #ef4444;
    font-size: 0.75rem;
    list-style: disc;
    line-height: 1.8;
  }}

  /* Layer summary */
  .layer-summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
  }}
  .version-layer-block {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }}
  .vlb-version {{
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 16px;
  }}
  .layer-mini-card {{ margin-bottom: 14px; }}
  .lmc-label {{ font-size: 0.78rem; font-weight: 600; margin-bottom: 4px; }}
  .lmc-score {{
    font-family: 'Space Mono', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 6px;
  }}
  .lmc-bar-bg {{
    height: 6px;
    background: var(--border);
    border-radius: 99px;
    overflow: hidden;
  }}
  .lmc-bar-fill {{ height: 100%; border-radius: 99px; transition: width 0.6s ease; }}

  /* Charts */
  .chart-grid {{
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 24px;
  }}
  @media (max-width: 900px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
  .chart-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
  }}
  .chart-card h3 {{
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 20px;
  }}

  /* Score table */
  .table-wrap {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{
    background: var(--surface2);
    padding: 12px 16px;
    text-align: left;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  td {{
    padding: 10px 16px;
    border-bottom: 1px solid var(--border)88;
    vertical-align: middle;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: var(--surface2); }}
  .eval-name {{
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--text);
  }}
  .threshold-cell {{
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
  }}
  .score-cell {{ min-width: 100px; }}
  .score-bar-wrap {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .score-bar {{
    height: 6px;
    border-radius: 99px;
    min-width: 4px;
  }}
  .passrate-cell {{
    font-size: 0.75rem;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    white-space: nowrap;
  }}

  /* Footer */
  .footer {{
    margin-top: 60px;
    padding: 24px 48px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-top">
    <div>
      <h1>Agent <span>Eval</span> Framework</h1>
      <div class="header-sub">Multi-layer evaluation dashboard · Groq-powered judges</div>
      <div class="badges">
        <span class="badge badge-versions">🔢 {len(versions)} versions compared</span>
        <span class="badge badge-evals">⚡ {len(all_evals)} evals · 3 layers</span>
      </div>
    </div>
  </div>
</div>

<div class="container">

  {history_section_html}

  <!-- Deployment Gate -->
  <div class="section">
    <div class="section-title">Deployment Gate Status</div>
    <div class="gate-grid">
      {gate_cards_html}
    </div>
  </div>

  <!-- Layer Averages -->
  <div class="section">
    <div class="section-title">Layer Score Averages by Version</div>
    <div class="layer-summary-grid">
      {layer_summary_html}
    </div>
  </div>

  <!-- Charts -->
  <div class="section">
    <div class="section-title">Score History &amp; Layer Radar</div>
    <div class="chart-grid">
      <div class="chart-card">
        <h3>Eval Score History Across Versions</h3>
        <canvas id="lineChart" height="280"></canvas>
      </div>
      <div class="chart-card">
        <h3>Layer Radar</h3>
        <canvas id="radarChart" height="280"></canvas>
      </div>
    </div>
  </div>

  <!-- Full Score Table -->
  <div class="section">
    <div class="section-title">Full Eval Score Table</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Eval Name</th>
            <th>Gate Threshold</th>
            {"".join(f"<th>Agent {v} Score</th><th>Agent {v} Pass%</th>" for v in versions)}
          </tr>
        </thead>
        <tbody>
          {table_rows_html}
        </tbody>
      </table>
    </div>
  </div>

</div>

<div class="footer">
  <span>Agent Eval Framework · JSON-backed · Groq judge</span>
  <span>Generated automatically · {len(versions)} versions · {len(all_evals)} evals</span>
</div>

<script>
const chartData = {chart_data_json};
const radarData = {radar_data_json};

// Line chart
new Chart(document.getElementById('lineChart'), {{
  type: 'line',
  data: chartData,
  options: {{
    responsive: true,
    plugins: {{
      legend: {{
        position: 'bottom',
        labels: {{ color: '#94a3b8', font: {{ size: 10, family: 'Space Mono' }}, boxWidth: 12 }}
      }},
      tooltip: {{ mode: 'index', intersect: false }}
    }},
    scales: {{
      y: {{
        min: 0, max: 1,
        ticks: {{ color: '#64748b', font: {{ size: 10, family: 'Space Mono' }}, stepSize: 0.2 }},
        grid: {{ color: '#1e2d4588' }}
      }},
      x: {{
        ticks: {{ color: '#64748b', font: {{ size: 10, family: 'Space Mono' }} }},
        grid: {{ color: '#1e2d4544' }}
      }}
    }}
  }}
}});

// Radar chart
new Chart(document.getElementById('radarChart'), {{
  type: 'radar',
  data: radarData,
  options: {{
    responsive: true,
    scales: {{
      r: {{
        min: 0, max: 1,
        ticks: {{ color: '#64748b', font: {{ size: 9 }}, stepSize: 0.25, backdropColor: 'transparent' }},
        grid: {{ color: '#1e2d4588' }},
        pointLabels: {{ color: '#94a3b8', font: {{ size: 11, family: 'DM Sans' }} }}
      }}
    }},
    plugins: {{
      legend: {{
        labels: {{ color: '#94a3b8', font: {{ size: 11, family: 'DM Sans' }}, boxWidth: 14 }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""
