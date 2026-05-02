"""pages/3_Overview.py — Aggregated stats from all evaluators."""

import json
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent.parent
st.set_page_config(page_title="Overview", page_icon="🏠", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.ph { font-size:1.6rem; font-weight:700; color:#e6edf3; margin-bottom:4px; }
.ps { color:#6e7681; font-size:0.88rem; margin-bottom:28px; }
.sc { background:#0d1117; border:1px solid #21262d; border-radius:10px; padding:18px; text-align:center; }
.sv { font-size:2.2rem; font-weight:700; color:#58a6ff; font-family:'JetBrains Mono',monospace; }
.sl { font-size:0.72rem; color:#6e7681; text-transform:uppercase; letter-spacing:0.08em; margin-top:3px; }
.sec { font-weight:600; color:#e6edf3; font-size:1rem; margin:24px 0 10px; }
.er { display:flex; justify-content:space-between; padding:9px 0; border-bottom:1px solid #21262d; }
.en { color:#e6edf3; font-size:0.85rem; }
.et { color:#6e7681; font-family:'JetBrains Mono',monospace; font-size:0.72rem; }
.eh { color:#6e7681; font-size:0.85rem; text-align:center; padding:28px 0; }
.source-tag { display:inline-block; padding:1px 7px; border-radius:20px; font-size:0.65rem;
              font-family:'JetBrains Mono',monospace; margin-right:4px; }
.st-g { background:#0d2818; color:#3fb950; border:1px solid #238636; }
.st-b { background:#0c1f3f; color:#58a6ff; border:1px solid #1f6feb; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="ph">🏠 Overview</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ps">Aggregated stats across all evaluators — JSON-based and SQLite pipeline.</div>',
    unsafe_allow_html=True,
)

# ── Collect stats from both sources ──────────────────────────────────────────
json_runs, json_pass, json_fail = 0, 0, 0
for fname in ["evaluation_results.json", "evaluation_history.json"]:
    p = ROOT / fname
    if p.exists():
        try:
            data = json.loads(p.read_text())
            items = data if isinstance(data, list) else [data]
            json_runs += len(items)
            for item in items:
                r = item.get("EvaluationReport", item)
                pf = str(r.get("pass_fail", "")).lower()
                if pf == "pass":
                    json_pass += 1
                elif pf == "fail":
                    json_fail += 1
        except Exception:
            pass

try:
    from sqlite_store import get_all_metrics_df, get_experiments, get_leaderboard

    lb = get_leaderboard()
    exps = get_experiments()
    df = get_all_metrics_df()
    db_models = len(lb)
    db_responses = len(df)
    db_exps = len(exps)
    avg_score = (
        f"{df['judge_score'].mean():.2f}" if not df.empty and "judge_score" in df.columns else "—"
    )
except Exception:
    lb = exps = []
    df = None
    db_models = db_responses = db_exps = 0
    avg_score = "—"

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
for col, val, lbl in [
    (c1, json_runs, "Total Eval Runs"),
    (c2, json_pass, "Passed"),
    (c3, json_fail, "Failed"),
    (c4, db_models, "Pipeline Models"),
    (c5, db_responses, "Pipeline Responses"),
    (c6, avg_score, "Avg Judge Score"),
]:
    col.markdown(
        f'<div class="sc"><div class="sv">{val}</div><div class="sl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

col_a, col_b = st.columns([3, 2], gap="large")

with col_a:
    st.markdown('<div class="sec">📈 Pipeline Model Performance</div>', unsafe_allow_html=True)
    if lb:
        import pandas as pd

        ldf = pd.DataFrame(lb)
        cols = [
            c for c in ["avg_judge_score", "avg_clarity", "avg_completeness"] if c in ldf.columns
        ]
        if cols:
            fig = px.bar(
                ldf,
                x="name",
                y=cols,
                barmode="group",
                labels={"name": "Model", "value": "Score", "variable": "Metric"},
                color_discrete_sequence=["#58a6ff", "#3fb950", "#d29922"],
            )
            fig.update_layout(
                plot_bgcolor="#0d1117",
                paper_bgcolor="#0d1117",
                font_color="#8b949e",
                legend_title_text="",
                margin=dict(t=10, b=0, l=0, r=0),
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(
            '<div class="eh">No pipeline data yet.<br>Run the Professional Pipeline to see model comparisons.</div>',
            unsafe_allow_html=True,
        )

    # JSON pass/fail pie if we have data
    if json_runs > 0 and (json_pass + json_fail) > 0:
        st.markdown('<div class="sec">🟢🔴 Ollama + CrewAI Pass/Fail</div>', unsafe_allow_html=True)
        fig2 = px.pie(
            values=[json_pass, json_fail, json_runs - json_pass - json_fail],
            names=["Pass", "Fail", "Unknown"],
            color_discrete_sequence=["#3fb950", "#f85149", "#8b949e"],
            hole=0.55,
        )
        fig2.update_layout(
            paper_bgcolor="#0d1117",
            font_color="#8b949e",
            margin=dict(t=10, b=0, l=0, r=0),
            height=220,
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
        )
        fig2.update_traces(textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.markdown('<div class="sec">🕐 Recent Experiments</div>', unsafe_allow_html=True)
    if exps:
        for e in exps[:8]:
            name = e.get("name", "unnamed")
            ts = str(e.get("created_at", ""))[:16].replace("T", " ")
            st.markdown(
                f'<div class="er"><span class="en">🧪 {name}</span><span class="et">{ts}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="eh">No experiments yet.</div>', unsafe_allow_html=True)

    if df is not None and not df.empty and "judge_score" in df.columns:
        st.markdown('<div class="sec">📊 Score Distribution</div>', unsafe_allow_html=True)
        fig3 = px.histogram(
            df,
            x="judge_score",
            nbins=15,
            color_discrete_sequence=["#58a6ff"],
            labels={"judge_score": "Judge Score"},
        )
        fig3.update_layout(
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font_color="#8b949e",
            showlegend=False,
            margin=dict(t=4, b=0, l=0, r=0),
            height=180,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", title="Count"),
        )
        st.plotly_chart(fig3, use_container_width=True)
