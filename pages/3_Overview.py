"""pages/3_Overview.py — Analytics-style overview dashboard."""

import json
from datetime import datetime
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
st.set_page_config(page_title="Overview", page_icon="🏠", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.top-bar { display:flex; align-items:center; justify-content:space-between;
           padding:0 0 20px; border-bottom:1px solid #21262d; margin-bottom:24px; }
.top-title { font-size:1.3rem; font-weight:600; color:#e6edf3; }
.top-sub   { font-size:0.8rem; color:#6e7681; margin-top:2px; }
.section-label { font-size:0.72rem; font-weight:600; color:#6e7681;
                 text-transform:uppercase; letter-spacing:0.1em;
                 margin-bottom:12px; margin-top:4px; }
.kpi-grid { display:grid; grid-template-columns:repeat(6,1fr); gap:10px; margin-bottom:24px; }
.kpi-card { background:#0d1117; border:1px solid #21262d; border-radius:8px;
            padding:14px 14px; position:relative; overflow:hidden; }
.kpi-card::before { content:''; position:absolute; top:0; left:0; right:0;
                    height:2px; border-radius:8px 8px 0 0; }
.kpi-card.c1::before { background:#58a6ff; }
.kpi-card.c2::before { background:#3fb950; }
.kpi-card.c3::before { background:#f85149; }
.kpi-card.c4::before { background:#d29922; }
.kpi-card.c5::before { background:#bc8cff; }
.kpi-card.c6::before { background:#79c0ff; }
.kpi-label { font-size:0.67rem; color:#6e7681; text-transform:uppercase;
             letter-spacing:0.07em; margin-bottom:4px; }
.kpi-value { font-size:1.5rem; font-weight:600; color:#e6edf3;
             font-family:'JetBrains Mono',monospace; }
.kpi-sub   { font-size:0.68rem; color:#6e7681; margin-top:4px; }

.exp-row { display:flex; justify-content:space-between; align-items:center;
           padding:8px 0; border-bottom:1px solid #161b22; }
.exp-name { font-size:0.83rem; color:#e6edf3; }
.exp-time { font-size:0.72rem; color:#6e7681; font-family:'JetBrains Mono',monospace; }
.exp-badge { display:inline-block; padding:1px 6px; border-radius:4px; font-size:0.65rem;
             background:#161b22; color:#8b949e; border:1px solid #21262d;
             font-family:'JetBrains Mono',monospace; margin-left:6px; }

.safety-row { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-bottom:20px; }
.safety-card { background:#0d1117; border:1px solid #21262d; border-radius:8px;
               padding:14px 16px; text-align:center; }
.safety-icon { font-size:1.4rem; margin-bottom:4px; }
.safety-label { font-size:0.68rem; color:#6e7681; text-transform:uppercase;
                letter-spacing:0.07em; }
.safety-count { font-size:1.4rem; font-weight:600; font-family:'JetBrains Mono',monospace; }
.safe-ok   { color:#3fb950; }
.safe-warn { color:#f85149; }
.eb { text-align:center; padding:32px 20px; color:#6e7681; font-size:0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)

now = datetime.now().strftime("%d %b %Y, %H:%M")
st.markdown(
    f"""
<div class="top-bar">
    <div>
        <div class="top-title">🏠 Overview</div>
        <div class="top-sub">Aggregated stats across all evaluators</div>
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#6e7681;
                background:#161b22;padding:4px 10px;border-radius:6px;border:1px solid #21262d">
        {now}
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Collect all data ──────────────────────────────────────────────────────────
json_results = []
for fname in ["evaluation_results.json", "evaluation_history.json"]:
    p = ROOT / fname
    if p.exists():
        try:
            data = json.loads(p.read_text())
            json_results.extend(data if isinstance(data, list) else [data])
        except Exception:
            pass


def unwrap(r: dict) -> dict:
    return (
        {**r["EvaluationReport"], **{k: v for k, v in r.items() if k != "EvaluationReport"}}
        if "EvaluationReport" in r and isinstance(r["EvaluationReport"], dict)
        else r
    )


json_runs = len(json_results)
items = [unwrap(r) for r in json_results]
json_pass = sum(1 for r in items if str(r.get("pass_fail", "")).lower() == "pass")
json_fail = sum(1 for r in items if str(r.get("pass_fail", "")).lower() == "fail")
pass_rate = round(json_pass / json_runs * 100) if json_runs else 0
halluc = sum(1 for r in items if r.get("hallucination_detected"))
bias = sum(1 for r in items if r.get("bias_detected"))
toxic = sum(1 for r in items if r.get("toxicity_detected"))

try:
    from sqlite_store import get_all_metrics_df, get_experiments, get_leaderboard

    lb = get_leaderboard()
    exps = get_experiments()
    df = get_all_metrics_df()
    db_models = len(lb)
    db_responses = len(df)
    avg_score = (
        round(df["judge_score"].mean(), 1)
        if df is not None and not df.empty and "judge_score" in df.columns
        else "—"
    )
except Exception:
    lb = exps = []
    df = None
    db_models = db_responses = 0
    avg_score = "—"

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="kpi-grid">
    <div class="kpi-card c1">
        <div class="kpi-label">Total Runs</div>
        <div class="kpi-value">{json_runs}</div>
        <div class="kpi-sub">Ollama + CrewAI</div>
    </div>
    <div class="kpi-card c2">
        <div class="kpi-label">Passed</div>
        <div class="kpi-value" style="color:#3fb950">{json_pass}</div>
        <div class="kpi-sub">{pass_rate}% pass rate</div>
    </div>
    <div class="kpi-card c3">
        <div class="kpi-label">Failed</div>
        <div class="kpi-value" style="color:#f85149">{json_fail}</div>
        <div class="kpi-sub">{100 - pass_rate}% fail rate</div>
    </div>
    <div class="kpi-card c4">
        <div class="kpi-label">Pipeline Models</div>
        <div class="kpi-value">{db_models}</div>
        <div class="kpi-sub">{db_responses} responses</div>
    </div>
    <div class="kpi-card c5">
        <div class="kpi-label">Avg Judge Score</div>
        <div class="kpi-value">{avg_score}</div>
        <div class="kpi-sub">Pipeline only</div>
    </div>
    <div class="kpi-card c6">
        <div class="kpi-label">Experiments</div>
        <div class="kpi-value">{len(exps)}</div>
        <div class="kpi-sub">Pipeline runs</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Main layout ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    # Pass/fail trend
    st.markdown('<div class="section-label">Pass / Fail Trend</div>', unsafe_allow_html=True)
    if json_runs > 0:
        import pandas as pd

        trend_data = []
        for i, r in enumerate(json_results):
            u = unwrap(r)
            pf = str(u.get("pass_fail", "UNKNOWN")).upper()
            ts = r.get("timestamp", "")[:10]
            trend_data.append({"Run": i + 1, "Result": pf, "Date": ts})
        tdf = pd.DataFrame(trend_data)

        # Cumulative pass rate
        tdf["Pass"] = (tdf["Result"] == "PASS").astype(int)
        tdf["CumPassRate"] = (tdf["Pass"].cumsum() / (tdf.index + 1) * 100).round(1)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=tdf["Run"],
                y=tdf["CumPassRate"],
                mode="lines+markers",
                line=dict(color="#3fb950", width=2),
                marker=dict(size=6, color="#3fb950"),
                fill="tozeroy",
                fillcolor="rgba(63,185,80,0.08)",
                name="Pass Rate %",
            )
        )
        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="#d29922",
            annotation_text="70% target",
            annotation_position="right",
        )
        fig.update_layout(
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font_color="#8b949e",
            margin=dict(t=8, b=0, l=0, r=0),
            height=200,
            xaxis=dict(gridcolor="#21262d", title="Run #"),
            yaxis=dict(gridcolor="#21262d", title="Pass Rate %", range=[0, 105]),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="eb">No data yet.</div>', unsafe_allow_html=True)

    # Pipeline model scores
    st.markdown(
        '<div class="section-label" style="margin-top:20px">Pipeline Model Scores</div>',
        unsafe_allow_html=True,
    )
    if lb:
        import pandas as pd

        ldf = pd.DataFrame(lb)
        avail = [
            c for c in ["avg_judge_score", "avg_clarity", "avg_completeness"] if c in ldf.columns
        ]
        if avail:
            fig2 = px.bar(
                ldf,
                x="name",
                y=avail,
                barmode="group",
                color_discrete_sequence=["#58a6ff", "#3fb950", "#d29922"],
                labels={"name": "Model", "value": "Score", "variable": "Metric"},
            )
            fig2.update_layout(
                plot_bgcolor="#0d1117",
                paper_bgcolor="#0d1117",
                font_color="#8b949e",
                legend=dict(orientation="h", y=1.1, x=0),
                margin=dict(t=24, b=0, l=0, r=0),
                height=200,
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.markdown('<div class="eb">No pipeline data yet.</div>', unsafe_allow_html=True)

with col_right:
    # Safety overview
    st.markdown('<div class="section-label">Safety Overview</div>', unsafe_allow_html=True)
    h_cls = "safe-warn" if halluc > 0 else "safe-ok"
    b_cls = "safe-warn" if bias > 0 else "safe-ok"
    t_cls = "safe-warn" if toxic > 0 else "safe-ok"
    st.markdown(
        f"""
    <div class="safety-row">
        <div class="safety-card">
            <div class="safety-icon">🧠</div>
            <div class="safety-label">Hallucination</div>
            <div class="safety-count {h_cls}">{halluc}</div>
        </div>
        <div class="safety-card">
            <div class="safety-icon">⚖️</div>
            <div class="safety-label">Bias</div>
            <div class="safety-count {b_cls}">{bias}</div>
        </div>
        <div class="safety-card">
            <div class="safety-icon">☣️</div>
            <div class="safety-label">Toxicity</div>
            <div class="safety-count {t_cls}">{toxic}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Failure mode breakdown
    st.markdown(
        '<div class="section-label" style="margin-top:4px">Failure Mode Breakdown</div>',
        unsafe_allow_html=True,
    )
    fm_counts = {}
    for r in items:
        fm = r.get("failure_mode") or "none"
        fm_counts[fm] = fm_counts.get(fm, 0) + 1

    if fm_counts:
        import pandas as pd

        fdf = pd.DataFrame(fm_counts.items(), columns=["Mode", "Count"]).sort_values(
            "Count", ascending=False
        )
        fig3 = px.bar(
            fdf, x="Count", y="Mode", orientation="h", color="Count", color_continuous_scale="Blues"
        )
        fig3.update_layout(
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font_color="#8b949e",
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(t=4, b=0, l=0, r=0),
            height=180,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", title=""),
        )
        fig3.update_traces(marker_line_width=0)
        st.plotly_chart(fig3, use_container_width=True)

    # Recent experiments
    st.markdown(
        '<div class="section-label" style="margin-top:8px">Recent Experiments</div>',
        unsafe_allow_html=True,
    )
    if exps:
        for e in exps[:6]:
            name = e.get("name", "unnamed")
            ts = str(e.get("created_at", ""))[:16].replace("T", " ")
            st.markdown(
                f"""
            <div class="exp-row">
                <span class="exp-name">🧪 {name}</span>
                <span class="exp-time">{ts}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="eb">No experiments yet.</div>', unsafe_allow_html=True)
