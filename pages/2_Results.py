"""
pages/2_Results.py — Evaluation Results Viewer
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Results", page_icon="🔍", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.page-header { font-family:'Sora',sans-serif; font-size:1.8rem; font-weight:700; color:#e6edf3; margin-bottom:4px; }
.page-sub    { color:#6e7681; font-size:0.9rem; margin-bottom:28px; }

.result-card {
    background:#0d1117; border:1px solid #21262d; border-radius:10px;
    padding:20px 20px 16px; margin-bottom:16px;
}
.kpi-label { font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#6e7681; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:4px; }
.kpi-value { font-size:1.4rem; font-weight:700; color:#e6edf3; }
.kpi-pass  { color:#3fb950; }
.kpi-fail  { color:#f85149; }
.kpi-unknown { color:#8b949e; }

.safety-ok   { color:#3fb950; font-weight:600; }
.safety-flag { color:#f85149; font-weight:600; }

.empty-state {
    text-align:center; padding:60px 20px;
    color:#6e7681; font-size:0.95rem;
}
.empty-icon { font-size:2.5rem; margin-bottom:12px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="page-header">🔍 Evaluation Results</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Browse results from Ollama, CrewAI, and the professional pipeline.</div>',
    unsafe_allow_html=True,
)


def load_results() -> list:
    results = []
    for fname in ["evaluation_results.json", "evaluation_history.json"]:
        p = Path(fname)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                if isinstance(data, dict):
                    data = [data]
                results.extend(data)
            except Exception:
                pass
    return results


def pass_fail_class(val: str) -> str:
    v = str(val).lower()
    if v in ("pass", "true", "approved"):
        return "kpi-pass"
    if v in ("fail", "false", "rejected"):
        return "kpi-fail"
    return "kpi-unknown"


def render_result(r: dict) -> None:
    pf = str(r.get("pass_fail", "UNKNOWN")).upper()
    rd = str(r.get("release_decision", "—")).upper()
    fm = r.get("failure_mode") or "none"
    tc = r.get("test_case_id", "N/A")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="kpi-label">Test Case</div><div class="kpi-value">{tc}</div>',
            unsafe_allow_html=True,
        )
    with c2:
        cls = pass_fail_class(pf)
        st.markdown(
            f'<div class="kpi-label">Pass / Fail</div><div class="kpi-value {cls}">{pf}</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi-label">Release Decision</div><div class="kpi-value">{rd}</div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="kpi-label">Failure Mode</div><div class="kpi-value" style="font-size:1rem;padding-top:4px">{fm}</div>',
            unsafe_allow_html=True,
        )

    metrics = r.get("metrics", {})
    if metrics and isinstance(metrics, dict):
        numeric = {k: v for k, v in metrics.items() if isinstance(v, int | float)}
        if numeric:
            df = pd.DataFrame(numeric.items(), columns=["Metric", "Value"])
            fig = px.bar(
                df,
                x="Metric",
                y="Value",
                color="Value",
                color_continuous_scale="Blues",
                title="Metric Scores",
            )
            fig.update_layout(
                plot_bgcolor="#0d1117",
                paper_bgcolor="#0d1117",
                font_color="#8b949e",
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(t=40, b=0, l=0, r=0),
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Safety signals**")
    sc1, sc2, sc3 = st.columns(3)

    def safety_icon(flag: bool) -> str:
        return (
            '<span class="safety-flag">⚠ Detected</span>'
            if flag
            else '<span class="safety-ok">✓ Clean</span>'
        )

    sc1.markdown(
        f"Hallucination &nbsp; {safety_icon(r.get('hallucination_detected', False))}",
        unsafe_allow_html=True,
    )
    sc2.markdown(
        f"Bias &nbsp; {safety_icon(r.get('bias_detected', False))}", unsafe_allow_html=True
    )
    sc3.markdown(
        f"Toxicity &nbsp; {safety_icon(r.get('toxicity_detected', False))}", unsafe_allow_html=True
    )


all_results = load_results()

tab1, tab2, tab3 = st.tabs(["🟢 Ollama", "🔵 CrewAI", "🔴 Professional Pipeline"])

with tab1:
    res = [r for r in all_results if "ollama" in str(r).lower() or r.get("source") == "ollama"]
    if not res:
        st.markdown(
            '<div class="empty-state"><div class="empty-icon">📭</div>No Ollama results yet.<br>Run the Ollama Evaluator from the Launch page.</div>',
            unsafe_allow_html=True,
        )
    else:
        options = [f"{r.get('test_case_id', 'N/A')} · {r.get('timestamp', '')}" for r in res]
        sel = st.selectbox("Select run", options, key="ollama_sel")
        current = res[options.index(sel)]
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        render_result(current)
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    res = [r for r in all_results if "crew" in str(r).lower() or r.get("source") == "crew"]
    if not res:
        st.markdown(
            '<div class="empty-state"><div class="empty-icon">📭</div>No CrewAI results yet.<br>Run the CrewAI Evaluator from the Launch page.</div>',
            unsafe_allow_html=True,
        )
    else:
        options = [f"{r.get('test_case_id', 'N/A')} · {r.get('timestamp', '')}" for r in res]
        sel = st.selectbox("Select run", options, key="crew_sel")
        current = res[options.index(sel)]
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        render_result(current)
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown(
        '<div class="empty-state"><div class="empty-icon">🗄</div>Professional pipeline results live in<br><strong>Leaderboard · Responses · Pairwise · Metrics</strong></div>',
        unsafe_allow_html=True,
    )
