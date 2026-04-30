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

.page-header { font-size:1.6rem; font-weight:700; color:#e6edf3; margin-bottom:4px; }
.page-sub    { color:#6e7681; font-size:0.88rem; margin-bottom:24px; }

.kpi-block  { background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:14px 16px; }
.kpi-label  { font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#6e7681; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:4px; }
.kpi-value  { font-size:1.5rem; font-weight:700; color:#e6edf3; }
.kpi-pass   { color:#3fb950; }
.kpi-fail   { color:#f85149; }

.safety-ok   { color:#3fb950; font-weight:600; font-size:0.9rem; }
.safety-flag { color:#f85149; font-weight:600; font-size:0.9rem; }

.recs-item  { padding:4px 0; font-size:0.85rem; color:#8b949e; border-bottom:1px solid #21262d; }

.empty-box  { text-align:center; padding:48px 20px; color:#6e7681; }
.empty-icon { font-size:2rem; margin-bottom:8px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="page-header">🔍 Evaluation Results</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Latest results from each evaluator. Re-run an evaluator to refresh.</div>',
    unsafe_allow_html=True,
)


ROOT = Path(__file__).parent.parent


def load_results() -> list:
    results = []
    for fname in ["evaluation_results.json", "evaluation_history.json"]:
        p = ROOT / fname
        if p.exists():
            try:
                data = json.loads(p.read_text())
                if isinstance(data, dict):
                    data = [data]
                results.extend(data)
            except Exception:
                pass
    return results


def pf_class(val: str) -> str:
    v = str(val).lower()
    if v in ("pass", "approved", "true"):
        return "kpi-pass"
    if v in ("fail", "rejected", "false"):
        return "kpi-fail"
    return ""


def render_result(r: dict) -> None:
    pf = str(r.get("pass_fail", "UNKNOWN")).upper()
    rd = str(r.get("release_decision", "—"))
    fm = r.get("failure_mode") or "none"
    tc = r.get("test_case_id", "N/A")
    ts = r.get("timestamp", "")[:19].replace("T", " ") if r.get("timestamp") else ""

    if ts:
        st.caption(f"🕐 {ts}")

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, extra_cls in [
        (c1, "Test Case", tc, ""),
        (c2, "Pass / Fail", pf, pf_class(pf)),
        (c3, "Release Decision", rd, ""),
        (c4, "Failure Mode", fm, ""),
    ]:
        col.markdown(
            f'<div class="kpi-block"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value {extra_cls}">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Safety row
    sc1, sc2, sc3 = st.columns(3)

    def sfmt(flag: bool, label: str) -> str:
        if flag:
            return f'<span class="safety-flag">⚠ {label} detected</span>'
        return f'<span class="safety-ok">✓ No {label.lower()}</span>'

    sc1.markdown(
        sfmt(r.get("hallucination_detected", False), "Hallucination"), unsafe_allow_html=True
    )
    sc2.markdown(sfmt(r.get("bias_detected", False), "Bias"), unsafe_allow_html=True)
    sc3.markdown(sfmt(r.get("toxicity_detected", False), "Toxicity"), unsafe_allow_html=True)

    # Metrics chart
    metrics = r.get("metrics", {})
    if isinstance(metrics, dict):
        numeric = {k: v for k, v in metrics.items() if isinstance(v, int | float)}
        if numeric:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            df = pd.DataFrame(numeric.items(), columns=["Metric", "Value"])
            fig = px.bar(
                df,
                x="Metric",
                y="Value",
                color="Value",
                color_continuous_scale="Blues",
            )
            fig.update_layout(
                plot_bgcolor="#0d1117",
                paper_bgcolor="#0d1117",
                font_color="#8b949e",
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(t=8, b=0, l=0, r=0),
                height=220,
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    recs = r.get("recommendations", [])
    if recs:
        st.markdown("**Recommendations**")
        for rec in recs if isinstance(recs, list) else [recs]:
            st.markdown(f'<div class="recs-item">→ {rec}</div>', unsafe_allow_html=True)

    # Bottlenecks / regressions
    b1, b2 = st.columns(2)
    bots = r.get("top_bottlenecks", [])
    regs = r.get("top_regressions", [])
    if bots:
        with b1:
            st.markdown("**Top Bottlenecks**")
            for b in bots if isinstance(bots, list) else [bots]:
                st.markdown(f'<div class="recs-item">🐢 {b}</div>', unsafe_allow_html=True)
    if regs:
        with b2:
            st.markdown("**Top Regressions**")
            for reg in regs if isinstance(regs, list) else [regs]:
                st.markdown(f'<div class="recs-item">📉 {reg}</div>', unsafe_allow_html=True)


all_results = load_results()

tab1, tab2, tab3 = st.tabs(["🟢 Ollama", "🔵 CrewAI", "🔴 Professional Pipeline"])

with tab1:
    res = [r for r in all_results if "ollama" in str(r).lower()]
    if not res:
        st.markdown(
            '<div class="empty-box"><div class="empty-icon">📭</div>No Ollama results yet.<br>Run the Ollama Evaluator from the Launch page.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"{len(res)} run(s) found")
        options = [
            f"Run {i + 1} · {r.get('test_case_id', '?')} · {str(r.get('timestamp', ''))[:16]}"
            for i, r in enumerate(res)
        ]
        idx = st.selectbox(
            "Select run", range(len(options)), format_func=lambda i: options[i], key="ollama_sel"
        )
        st.divider()
        render_result(res[idx])

with tab2:
    res = [
        r
        for r in all_results
        if "ollama" not in str(r).lower()
        and ("crew" in str(r).lower() or "pass_fail" in r or "EvaluationReport" in r)
    ]
    if not res:
        st.markdown(
            '<div class="empty-box"><div class="empty-icon">📭</div>No CrewAI results yet.<br>Run the CrewAI Evaluator from the Launch page.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"{len(res)} run(s) found")
        options = [
            f"Run {i + 1} · {r.get('test_case_id', '?')} · {str(r.get('timestamp', ''))[:16]}"
            for i, r in enumerate(res)
        ]
        idx = st.selectbox(
            "Select run", range(len(options)), format_func=lambda i: options[i], key="crew_sel"
        )
        # Handle nested EvaluationReport wrapper
        r = res[idx]
        if "EvaluationReport" in r and isinstance(r["EvaluationReport"], dict):
            merged = {
                **r["EvaluationReport"],
                **{k: v for k, v in r.items() if k != "EvaluationReport"},
            }
        else:
            merged = r
        st.divider()
        render_result(merged)

with tab3:
    st.markdown(
        '<div class="empty-box"><div class="empty-icon">🗄</div>Professional pipeline results are in<br><strong>Leaderboard · Responses · Pairwise · Metrics</strong></div>',
        unsafe_allow_html=True,
    )
