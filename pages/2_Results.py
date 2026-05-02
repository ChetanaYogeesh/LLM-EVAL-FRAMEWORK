"""pages/2_Results.py — Results from Ollama and CrewAI evaluators."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent.parent
st.set_page_config(page_title="Results", page_icon="🔍", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.ph { font-size:1.6rem; font-weight:700; color:#e6edf3; margin-bottom:4px; }
.ps { color:#6e7681; font-size:0.88rem; margin-bottom:24px; }
.kb { background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:14px 16px; }
.kl { font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#6e7681; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:4px; }
.kv { font-size:1.5rem; font-weight:700; color:#e6edf3; }
.kp { color:#3fb950; } .kf { color:#f85149; }
.so { color:#3fb950; font-weight:600; font-size:0.88rem; }
.sf { color:#f85149; font-weight:600; font-size:0.88rem; }
.ri { padding:4px 0; font-size:0.84rem; color:#8b949e; border-bottom:1px solid #21262d; }
.eb { text-align:center; padding:48px 20px; color:#6e7681; font-size:0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="ph">🔍 Evaluation Results</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ps">Results from Ollama and CrewAI evaluators. Re-run to refresh.</div>',
    unsafe_allow_html=True,
)


def load_results() -> list:
    out = []
    for fname in ["evaluation_results.json", "evaluation_history.json"]:
        p = ROOT / fname
        if p.exists():
            try:
                data = json.loads(p.read_text())
                out.extend(data if isinstance(data, list) else [data])
            except Exception:
                pass
    return out


def pf_cls(v: str) -> str:
    return (
        "kp"
        if str(v).lower() in ("pass", "approved")
        else "kf"
        if str(v).lower() in ("fail", "rejected")
        else ""
    )


def sfmt(flag: bool, label: str) -> str:
    return (
        f'<span class="sf">⚠ {label}</span>'
        if flag
        else f'<span class="so">✓ No {label.lower()}</span>'
    )


def render(r: dict) -> None:
    pf = str(r.get("pass_fail", "UNKNOWN")).upper()
    rd = str(r.get("release_decision", "—"))
    fm = r.get("failure_mode") or "none"
    tc = r.get("test_case_id", "N/A")
    ts = str(r.get("timestamp", ""))[:19].replace("T", " ")
    if ts:
        st.caption(f"🕐 {ts}")

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val, cls in [
        (c1, "Test Case", tc, ""),
        (c2, "Pass / Fail", pf, pf_cls(pf)),
        (c3, "Release", rd, ""),
        (c4, "Failure Mode", fm, ""),
    ]:
        col.markdown(
            f'<div class="kb"><div class="kl">{lbl}</div><div class="kv {cls}">{val}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.markdown(
        sfmt(r.get("hallucination_detected", False), "Hallucination"), unsafe_allow_html=True
    )
    s2.markdown(sfmt(r.get("bias_detected", False), "Bias"), unsafe_allow_html=True)
    s3.markdown(sfmt(r.get("toxicity_detected", False), "Toxicity"), unsafe_allow_html=True)

    metrics = r.get("metrics", {})
    if isinstance(metrics, dict):
        numeric = {k: v for k, v in metrics.items() if isinstance(v, int | float)}
        if numeric:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            df = pd.DataFrame(numeric.items(), columns=["Metric", "Value"])
            fig = px.bar(df, x="Metric", y="Value", color="Value", color_continuous_scale="Blues")
            fig.update_layout(
                plot_bgcolor="#0d1117",
                paper_bgcolor="#0d1117",
                font_color="#8b949e",
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(t=4, b=0, l=0, r=0),
                height=200,
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

    recs = r.get("recommendations", [])
    if recs:
        st.markdown("**Recommendations**")
        for rec in recs if isinstance(recs, list) else [recs]:
            st.markdown(f'<div class="ri">→ {rec}</div>', unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    bots = r.get("top_bottlenecks", [])
    regs = r.get("top_regressions", [])
    if bots:
        with b1:
            st.markdown("**Bottlenecks**")
            for b in bots if isinstance(bots, list) else [bots]:
                st.markdown(f'<div class="ri">🐢 {b}</div>', unsafe_allow_html=True)
    if regs:
        with b2:
            st.markdown("**Regressions**")
            for reg in regs if isinstance(regs, list) else [regs]:
                st.markdown(f'<div class="ri">📉 {reg}</div>', unsafe_allow_html=True)


all_results = load_results()

tab1, tab2 = st.tabs(["🟢 Ollama", "🔵 CrewAI"])

with tab1:
    res = [r for r in all_results if "ollama" in str(r).lower()]
    if not res:
        st.markdown(
            '<div class="eb">📭 No Ollama results yet.<br>Run the Ollama Evaluator from the Launch page.</div>',
            unsafe_allow_html=True,
        )
    else:
        opts = [
            f"Run {i + 1} · {r.get('test_case_id', '?')} · {str(r.get('timestamp', ''))[:16]}"
            for i, r in enumerate(res)
        ]
        idx = st.selectbox(
            "Select run", range(len(opts)), format_func=lambda i: opts[i], key="o_sel"
        )
        st.divider()
        render(res[idx])

with tab2:
    res = [
        r
        for r in all_results
        if "ollama" not in str(r).lower()
        and ("crew" in str(r).lower() or "pass_fail" in r or "EvaluationReport" in r)
    ]
    if not res:
        st.markdown(
            '<div class="eb">📭 No CrewAI results yet.<br>Run the CrewAI Evaluator from the Launch page.</div>',
            unsafe_allow_html=True,
        )
    else:
        opts = [
            f"Run {i + 1} · {r.get('test_case_id', '?')} · {str(r.get('timestamp', ''))[:16]}"
            for i, r in enumerate(res)
        ]
        idx = st.selectbox(
            "Select run", range(len(opts)), format_func=lambda i: opts[i], key="c_sel"
        )
        r = res[idx]
        merged = (
            {**r["EvaluationReport"], **{k: v for k, v in r.items() if k != "EvaluationReport"}}
            if "EvaluationReport" in r and isinstance(r["EvaluationReport"], dict)
            else r
        )
        st.divider()
        render(merged)
