"""pages/2_Results.py — Analytics-style results viewer."""

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.top-bar { display:flex; align-items:center; justify-content:space-between;
           padding:0 0 20px; border-bottom:1px solid #21262d; margin-bottom:24px; }
.top-title { font-size:1.3rem; font-weight:600; color:#e6edf3; }
.top-sub   { font-size:0.8rem; color:#6e7681; margin-top:2px; }
.section-label { font-size:0.72rem; font-weight:600; color:#6e7681;
                 text-transform:uppercase; letter-spacing:0.1em; margin-bottom:12px; }
.kpi-row { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:20px; }
.kpi-card { background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:14px 16px; }
.kpi-label { font-size:0.68rem; color:#6e7681; text-transform:uppercase;
             letter-spacing:0.07em; margin-bottom:4px; }
.kpi-value { font-size:1.6rem; font-weight:600; color:#e6edf3;
             font-family:'JetBrains Mono',monospace; }
.kpi-pass   { color:#3fb950; }
.kpi-fail   { color:#f85149; }
.status-pill { display:inline-block; padding:2px 8px; border-radius:20px;
               font-size:0.68rem; font-weight:600; font-family:'JetBrains Mono',monospace; }
.pill-pass { background:#0d2818; color:#3fb950; border:1px solid #238636; }
.pill-fail { background:#1a0a0a; color:#f85149; border:1px solid #6e1a1a; }
.pill-unk  { background:#161b22; color:#8b949e; border:1px solid #30363d; }
.safety-ok   { color:#3fb950; font-size:0.85rem; font-weight:500; }
.safety-flag { color:#f85149; font-size:0.85rem; font-weight:500; }
.run-row { display:flex; align-items:center; gap:10px; padding:9px 12px;
           background:#0d1117; border:1px solid #21262d; border-radius:7px;
           margin-bottom:6px; cursor:pointer; }
.run-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.eb { text-align:center; padding:48px 20px; color:#6e7681; font-size:0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="top-bar">
    <div>
        <div class="top-title">🔍 Evaluation Results</div>
        <div class="top-sub">Results from Ollama and CrewAI evaluators</div>
    </div>
</div>
""",
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


def unwrap(r: dict) -> dict:
    if "EvaluationReport" in r and isinstance(r["EvaluationReport"], dict):
        return {**r["EvaluationReport"], **{k: v for k, v in r.items() if k != "EvaluationReport"}}
    return r


def pf_cls(v: str) -> str:
    return (
        "kpi-pass"
        if str(v).lower() in ("pass", "approved")
        else "kpi-fail"
        if str(v).lower() in ("fail", "rejected")
        else ""
    )


def pill_cls(v: str) -> str:
    return (
        "pill-pass"
        if str(v).lower() == "pass"
        else "pill-fail"
        if str(v).lower() == "fail"
        else "pill-unk"
    )


def sfmt(flag: bool, label: str) -> str:
    return (
        f'<span class="safety-flag">⚠ {label}</span>'
        if flag
        else f'<span class="safety-ok">✓ No {label.lower()}</span>'
    )


PLOT_LAYOUT = dict(
    plot_bgcolor="#0d1117",
    paper_bgcolor="#0d1117",
    font_color="#8b949e",
    showlegend=False,
    margin=dict(t=8, b=0, l=0, r=0),
)


def render_result(r: dict) -> None:
    pf = str(r.get("pass_fail", "UNKNOWN")).upper()
    rd = str(r.get("release_decision", "—"))
    fm = r.get("failure_mode") or "none"
    tc = r.get("test_case_id", "N/A")
    ts = str(r.get("timestamp", ""))[:19].replace("T", " ")

    if ts:
        st.caption(f"🕐 {ts}")

    st.markdown(
        f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">Test Case</div>
            <div class="kpi-value" style="font-size:1.1rem">{tc}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Pass / Fail</div>
            <div class="kpi-value {pf_cls(pf)}">{pf}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Release Decision</div>
            <div class="kpi-value" style="font-size:1.1rem;text-transform:capitalize">{rd}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Failure Mode</div>
            <div class="kpi-value" style="font-size:1rem;margin-top:4px">{fm}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Safety signals
    st.markdown('<div class="section-label">Safety Signals</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.markdown(
        sfmt(r.get("hallucination_detected", False), "Hallucination"), unsafe_allow_html=True
    )
    s2.markdown(sfmt(r.get("bias_detected", False), "Bias"), unsafe_allow_html=True)
    s3.markdown(sfmt(r.get("toxicity_detected", False), "Toxicity"), unsafe_allow_html=True)

    # Metrics chart
    metrics = r.get("metrics", {})
    if isinstance(metrics, dict):
        numeric = {k: v for k, v in metrics.items() if isinstance(v, int | float)}
        if numeric:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Metrics</div>', unsafe_allow_html=True)
            df_m = pd.DataFrame(numeric.items(), columns=["Metric", "Value"])

            col_chart, col_vals = st.columns([2, 1])
            with col_chart:
                fig = px.bar(
                    df_m, x="Metric", y="Value", color="Value", color_continuous_scale="Blues"
                )
                fig.update_layout(**PLOT_LAYOUT, height=180, coloraxis_showscale=False)
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(fig, use_container_width=True)
            with col_vals:
                for _, row in df_m.iterrows():
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:4px 0;border-bottom:1px solid #21262d;font-size:0.8rem">'
                        f'<span style="color:#6e7681">{row["Metric"]}</span>'
                        f"<span style=\"color:#e6edf3;font-family:'JetBrains Mono',monospace\">"
                        f"{row['Value']}</span></div>",
                        unsafe_allow_html=True,
                    )

    # Recommendations & bottlenecks
    recs = r.get("recommendations", [])
    bots = r.get("top_bottlenecks", [])
    regs = r.get("top_regressions", [])

    if any([recs, bots, regs]):
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        if recs:
            with c1:
                st.markdown(
                    '<div class="section-label">Recommendations</div>', unsafe_allow_html=True
                )
                for rec in recs if isinstance(recs, list) else [recs]:
                    st.markdown(
                        f'<div style="font-size:0.82rem;color:#8b949e;padding:3px 0;border-bottom:1px solid #21262d">→ {rec}</div>',
                        unsafe_allow_html=True,
                    )
        if bots:
            with c2:
                st.markdown('<div class="section-label">Bottlenecks</div>', unsafe_allow_html=True)
                for b in bots if isinstance(bots, list) else [bots]:
                    st.markdown(
                        f'<div style="font-size:0.82rem;color:#8b949e;padding:3px 0;border-bottom:1px solid #21262d">🐢 {b}</div>',
                        unsafe_allow_html=True,
                    )
        if regs:
            with c3:
                st.markdown('<div class="section-label">Regressions</div>', unsafe_allow_html=True)
                for reg in regs if isinstance(regs, list) else [regs]:
                    st.markdown(
                        f'<div style="font-size:0.82rem;color:#8b949e;padding:3px 0;border-bottom:1px solid #21262d">📉 {reg}</div>',
                        unsafe_allow_html=True,
                    )


all_results = load_results()

# ── Summary bar ───────────────────────────────────────────────────────────────
if all_results:
    total = len(all_results)
    passes = sum(1 for r in all_results if str(unwrap(r).get("pass_fail", "")).lower() == "pass")
    fails = sum(1 for r in all_results if str(unwrap(r).get("pass_fail", "")).lower() == "fail")

    st.markdown(
        f"""
    <div style="display:flex;gap:16px;margin-bottom:20px">
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;
                    padding:10px 16px;display:flex;align-items:center;gap:8px">
            <span style="font-size:0.72rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.07em">Total</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:1rem;
                         font-weight:600;color:#e6edf3">{total}</span>
        </div>
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;
                    padding:10px 16px;display:flex;align-items:center;gap:8px">
            <span style="font-size:0.72rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.07em">Passed</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:1rem;
                         font-weight:600;color:#3fb950">{passes}</span>
        </div>
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;
                    padding:10px 16px;display:flex;align-items:center;gap:8px">
            <span style="font-size:0.72rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.07em">Failed</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:1rem;
                         font-weight:600;color:#f85149">{fails}</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

tab1, tab2 = st.tabs(["🟢 Ollama", "🔵 CrewAI"])

with tab1:
    res = [r for r in all_results if "ollama" in str(r).lower()]
    if not res:
        st.markdown(
            '<div class="eb">📭 No Ollama results yet.<br>Run the Ollama Evaluator from the Launch page.</div>',
            unsafe_allow_html=True,
        )
    else:
        col_list, col_detail = st.columns([1, 2], gap="large")
        with col_list:
            st.markdown('<div class="section-label">Runs</div>', unsafe_allow_html=True)
            for i, r in enumerate(reversed(res)):
                u = unwrap(r)
                pf = str(u.get("pass_fail", "?")).upper()
                tc = u.get("test_case_id", "?")
                ts = str(r.get("timestamp", ""))[:10]
                dot = (
                    "background:#3fb950"
                    if pf == "PASS"
                    else "background:#f85149"
                    if pf == "FAIL"
                    else "background:#d29922"
                )
                if st.button(f"{tc} · {pf} · {ts}", key=f"o_{i}", use_container_width=True):
                    st.session_state["o_sel"] = i
            sel = st.session_state.get("o_sel", 0)
        with col_detail:
            st.markdown('<div class="section-label">Detail</div>', unsafe_allow_html=True)
            sel_idx = min(st.session_state.get("o_sel", 0), len(res) - 1)
            render_result(unwrap(list(reversed(res))[sel_idx]))

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
        col_list, col_detail = st.columns([1, 2], gap="large")
        with col_list:
            st.markdown('<div class="section-label">Runs</div>', unsafe_allow_html=True)
            for i, r in enumerate(reversed(res)):
                u = unwrap(r)
                pf = str(u.get("pass_fail", "?")).upper()
                tc = u.get("test_case_id", "?")
                ts = str(r.get("timestamp", ""))[:10]
                if st.button(f"{tc} · {pf} · {ts}", key=f"c_{i}", use_container_width=True):
                    st.session_state["c_sel"] = i
        with col_detail:
            st.markdown('<div class="section-label">Detail</div>', unsafe_allow_html=True)
            sel_idx = min(st.session_state.get("c_sel", 0), len(res) - 1)
            render_result(unwrap(list(reversed(res))[sel_idx]))
