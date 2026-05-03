"""
dashboard.py — Entry point for Streamlit.
Navigation via sidebar (auto-generated from pages/) and clickable buttons below.
"""

import json
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="LLM Eval Framework",
    page_icon="🧪",
    layout="wide",
)

ROOT = Path(__file__).parent

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.hero-title { font-size:2.2rem; font-weight:700; color:#e6edf3; margin-bottom:6px; }
.hero-sub   { font-size:0.95rem; color:#6e7681; margin-bottom:32px; }
.sc  { background:#0d1117; border:1px solid #21262d; border-radius:10px; padding:18px; text-align:center; }
.sv  { font-size:2rem; font-weight:700; color:#58a6ff; font-family:'JetBrains Mono',monospace; }
.sl  { font-size:0.7rem; color:#6e7681; text-transform:uppercase; letter-spacing:0.08em; margin-top:2px; }
.nav-hint { background:#0c1f3f; border:1px solid #1f6feb; border-radius:8px;
            padding:10px 16px; font-size:0.85rem; color:#58a6ff; margin-bottom:24px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="hero-title">🧪 LLM Evaluation Framework</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Evaluate language models with Ollama, CrewAI multi-agent crews, and a full professional pipeline.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="nav-hint">👈 Use the <strong>sidebar</strong> to navigate, or click a page button below.</div>',
    unsafe_allow_html=True,
)

# ── Quick stats ───────────────────────────────────────────────────────────────
json_runs = 0
for fname in ["evaluation_results.json", "evaluation_history.json"]:
    p = ROOT / fname
    if p.exists():
        try:
            data = json.loads(p.read_text())
            json_runs += len(data) if isinstance(data, list) else 1
        except Exception:
            pass

try:
    from sqlite_store import get_all_metrics_df, get_experiments, get_leaderboard

    lb = get_leaderboard()
    exps = get_experiments()
    df = get_all_metrics_df()
    db_responses = len(df)
    db_models = len(lb)
    db_exps = len(exps)
except Exception:
    db_responses = db_models = db_exps = 0

c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in [
    (c1, json_runs, "JSON Eval Runs"),
    (c2, db_models, "Pipeline Models"),
    (c3, db_responses, "Pipeline Responses"),
    (c4, db_exps, "Experiments"),
]:
    col.markdown(
        f'<div class="sc"><div class="sv">{val}</div><div class="sl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
st.markdown("### Navigate")

# ── Clickable page links ──────────────────────────────────────────────────────
PAGES = [
    ("pages/1_Launch.py", "🚀", "Launch", "Run Ollama, CrewAI, or the Professional Pipeline"),
    ("pages/2_Results.py", "🔍", "Results", "Browse Ollama & CrewAI evaluation results"),
    ("pages/3_Overview.py", "🏠", "Overview", "Aggregated stats from all evaluators"),
    (
        "pages/4_Professional_Pipeline.py",
        "⚙️",
        "Professional Pipeline",
        "Run pipeline · leaderboard · pairwise · metrics",
    ),
]

cols = st.columns(4, gap="medium")

for col, (path, icon, title, desc) in zip(cols, PAGES, strict=False):
    with col:
        st.page_link(path, label=f"{icon} **{title}**", use_container_width=True)
        st.caption(desc)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.caption("Results auto-refresh on page load · DB stored in evals.db")
