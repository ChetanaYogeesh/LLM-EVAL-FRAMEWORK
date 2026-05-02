"""
dashboard.py — Entry point for Streamlit.
Navigation is handled automatically via the pages/ directory.
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
.hero { padding: 48px 0 32px; }
.hero-title { font-size: 2.4rem; font-weight: 700; color: #e6edf3; margin-bottom: 8px; }
.hero-sub   { font-size: 1rem; color: #6e7681; margin-bottom: 40px; }
.card { background:#0d1117; border:1px solid #21262d; border-radius:12px; padding:20px 22px; height:100%; }
.card-icon  { font-size:1.8rem; margin-bottom:10px; }
.card-title { font-weight:700; color:#e6edf3; font-size:0.95rem; margin-bottom:4px; }
.card-desc  { font-size:0.82rem; color:#6e7681; line-height:1.5; }
.stat-row   { background:#0d1117; border:1px solid #21262d; border-radius:10px; padding:16px 20px; text-align:center; }
.stat-val   { font-size:2rem; font-weight:700; color:#58a6ff; font-family:'JetBrains Mono',monospace; }
.stat-lbl   { font-size:0.72rem; color:#6e7681; text-transform:uppercase; letter-spacing:0.08em; margin-top:2px; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
    <div class="hero-title">🧪 LLM Evaluation Framework</div>
    <div class="hero-sub">
        Evaluate language models with Ollama, CrewAI multi-agent crews,
        and a full professional pipeline — all in one place.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Quick stats from both data sources ───────────────────────────────────────
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
        f'<div class="stat-row"><div class="stat-val">{val}</div><div class="stat-lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

# ── Page guide ────────────────────────────────────────────────────────────────
st.markdown("### Navigate")
cols = st.columns(3)
pages = [
    ("🚀", "1 · Launch", "Run Ollama, CrewAI, or the Professional Pipeline"),
    ("🔍", "2 · Results", "Browse Ollama & CrewAI evaluation results"),
    ("🏠", "3 · Overview", "Aggregated stats from all evaluators"),
    ("⚙️", "4 · Pipeline", "Configure and run the full professional pipeline"),
    ("⚔️", "5 · Pairwise", "Head-to-head model comparisons"),
    ("📊", "6 · Metrics", "Deep dive into NLP and quality metrics"),
]
for i, (icon, title, desc) in enumerate(pages):
    with cols[i % 3]:
        st.markdown(
            f"""
        <div class="card">
            <div class="card-icon">{icon}</div>
            <div class="card-title">{title}</div>
            <div class="card-desc">{desc}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
st.caption("Use the sidebar to navigate · Results auto-refresh on page load")
