"""
dashboard.py — Entry point for Streamlit. Analytics-style home dashboard.
"""

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="LLM Eval Framework", page_icon="🧪", layout="wide")

ROOT = Path(__file__).parent

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.top-bar {
    display:flex; align-items:center; justify-content:space-between;
    padding:0 0 20px; border-bottom:1px solid #21262d; margin-bottom:24px;
}
.top-title { font-size:1.3rem; font-weight:600; color:#e6edf3; }
.top-sub   { font-size:0.8rem; color:#6e7681; margin-top:2px; }
.top-time  { font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#6e7681;
             background:#161b22; padding:4px 10px; border-radius:6px; border:1px solid #21262d; }

.kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:24px; }
.kpi-card {
    background:#0d1117; border:1px solid #21262d; border-radius:10px;
    padding:16px 18px; position:relative; overflow:hidden;
}
.kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px; border-radius:10px 10px 0 0;
}
.kpi-card.blue::before  { background:#58a6ff; }
.kpi-card.green::before { background:#3fb950; }
.kpi-card.amber::before { background:#d29922; }
.kpi-card.purple::before{ background:#bc8cff; }

.kpi-label { font-size:0.72rem; color:#6e7681; text-transform:uppercase;
             letter-spacing:0.07em; margin-bottom:6px; }
.kpi-value { font-size:2rem; font-weight:600; color:#e6edf3;
             font-family:'JetBrains Mono',monospace; line-height:1; }
.kpi-delta { font-size:0.72rem; margin-top:6px; }
.kpi-delta.up   { color:#3fb950; }
.kpi-delta.down { color:#f85149; }
.kpi-delta.neutral { color:#6e7681; }

.section-label {
    font-size:0.72rem; font-weight:600; color:#6e7681; text-transform:uppercase;
    letter-spacing:0.1em; margin-bottom:12px; margin-top:4px;
}

.activity-item {
    display:flex; align-items:center; gap:12px;
    padding:10px 0; border-bottom:1px solid #161b22;
}
.activity-dot {
    width:8px; height:8px; border-radius:50%; flex-shrink:0;
}
.dot-green  { background:#3fb950; }
.dot-blue   { background:#58a6ff; }
.dot-amber  { background:#d29922; }
.dot-red    { background:#f85149; }
.activity-text { font-size:0.83rem; color:#e6edf3; flex:1; }
.activity-time { font-size:0.72rem; color:#6e7681;
                 font-family:'JetBrains Mono',monospace; }

.nav-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:8px; }
.nav-card {
    background:#0d1117; border:1px solid #21262d; border-radius:8px;
    padding:14px 16px; cursor:pointer; transition:border-color 0.15s;
}
.nav-card:hover { border-color:#58a6ff; }
.nav-icon  { font-size:1.2rem; margin-bottom:6px; }
.nav-title { font-size:0.85rem; font-weight:500; color:#e6edf3; margin-bottom:2px; }
.nav-desc  { font-size:0.72rem; color:#6e7681; }

.status-pill {
    display:inline-block; padding:2px 8px; border-radius:20px;
    font-size:0.68rem; font-weight:600; font-family:'JetBrains Mono',monospace;
}
.pill-pass { background:#0d2818; color:#3fb950; border:1px solid #238636; }
.pill-fail { background:#1a0a0a; color:#f85149; border:1px solid #6e1a1a; }
.pill-unk  { background:#161b22; color:#8b949e; border:1px solid #30363d; }
</style>
""",
    unsafe_allow_html=True,
)

now = datetime.now().strftime("%d %b %Y, %H:%M")
st.markdown(
    f"""
<div class="top-bar">
    <div>
        <div class="top-title">🧪 LLM Evaluation Framework</div>
        <div class="top-sub">Monitor, compare and ship better language models</div>
    </div>
    <div class="top-time">Last updated: {now}</div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Collect data ──────────────────────────────────────────────────────────────
json_results = []
for fname in ["evaluation_results.json", "evaluation_history.json"]:
    p = ROOT / fname
    if p.exists():
        try:
            data = json.loads(p.read_text())
            items = data if isinstance(data, list) else [data]
            json_results.extend(items)
        except Exception:
            pass

json_runs = len(json_results)
json_pass = sum(
    1
    for r in json_results
    if str(r.get("pass_fail", r.get("EvaluationReport", {}).get("pass_fail", ""))).lower() == "pass"
)
pass_rate = round(json_pass / json_runs * 100) if json_runs else 0

try:
    from sqlite_store import get_all_metrics_df, get_experiments, get_leaderboard

    lb = get_leaderboard()
    exps = get_experiments()
    df = get_all_metrics_df()
    db_models = len(lb)
    db_responses = len(df)
    avg_score = (
        round(df["judge_score"].mean(), 1) if not df.empty and "judge_score" in df.columns else "—"
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
    <div class="kpi-card blue">
        <div class="kpi-label">Total Eval Runs</div>
        <div class="kpi-value">{json_runs}</div>
        <div class="kpi-delta neutral">Ollama + CrewAI evaluators</div>
    </div>
    <div class="kpi-card green">
        <div class="kpi-label">Pass Rate</div>
        <div class="kpi-value">{pass_rate}%</div>
        <div class="kpi-delta {"up" if pass_rate >= 70 else "down"}">
            {"Above" if pass_rate >= 70 else "Below"} 70% threshold
        </div>
    </div>
    <div class="kpi-card amber">
        <div class="kpi-label">Pipeline Models</div>
        <div class="kpi-value">{db_models}</div>
        <div class="kpi-delta neutral">{db_responses} total responses</div>
    </div>
    <div class="kpi-card purple">
        <div class="kpi-label">Avg Judge Score</div>
        <div class="kpi-value">{avg_score}</div>
        <div class="kpi-delta neutral">Professional pipeline</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Main content: activity feed + leaderboard ─────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown('<div class="section-label">Recent Activity</div>', unsafe_allow_html=True)

    if not json_results:
        st.markdown(
            '<div style="color:#6e7681;font-size:0.85rem;padding:20px 0">No evaluations run yet. Go to Launch to get started.</div>',
            unsafe_allow_html=True,
        )
    else:
        recent = json_results[-10:][::-1]
        for r in recent:
            inner = r.get("EvaluationReport", r)
            pf = str(inner.get("pass_fail", "UNKNOWN")).upper()
            tc = inner.get("test_case_id", r.get("test_case_id", "—"))
            ts = r.get("timestamp", "")[:16].replace("T", " ")
            fm = inner.get("failure_mode") or "none"
            dot = "dot-green" if pf == "PASS" else "dot-red" if pf == "FAIL" else "dot-amber"
            pill = "pill-pass" if pf == "PASS" else "pill-fail" if pf == "FAIL" else "pill-unk"

            st.markdown(
                f"""
            <div class="activity-item">
                <div class="activity-dot {dot}"></div>
                <div class="activity-text">
                    <span style="font-weight:500">{tc}</span>
                    &nbsp;<span class="status-pill {pill}">{pf}</span>
                    &nbsp;<span style="color:#6e7681;font-size:0.75rem">· {fm}</span>
                </div>
                <div class="activity-time">{ts}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

with col_right:
    st.markdown('<div class="section-label">Pipeline Leaderboard</div>', unsafe_allow_html=True)

    if not lb:
        st.markdown(
            '<div style="color:#6e7681;font-size:0.85rem;padding:20px 0">No pipeline data yet. Run the Professional Pipeline to see model rankings.</div>',
            unsafe_allow_html=True,
        )
    else:
        for i, row in enumerate(lb[:6], 1):
            score = row.get("avg_judge_score", 0) or 0
            bar_w = min(100, int(score * 10))
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            st.markdown(
                f"""
            <div style="margin-bottom:12px">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                    <span style="font-size:0.83rem;color:#e6edf3">{medal} {row["name"]}</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#6e7681">{score}/10</span>
                </div>
                <div style="background:#161b22;border-radius:4px;height:4px">
                    <div style="background:#58a6ff;width:{bar_w}%;height:4px;border-radius:4px"></div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

# ── Navigate ──────────────────────────────────────────────────────────────────
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Navigate</div>', unsafe_allow_html=True)

nav_cols = st.columns(4, gap="small")
nav_pages = [
    ("pages/1_Launch.py", "🚀", "Launch", "Run any evaluator"),
    ("pages/2_Results.py", "🔍", "Results", "Ollama & CrewAI results"),
    ("pages/3_Overview.py", "🏠", "Overview", "Aggregated stats"),
    (
        "pages/4_Professional_Pipeline.py",
        "⚙️",
        "Professional Pipeline",
        "Pipeline · leaderboard · metrics",
    ),
]
for col, (path, icon, title, desc) in zip(nav_cols, nav_pages, strict=False):
    with col:
        st.page_link(path, label=f"{icon}  {title}", use_container_width=True)
        st.caption(desc)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.caption(f"LLM Eval Framework · {now}")
