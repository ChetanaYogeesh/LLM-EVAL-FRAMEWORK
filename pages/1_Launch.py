"""pages/1_Launch.py — Run any evaluator."""

import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
st.set_page_config(page_title="Launch", page_icon="🚀", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.ph { font-size:1.6rem; font-weight:700; color:#e6edf3; margin-bottom:4px; }
.ps { color:#6e7681; font-size:0.88rem; margin-bottom:28px; }
.row { display:flex; align-items:center; gap:16px; background:#0d1117; border:1px solid #21262d; border-radius:10px; padding:16px 20px; margin-bottom:12px; }
.ri  { font-size:1.8rem; flex-shrink:0; }
.rt  { font-weight:700; color:#e6edf3; font-size:0.95rem; margin:0 0 2px; }
.rm  { font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#6e7681; }
.badge { display:inline-block; padding:1px 7px; border-radius:20px; font-size:0.65rem; font-family:'JetBrains Mono',monospace; font-weight:600; margin-left:6px; }
.bg { background:#0d2818; color:#3fb950; border:1px solid #238636; }
.bb { background:#0c1f3f; color:#58a6ff; border:1px solid #1f6feb; }
.ba { background:#2d1e0f; color:#d29922; border:1px solid #9e6a03; }
.bx { background:#161b22; color:#8b949e; border:1px solid #30363d; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="ph">🚀 Launch Evaluators</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ps">Run any evaluator below. Results save automatically and appear in Results & Pipeline pages.</div>',
    unsafe_allow_html=True,
)

EVALUATORS = [
    {
        "key": "ollama",
        "icon": "🟢",
        "title": "Ollama Evaluator",
        "badges": [("Free", "bg"), ("Local → OpenRouter fallback", "bx")],
        "meta": "llama3.2 locally · falls back to gpt-4o-mini via OpenRouter",
        "script": "ollama_evaluator.py",
        "timeout": 90,
        "note": None,
    },
    {
        "key": "crew",
        "icon": "🔵",
        "title": "CrewAI Evaluator",
        "badges": [("Multi-agent", "bb"), ("Requires OPENAI_API_KEY", "bx")],
        "meta": "6 specialist agents · hierarchical · trace, quality, safety, cost, regression",
        "script": "crewai_evaluator.py",
        "timeout": 180,
        "note": "May take up to 3 minutes",
    },
    {
        "key": "pro",
        "icon": "🔴",
        "title": "Professional Pipeline",
        "badges": [("SQLite", "ba"), ("Multi-model · NLP scoring", "bx")],
        "meta": "Full pipeline · pairwise comparisons · LLM judge · leaderboard",
        "script": None,
        "timeout": None,
        "note": "→ Configure in the Pipeline page (sidebar)",
    },
]

for ev in EVALUATORS:
    badges_html = "".join(f'<span class="badge {c}">{t}</span>' for t, c in ev["badges"])
    col_info, col_btn = st.columns([5, 1])
    with col_info:
        st.markdown(
            f"""
        <div class="row">
            <div class="ri">{ev["icon"]}</div>
            <div>
                <div class="rt">{ev["title"]} {badges_html}</div>
                <div class="rm">{ev["meta"]}</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col_btn:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if ev["script"]:
            clicked = st.button("▶ Run", key=ev["key"], use_container_width=True, type="primary")
        else:
            clicked = False
            st.button("→ Pipeline", key=ev["key"], use_container_width=True, disabled=True)

    if ev["note"] and not ev["script"]:
        st.caption(ev["note"])

    if ev["script"] and clicked:
        note = f" — {ev['note']}" if ev["note"] else ""
        with st.spinner(f"Running {ev['title']}{note}..."):
            result = subprocess.run(
                [sys.executable, str(ROOT / ev["script"])],
                capture_output=True,
                text=True,
                cwd=ROOT,
                timeout=ev["timeout"],
            )
        combined_out = (result.stdout or "").strip()
        combined_err = (result.stderr or "").strip()
        full_log = "\n".join(filter(None, [combined_out, combined_err])) or "No output captured."

        if result.returncode == 0:
            st.success(f"✅ {ev['title']} completed! See **Results** in the sidebar.")
            with st.expander("Output log"):
                st.code(full_log)
        else:
            st.error(f"❌ {ev['title']} failed")
            st.code(full_log)
