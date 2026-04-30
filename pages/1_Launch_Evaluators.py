import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent

st.set_page_config(page_title="Launch Evaluators", page_icon="🚀", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.page-header { font-size:1.6rem; font-weight:700; color:#e6edf3; margin-bottom:4px; }
.page-sub    { color:#6e7681; font-size:0.88rem; margin-bottom:24px; }

.eval-row {
    display:flex; align-items:center; gap:16px;
    background:#0d1117; border:1px solid #21262d; border-radius:10px;
    padding:16px 20px; margin-bottom:12px;
}
.eval-icon  { font-size:1.8rem; flex-shrink:0; }
.eval-info  { flex:1; min-width:0; }
.eval-title { font-weight:700; color:#e6edf3; font-size:0.95rem; margin:0 0 2px; }
.eval-meta  { font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#6e7681; letter-spacing:0.04em; }
.badge {
    display:inline-block; padding:1px 7px; border-radius:20px;
    font-size:0.65rem; font-family:'JetBrains Mono',monospace; font-weight:600;
    letter-spacing:0.04em; margin-left:6px;
}
.badge-green { background:#0d2818; color:#3fb950; border:1px solid #238636; }
.badge-blue  { background:#0c1f3f; color:#58a6ff; border:1px solid #1f6feb; }
.badge-amber { background:#2d1e0f; color:#d29922; border:1px solid #9e6a03; }
.badge-gray  { background:#161b22; color:#8b949e; border:1px solid #30363d; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="page-header">🚀 Launch Evaluators</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Select an evaluator to run. Results are saved automatically and visible in the Results page.</div>',
    unsafe_allow_html=True,
)

# ── Ollama ────────────────────────────────────────────────────────────────────
col_info, col_btn = st.columns([5, 1])
with col_info:
    st.markdown(
        """
    <div class="eval-row">
        <div class="eval-icon">🟢</div>
        <div class="eval-info">
            <div class="eval-title">Ollama Evaluator
                <span class="badge badge-green">Free</span>
                <span class="badge badge-gray">Local → OpenRouter fallback</span>
            </div>
            <div class="eval-meta">Lightweight · Uses llama3.2 locally or gpt-4o-mini via OpenRouter when offline</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col_btn:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    run_ollama = st.button("▶ Run", key="ollama", use_container_width=True, type="primary")

if run_ollama:
    with st.spinner("Running Ollama Evaluator..."):
        result = subprocess.run(
            [sys.executable, str(ROOT / "ollama_evaluator.py")],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=90,
        )
    if result.returncode == 0:
        st.success("✅ Ollama Evaluator completed! Go to **Results** to view output.")
        with st.expander("Show output log"):
            st.code(result.stdout or "No output.")
    else:
        st.error("❌ Ollama Evaluator failed")
        st.code(result.stderr or "No error output captured.")

# ── CrewAI ────────────────────────────────────────────────────────────────────
col_info, col_btn = st.columns([5, 1])
with col_info:
    st.markdown(
        """
    <div class="eval-row">
        <div class="eval-icon">🔵</div>
        <div class="eval-info">
            <div class="eval-title">CrewAI Evaluator
                <span class="badge badge-blue">Multi-agent</span>
                <span class="badge badge-gray">Requires OPENAI_API_KEY</span>
            </div>
            <div class="eval-meta">Hierarchical crew · 6 agents · Trace, quality, safety, cost, regression analysis</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col_btn:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    run_crew = st.button("▶ Run", key="crew", use_container_width=True, type="primary")

if run_crew:
    with st.spinner("Running CrewAI Evaluator — up to 3 minutes..."):
        result = subprocess.run(
            [sys.executable, str(ROOT / "crewai_evaluator.py")],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=180,
        )
    if result.returncode == 0:
        st.success("✅ CrewAI Evaluator completed! Go to **Results** to view output.")
        with st.expander("Show output log"):
            st.code(result.stdout or "No output.")
    else:
        st.error("❌ CrewAI Evaluator failed")
        st.code(result.stderr or "No error output captured.")

# ── Professional Pipeline ─────────────────────────────────────────────────────
col_info, col_btn = st.columns([5, 1])
with col_info:
    st.markdown(
        """
    <div class="eval-row">
        <div class="eval-icon">🔴</div>
        <div class="eval-info">
            <div class="eval-title">Professional Pipeline
                <span class="badge badge-amber">Advanced</span>
                <span class="badge badge-gray">SQLite backed</span>
            </div>
            <div class="eval-meta">Full pipeline · NLP scoring · LLM judge · Pairwise comparisons · Multi-model</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col_btn:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.button("→ Run Eval", key="pro", use_container_width=True, disabled=True)

st.caption("→ Use the **Run Eval** page in the sidebar to launch the professional pipeline.")
