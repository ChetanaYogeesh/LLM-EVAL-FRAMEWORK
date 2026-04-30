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

.eval-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 28px 24px;
    transition: border-color 0.2s, box-shadow 0.2s;
    height: 100%;
}
.eval-card:hover { border-color: #58a6ff; box-shadow: 0 0 20px rgba(88,166,255,0.08); }

.card-icon { font-size: 2.2rem; margin-bottom: 10px; }

.card-title {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: #e6edf3;
    margin: 0 0 4px 0;
}
.card-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #6e7681;
    margin: 0 0 16px 0;
    letter-spacing: 0.03em;
}
.card-desc {
    font-size: 0.85rem;
    color: #8b949e;
    line-height: 1.5;
    margin-bottom: 20px;
}
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-right: 4px;
    margin-bottom: 12px;
}
.badge-green { background: #0d2818; color: #3fb950; border: 1px solid #238636; }
.badge-blue  { background: #0c1f3f; color: #58a6ff; border: 1px solid #1f6feb; }
.badge-amber { background: #2d1e0f; color: #d29922; border: 1px solid #9e6a03; }
.badge-gray  { background: #161b22; color: #8b949e; border: 1px solid #30363d; }

.warn-box {
    background: #2d1e0f;
    border: 1px solid #9e6a03;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #d29922;
    margin-bottom: 16px;
}
.page-header {
    font-family: 'Sora', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 4px;
}
.page-sub {
    color: #6e7681;
    font-size: 0.9rem;
    margin-bottom: 32px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="page-header">🚀 Launch Evaluators</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Run any evaluator with one click. Results are saved automatically.</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown(
        """
    <div class="eval-card">
        <div class="card-icon">🟢</div>
        <div class="card-title">Ollama Evaluator</div>
        <div class="card-subtitle">LOCAL · OFFLINE · llama3.2</div>
        <div class="card-desc">
            Runs a lightweight LLM evaluation pipeline using your local Ollama instance.
            Falls back to OpenRouter when Ollama is not reachable.
        </div>
        <span class="badge badge-green">Free</span>
        <span class="badge badge-gray">Local first</span>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("▶ Run Ollama Evaluator", type="primary", use_container_width=True, key="ollama"):
        with st.spinner("Running..."):
            result = subprocess.run(
                [sys.executable, str(ROOT / "ollama_evaluator.py")],
                capture_output=True,
                text=True,
                cwd=ROOT,
                timeout=90,
            )
        if result.returncode == 0:
            st.success("✅ Completed!")
            with st.expander("View output"):
                st.code(result.stdout or "No output.")
        else:
            st.error("❌ Failed")
            with st.expander("View error"):
                st.code(result.stderr or "No error output captured.")

with col2:
    st.markdown(
        """
    <div class="eval-card">
        <div class="card-icon">🔵</div>
        <div class="card-title">CrewAI Evaluator</div>
        <div class="card-subtitle">MULTI-AGENT · HIERARCHICAL · OpenRouter</div>
        <div class="card-desc">
            Orchestrates six specialized AI agents in a hierarchical crew.
            Evaluates trace quality, safety, cost, latency, and regressions.
        </div>
        <span class="badge badge-blue">OpenRouter</span>
        <span class="badge badge-gray">Requires API key</span>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("▶ Run CrewAI Evaluator", type="primary", use_container_width=True, key="crew"):
        with st.spinner("Running multi-agent evaluation... (up to 3 min)"):
            result = subprocess.run(
                [sys.executable, str(ROOT / "crewai_evaluator.py")],
                capture_output=True,
                text=True,
                cwd=ROOT,
                timeout=180,
            )
        if result.returncode == 0:
            st.success("✅ Completed!")
            with st.expander("View output"):
                st.code(result.stdout or "No output.")
        else:
            st.error("❌ Failed")
            with st.expander("View error"):
                st.code(result.stderr or "No error output captured.")

with col3:
    st.markdown(
        """
    <div class="eval-card">
        <div class="card-icon">🔴</div>
        <div class="card-title">Professional Pipeline</div>
        <div class="card-subtitle">FULL PIPELINE · SQLite · Multi-model</div>
        <div class="card-desc">
            Runs the full evaluation pipeline with NLP scoring, LLM-as-a-judge,
            pairwise comparisons, and SQLite persistence.
        </div>
        <span class="badge badge-amber">Advanced</span>
        <span class="badge badge-gray">SQLite backed</span>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.info("→ Use the **Run Eval** page in the sidebar to launch the full pipeline.")
