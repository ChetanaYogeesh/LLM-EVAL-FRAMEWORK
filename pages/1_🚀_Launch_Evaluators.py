import streamlit as st
import subprocess
from pathlib import Path

st.title("🚀 Launch Evaluators")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🟢 Ollama Evaluator")
    st.caption("Fast • Local • llama3.2")
    if st.button("▶️ Run Ollama Evaluator", type="primary", use_container_width=True):
        with st.spinner("Running Ollama Evaluator..."):
            result = subprocess.run(
                ["python", str(Path(__file__).parent.parent / "ollama_evaluator.py")],
                capture_output=True, text=True, cwd=Path(__file__).parent.parent
            )
            if result.returncode == 0:
                st.success("✅ Ollama Evaluator completed!")
                st.code(result.stdout)
            else:
                st.error("❌ Failed")
                st.code(result.stderr)

with col2:
    st.subheader("🔵 CrewAI Evaluator")
    st.caption("Multi-agent • Hierarchical")
    if st.button("▶️ Run CrewAI Evaluator", type="primary", use_container_width=True):
        with st.spinner("Running CrewAI Evaluator..."):
            result = subprocess.run(
                ["python", str(Path(__file__).parent.parent / "crewai_evaluator.py")],
                capture_output=True, text=True, cwd=Path(__file__).parent.parent
            )
            if result.returncode == 0:
                st.success("✅ CrewAI Evaluator completed!")
                st.code(result.stdout)
            else:
                st.error("❌ Failed")
                st.code(result.stderr)

with col3:
    st.subheader("🔴 Professional Pipeline")
    st.caption("Full SQLite evaluation")
    st.info("Go to the **Run Eval** page in the sidebar.")