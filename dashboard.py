"""
dashboard.py — LLM Eval Framework · Main Dashboard
Run with: streamlit run dashboard.py
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Setup — sys.path must be set before local imports ─────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from comparator import generate_comparison_report  # noqa: E402
from llm_judge import judge_response  # noqa: E402
from runners import evaluate_prompts, mock_model_a, mock_model_b  # noqa: E402
from scorer import compute_all_metrics  # noqa: E402
from sqlite_store import (  # noqa: E402
    create_experiment,
    get_connection,
    init_db,
    insert_metrics,
    insert_pairwise,
    insert_prompt,
    insert_response,
    upsert_model,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Eval Framework",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
[data-testid="stSidebar"] { background: #0f1117; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="metric-container"] {
    background: #1e2130; border: 1px solid #2e3250; border-radius: 10px; padding: 12px 18px;
}
thead tr th { background: #1e2130 !important; }
</style>
""",
    unsafe_allow_html=True,
)

init_db()


# ── Load JSON results from Ollama & CrewAI evaluators ────────────────────────
def load_json_results() -> list:
    results = []
    for file in ["evaluation_results.json", "evaluation_history.json"]:
        if Path(file).exists():
            try:
                with open(file) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                results.extend(data)
            except Exception:
                pass
    return results


all_json_results = load_json_results()


# ── Helpers ───────────────────────────────────────────────────────────────────
def badge(text: str, color: str = "blue") -> str:
    return f'<span class="badge-{color}">{text}</span>'


def query(sql: str, params: tuple = ()) -> list:
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def get_all_metrics_df() -> pd.DataFrame:
    rows = query("""
        SELECT m.name AS model, p.prompt, p.category, p.reference_answer,
               r.response, mt.judge_score, mt.bleu, mt.rouge, mt.bertscore,
               mt.clarity, mt.completeness, mt.conciseness, mt.tone,
               r.created_at
        FROM metrics mt JOIN responses r ON r.id = mt.response_id
        JOIN models m ON m.id = r.model_id
        JOIN prompts p ON p.id = r.prompt_id
    """)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def get_pairwise_df() -> pd.DataFrame:
    rows = query("""
        SELECT pr.winner, pr.score_a, pr.score_b, pr.breakdown, pr.created_at,
               p.prompt, p.category, ma.name AS model_a, mb.name AS model_b
        FROM pairwise_results pr JOIN prompts p ON p.id = pr.prompt_id
        JOIN models ma ON ma.id = pr.model_a_id
        JOIN models mb ON mb.id = pr.model_b_id
        ORDER BY pr.created_at DESC
    """)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Async helper ──────────────────────────────────────────────────────────────
def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ── Eval pipeline ─────────────────────────────────────────────────────────────
def run_eval_pipeline(prompts_data, use_real, judge_mode, exp_name, progress_cb=None):
    if use_real:
        from runners import run_claude, run_openai

        runners = [run_openai, run_claude]
        runner_names = ["GPT-4.1", "Claude-Sonnet"]
    else:
        runners = [mock_model_a, mock_model_b]
        runner_names = ["MockModel-A", "MockModel-B"]

    create_experiment(exp_name, config={"judge": judge_mode, "real": use_real})
    model_ids = {name: upsert_model(name) for name in runner_names}
    results = _run_async(evaluate_prompts(prompts_data, runners, runner_names))
    total = len(results)
    pairwise_reports = []

    for i, item in enumerate(results):
        prompt_text = item["prompt"]
        reference = item.get("reference", "")
        category = item.get("category", "general")
        prompt_id = insert_prompt(prompt_text, reference, category)
        resp_texts: dict = {}

        for model_name in runner_names:
            text = item["responses"].get(model_name, "")
            resp_texts[model_name] = text
            resp_id = insert_response(model_ids[model_name], prompt_id, text)
            nlp = compute_all_metrics(text, reference)
            judged = judge_response(prompt_text, text, reference, judge=judge_mode)
            scores = {
                **nlp,
                "judge_score": judged.get("overall", 5),
                "clarity": judged.get("clarity", 5),
                "completeness": judged.get("completeness", 5),
                "conciseness": 5,
                "tone": 5,
            }
            insert_metrics(resp_id, scores)

        if len(runner_names) >= 2:
            ma, mb = runner_names[0], runner_names[1]
            rpt = generate_comparison_report(prompt_text, resp_texts[ma], resp_texts[mb], reference)
            insert_pairwise(
                prompt_id,
                model_ids[ma],
                model_ids[mb],
                rpt["winner"],
                rpt["score_a"],
                rpt["score_b"],
                {k: list(v) for k, v in rpt["breakdown"].items()},
            )
            pairwise_reports.append(rpt)

        if progress_cb:
            progress_cb((i + 1) / total, f"Processed {i + 1}/{total}: {prompt_text[:50]}…")

    return runner_names, pairwise_reports


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧪 LLM Eval Framework")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "🚀 Launch Evaluators",
            "🔍 Results",
            "🏆 Leaderboard",
            "🔍 Responses",
            "⚔️ Pairwise",
            "📊 Metrics",
        ],
        label_visibility="collapsed",
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LAUNCH EVALUATORS
# ══════════════════════════════════════════════════════════════════════════════
if page == "🚀 Launch Evaluators":
    st.title("🚀 Launch Evaluators")
    st.markdown("### Run any evaluator with one click")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🟢 Ollama Evaluator")
        st.caption("Fast • Local • No API keys required")
        if st.button("▶️ Run Ollama Evaluator", type="primary", use_container_width=True):
            with st.spinner("Running Ollama Evaluator..."):
                try:
                    result = subprocess.run(
                        ["python", str(ROOT / "ollama_evaluator.py")],
                        capture_output=True,
                        text=True,
                        cwd=ROOT,
                        timeout=120,
                    )
                    if result.returncode == 0:
                        st.success("✅ Ollama Evaluator finished successfully!")
                        st.code(result.stdout)
                    else:
                        st.error("❌ Ollama Evaluator failed")
                        st.code(result.stderr)
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.markdown("### 🔵 CrewAI Evaluator")
        st.caption("Multi-agent • Hierarchical reasoning")
        if st.button("▶️ Run CrewAI Evaluator", type="primary", use_container_width=True):
            with st.spinner("Running CrewAI Evaluator..."):
                try:
                    result = subprocess.run(
                        ["python", str(ROOT / "crewai_evaluator.py")],
                        capture_output=True,
                        text=True,
                        cwd=ROOT,
                        timeout=180,
                    )
                    if result.returncode == 0:
                        st.success("✅ CrewAI Evaluator finished successfully!")
                        st.code(result.stdout)
                    else:
                        st.error("❌ CrewAI Evaluator failed")
                        st.code(result.stderr)
                except Exception as e:
                    st.error(f"Error: {e}")

    with col3:
        st.markdown("### 🔴 Professional Pipeline")
        st.caption("Full SQLite • Multi-model • Advanced scoring")
        if st.button("▶️ Run Professional Pipeline", type="primary", use_container_width=True):
            st.info(
                "Go to the **Run Eval** page in the sidebar to launch the full professional pipeline."
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Results":
    st.title("🔍 Evaluation Results")
    tab1, tab2, tab3 = st.tabs(
        ["🟢 Ollama Results", "🔵 CrewAI Results", "🔴 Professional Pipeline"]
    )

    with tab1:
        st.subheader("🟢 Ollama Evaluator Results")
        ollama_res = [r for r in all_json_results if "ollama" in str(r).lower()]
        if not ollama_res:
            st.info("No Ollama results yet. Run the Ollama Evaluator first.")
        else:
            selected = st.selectbox(
                "Select run",
                [f"{r.get('test_case_id', 'N/A')} - {r.get('timestamp', '')}" for r in ollama_res],
            )
            current = next(
                (
                    r
                    for r in ollama_res
                    if f"{r.get('test_case_id')} - {r.get('timestamp', '')}" == selected
                ),
                ollama_res[0],
            )
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Test Case", current.get("test_case_id", "N/A"))
            col2.metric("Pass/Fail", current.get("pass_fail", "UNKNOWN"))
            col3.metric("Release", current.get("release_decision", "N/A"))
            col4.metric("Failure Mode", current.get("failure_mode", "none"))
            metrics = current.get("metrics", {})
            if metrics:
                df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
                fig = px.bar(df, x="Metric", y="Value", title="Key Metrics")
                st.plotly_chart(fig, use_container_width=True)
            st.subheader("Safety Analysis")
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Hallucination",
                "Yes" if current.get("hallucination_detected") else "No",
            )
            c2.metric("Bias", "Yes" if current.get("bias_detected") else "No")
            c3.metric("Toxicity", "Yes" if current.get("toxicity_detected") else "No")

    with tab2:
        st.subheader("🔵 CrewAI Evaluator Results")
        crew_res = [r for r in all_json_results if "crew" in str(r).lower()]
        if not crew_res:
            st.info("No CrewAI results yet. Run the CrewAI Evaluator first.")
        else:
            selected = st.selectbox(
                "Select run",
                [f"{r.get('test_case_id', 'N/A')} - {r.get('timestamp', '')}" for r in crew_res],
            )
            current = next(
                (
                    r
                    for r in crew_res
                    if f"{r.get('test_case_id')} - {r.get('timestamp', '')}" == selected
                ),
                crew_res[0],
            )
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Test Case", current.get("test_case_id", "N/A"))
            col2.metric("Pass/Fail", current.get("pass_fail", "UNKNOWN"))
            col3.metric("Release", current.get("release_decision", "N/A"))
            col4.metric("Failure Mode", current.get("failure_mode", "none"))
            metrics = current.get("metrics", {})
            if metrics:
                df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
                fig = px.bar(df, x="Metric", y="Value", title="Key Metrics")
                st.plotly_chart(fig, use_container_width=True)
            st.subheader("Safety Analysis")
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Hallucination",
                "Yes" if current.get("hallucination_detected") else "No",
            )
            c2.metric("Bias", "Yes" if current.get("bias_detected") else "No")
            c3.metric("Toxicity", "Yes" if current.get("toxicity_detected") else "No")

    with tab3:
        st.subheader("🔴 Professional Pipeline Results")
        st.info(
            "View results in the **Leaderboard**, **Responses**, **Pairwise**, and **Metrics** pages."
        )

st.caption("✅ Fixed • Clear buttons • Separate result views for each evaluator")
