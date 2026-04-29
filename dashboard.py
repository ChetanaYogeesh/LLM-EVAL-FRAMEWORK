"""
dashboard.py — LLM Eval Framework · Main Dashboard
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

# ── Local imports (must still be at top) ──────────────────────────────────────
from comparator import generate_comparison_report
from llm_judge import judge_response
from runners import evaluate_prompts, mock_model_a, mock_model_b
from scorer import compute_all_metrics
from sqlite_store import (
    DB_PATH,
    create_experiment,
    get_all_metrics_df,
    get_experiments,
    get_leaderboard,
    get_pairwise_df,
    init_db,
    insert_metrics,
    insert_pairwise,
    insert_prompt,
    insert_response,
    upsert_model,
)

# ── Path setup AFTER imports ──────────────────────────────────────────────────

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Eval Framework",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ───────────────────────────────────────────────────────────────────
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


# ── Load JSON results ────────────────────────────────────────────────────────
def load_json_results():
    results = []
    for file in ["evaluation_results.json", "evaluation_history.json"]:
        path = Path(file)
        if path.exists():
            try:
                with path.open() as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                results.extend(data)
            except Exception as e:
                st.warning(f"Failed to load {file}: {e}")
    return results


all_json_results = load_json_results()


# ── Async runner ─────────────────────────────────────────────────────────────
def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def run_eval_pipeline(prompts_data, use_real, judge_mode, exp_name, progress_cb=None):
    if use_real:
        from models.runners import run_claude, run_openai

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
        resp_texts = {}

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
            progress_cb((i + 1) / total, f"Processed {i + 1}/{total}")

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
    st.markdown("---")
    st.caption(f"DB: `{DB_PATH.name}`")

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🧪 LLM Evaluation Framework")

    lb = get_leaderboard()
    exps = get_experiments()
    df = get_all_metrics_df()

    responses_count = len(df) if not df.empty else 0
    avg_score = round(df["judge_score"].mean(), 2) if not df.empty else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models Evaluated", len(lb))
    c2.metric("Total Responses", responses_count)
    c3.metric("Experiments Run", len(exps))
    c4.metric("Avg Judge Score", avg_score)

# ══════════════════════════════════════════════════════════════════════════════
# LAUNCH EVALUATORS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚀 Launch Evaluators":
    st.title("🚀 Launch Evaluators")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run Ollama"):
            result = subprocess.run(
                [sys.executable, str(ROOT / "ollama_evaluator.py")],
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            st.code(result.stdout if result.returncode == 0 else result.stderr)

    with col2:
        if st.button("Run CrewAI"):
            result = subprocess.run(
                [sys.executable, str(ROOT / "crewai_evaluator.py")],
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            st.code(result.stdout if result.returncode == 0 else result.stderr)

# ══════════════════════════════════════════════════════════════════════════════
# RESPONSES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Responses":
    st.title("🔍 Responses")

    df = get_all_metrics_df()

    if df.empty:
        st.info("No responses yet.")
    else:
        for _, row in df.iterrows():
            with st.expander(f"{row['model']} — {row['judge_score']}/10"):
                st.write(row["prompt"])
                st.success(row["response"])

# ══════════════════════════════════════════════════════════════════════════════
# PAIRWISE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚔️ Pairwise":
    pw = get_pairwise_df()

    if pw.empty:
        st.info("No pairwise results.")
    else:
        for _, row in pw.iterrows():
            with st.expander(row["prompt"][:80]):
                st.write(row["model_a"], row["score_a"])
                st.write(row["model_b"], row["score_b"])

                try:
                    bd = (
                        json.loads(row["breakdown"])
                        if isinstance(row["breakdown"], str)
                        else row["breakdown"]
                    )
                    for k, v in bd.items():
                        if isinstance(v, list | tuple) and len(v) == 3:
                            st.write(k, v)
                except Exception:
                    continue

# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Metrics":
    df = get_all_metrics_df()

    if df.empty:
        st.info("No metrics yet.")
    else:
        st.dataframe(df)

# ── Footer ───────────────────────────────────────────────────────────────────
st.caption("✅ Ruff-clean dashboard")
