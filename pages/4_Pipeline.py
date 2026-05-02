"""pages/4_Pipeline.py — Professional Pipeline runner + leaderboard."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Pipeline", page_icon="⚙️", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.ph { font-size:1.6rem; font-weight:700; color:#e6edf3; margin-bottom:4px; }
.ps { color:#6e7681; font-size:0.88rem; margin-bottom:24px; }
.sec { font-weight:600; color:#e6edf3; font-size:1rem; margin:20px 0 10px; }
.info-box { background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:14px 16px; font-size:0.85rem; color:#8b949e; margin-bottom:16px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="ph">⚙️ Professional Pipeline</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ps">Run the full evaluation pipeline with NLP scoring, LLM-as-a-judge, pairwise comparisons, and SQLite persistence.</div>',
    unsafe_allow_html=True,
)

tab_run, tab_lb = st.tabs(["▶ Run Pipeline", "🏆 Leaderboard"])

# ── Tab 1: Run ────────────────────────────────────────────────────────────────
with tab_run:
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown('<div class="sec">Configuration</div>', unsafe_allow_html=True)
        exp_name = st.text_input(
            "Experiment Name", value=f"run_{datetime.now().strftime('%H%M%S')}"
        )
        judge_mode = st.selectbox(
            "Judge Mode",
            ["heuristic", "auto", "openai", "claude"],
            help="heuristic = no API needed · auto = best available",
        )
        use_real = st.toggle(
            "Use Real Models (GPT-4.1 + Claude)",
            value=False,
            help="Requires OPENAI_API_KEY and ANTHROPIC_API_KEY",
        )

        st.markdown('<div class="sec">Prompt Dataset</div>', unsafe_allow_html=True)
        source = st.radio("Source", ["Built-in sample prompts", "Paste your own"], horizontal=True)

        prompts_data = []
        if source == "Built-in sample prompts":
            sample = Path("sample_prompts.json")
            if sample.exists():
                prompts_data = json.loads(sample.read_text())
                st.success(f"✅ Loaded {len(prompts_data)} sample prompts")
            else:
                prompts_data = [
                    {
                        "prompt": "What is the capital of France?",
                        "reference": "Paris is the capital of France.",
                        "category": "general",
                    }
                ]
                st.warning("sample_prompts.json not found — using 1 fallback prompt")
        else:
            raw = st.text_area("One prompt per line", height=120)
            if raw.strip():
                prompts_data = [
                    {"prompt": p.strip(), "reference": "", "category": "custom"}
                    for p in raw.splitlines()
                    if p.strip()
                ]

    with col2:
        st.markdown('<div class="sec">Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
        <div class="info-box">
            <b>Models:</b> {"GPT-4.1 + Claude" if use_real else "Mock A + Mock B"}<br>
            <b>Judge:</b> {judge_mode}<br>
            <b>Prompts:</b> {len(prompts_data)}<br>
            <b>Experiment:</b> {exp_name}
        </div>
        """,
            unsafe_allow_html=True,
        )

        if not use_real:
            st.info("Mock models run instantly without API keys — great for testing the pipeline.")
        else:
            st.warning("Real models require OPENAI_API_KEY and ANTHROPIC_API_KEY in secrets.")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    if st.button("▶ Start Pipeline", type="primary", use_container_width=False):
        if not prompts_data:
            st.error("Please provide at least one prompt.")
        else:
            tmp = Path("_tmp_prompts.json")
            tmp.write_text(json.dumps(prompts_data))
            with st.spinner(f"Running pipeline on {len(prompts_data)} prompt(s)..."):
                try:
                    from runner import run_evaluation

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(
                            run_evaluation(
                                dataset_path=str(tmp),
                                use_real_models=use_real,
                                judge_mode=judge_mode,
                                experiment_name=exp_name,
                                verbose=False,
                            )
                        )
                    finally:
                        loop.close()
                    tmp.unlink(missing_ok=True)
                    st.success(
                        f"✅ Done! {len(prompts_data)} prompts evaluated. Check the Leaderboard tab."
                    )
                    st.rerun()
                except Exception as e:
                    tmp.unlink(missing_ok=True)
                    st.error(f"Pipeline failed: {e}")
                    st.exception(e)

# ── Tab 2: Leaderboard ────────────────────────────────────────────────────────
with tab_lb:
    try:
        from sqlite_store import get_leaderboard

        lb = get_leaderboard()
    except Exception:
        lb = []

    if not lb:
        st.info("No pipeline results yet. Run the pipeline to see model rankings.")
    else:
        ldf = pd.DataFrame(lb)
        ldf.insert(0, "Rank", range(1, len(ldf) + 1))

        st.markdown('<div class="sec">Rankings</div>', unsafe_allow_html=True)
        st.dataframe(
            ldf,
            use_container_width=True,
            hide_index=True,
            column_config={"Rank": st.column_config.NumberColumn(width="small")},
        )

        st.markdown('<div class="sec">Score Comparison</div>', unsafe_allow_html=True)
        metric = st.selectbox(
            "Metric to chart",
            [
                c
                for c in [
                    "avg_judge_score",
                    "avg_clarity",
                    "avg_completeness",
                    "avg_bertscore",
                    "total_responses",
                ]
                if c in ldf.columns
            ],
        )
        fig = px.bar(
            ldf,
            x="name",
            y=metric,
            color="name",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"name": "Model", metric: metric.replace("avg_", "").replace("_", " ").title()},
        )
        fig.update_layout(
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font_color="#8b949e",
            showlegend=False,
            margin=dict(t=10, b=0, l=0, r=0),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig, use_container_width=True)
