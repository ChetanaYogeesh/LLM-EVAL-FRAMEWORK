import asyncio
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from runner import run_evaluation

st.title("🚀 Run Professional Pipeline")

col1, col2 = st.columns([2, 1])

with col1:
    exp_name = st.text_input("Experiment Name", value=f"run_{datetime.now().strftime('%H%M%S')}")
    judge_mode = st.selectbox("Judge Mode", ["heuristic", "auto", "openai", "claude"])
    use_real = st.toggle("Use Real Models (GPT + Claude)", value=False)

    st.markdown("#### Prompt Dataset")
    dataset_choice = st.radio("Source", ["Built-in sample", "Paste prompts"])

    prompts_data = []

    if dataset_choice == "Built-in sample":
        sample_path = Path("sample_prompts.json")
        if sample_path.exists():
            with sample_path.open() as f:
                prompts_data = json.load(f)
            st.success(f"Loaded {len(prompts_data)} sample prompts")
        else:
            prompts_data = [
                {
                    "prompt": "What is the capital of France?",
                    "reference": "Paris is the capital of France.",
                    "category": "general",
                }
            ]
            st.warning("sample_prompts.json not found — using fallback prompt")
    else:
        raw = st.text_area("Paste prompts (one per line)")
        if raw.strip():
            prompts_data = [
                {"prompt": p.strip(), "reference": "", "category": "custom"}
                for p in raw.strip().splitlines()
                if p.strip()
            ]

with col2:
    st.markdown("**Pipeline Summary**")
    st.write(f"Models: {'Real (GPT + Claude)' if use_real else 'Mock'}")
    st.write(f"Judge: {judge_mode}")
    st.write(f"Prompts: {len(prompts_data)}")

if st.button("▶️ Start Evaluation", type="primary"):
    if not prompts_data:
        st.error("Please provide at least one prompt.")
    else:
        # Write prompts to a temp file so run_evaluation can load them
        tmp_path = Path("_tmp_prompts.json")
        with tmp_path.open("w") as f:
            json.dump(prompts_data, f)

        with st.spinner("Running full professional evaluation pipeline..."):
            try:
                asyncio.run(
                    run_evaluation(
                        dataset_path=str(tmp_path),
                        use_real_models=use_real,
                        judge_mode=judge_mode,
                        experiment_name=exp_name,
                        verbose=False,
                    )
                )
                tmp_path.unlink(missing_ok=True)
                st.success(
                    f"✅ Evaluation complete! {len(prompts_data)} prompts evaluated. "
                    "Check **Leaderboard** for results."
                )
            except Exception as e:
                tmp_path.unlink(missing_ok=True)
                st.error(f"Evaluation failed: {e}")
                st.exception(e)
