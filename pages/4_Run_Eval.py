import json
from datetime import datetime

import streamlit as st

from runner import run_eval_pipeline  # Import from root runner.py

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
        try:
            with open("sample_prompts.json") as f:
                prompts_data = json.load(f)
            st.success(f"Loaded {len(prompts_data)} sample prompts")
        except Exception:
            prompts_data = [
                {
                    "prompt": "What is the capital of France?",
                    "reference": "Paris is the capital of France.",
                    "category": "general",
                }
            ]
            st.warning("Using fallback sample prompt")
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
    st.write(f"Models: {'Real' if use_real else 'Mock'}")
    st.write(f"Judge: {judge_mode}")
    st.write(f"Prompts: {len(prompts_data)}")

if st.button("▶️ Start Evaluation", type="primary"):
    if not prompts_data:
        st.error("Please provide at least one prompt")
    else:
        with st.spinner("Running full professional evaluation pipeline..."):
            try:
                runner_names, _ = run_eval_pipeline(prompts_data, use_real, judge_mode, exp_name)
                st.success(
                    f"✅ Evaluation completed! {len(prompts_data)} prompts × {len(runner_names)} models"
                )
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
