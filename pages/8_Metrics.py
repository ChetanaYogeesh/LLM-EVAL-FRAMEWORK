"""
pages/8_Metrics.py — Metrics Deep Dive
"""

import pandas as pd
import streamlit as st

from sqlite_store import get_all_metrics_df, init_db

# Initialize database (ensures all tables exist)
init_db()

st.title("📊 Metrics Deep Dive")

df = get_all_metrics_df()

if df.empty:
    st.info("No metrics data yet. Run some evaluations first!")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Correlation", "📋 Raw Data"])

with tab1:
    st.subheader("Metric Distributions")
    metric = st.selectbox(
        "Select Metric", ["judge_score", "bleu", "rouge", "bertscore", "clarity", "completeness"]
    )

    if not df[metric].empty:
        fig = pd.DataFrame(df[metric].value_counts().sort_index())
        st.bar_chart(fig)
    else:
        st.warning(f"No data available for {metric}")

with tab2:
    st.subheader("Metric Correlation")
    numeric_cols = ["judge_score", "bleu", "rouge", "bertscore", "clarity", "completeness"]
    available_cols = [col for col in numeric_cols if col in df.columns]

    if len(available_cols) > 1:
        corr = df[available_cols].corr().round(2)
        st.dataframe(corr, use_container_width=True)
    else:
        st.info("Not enough numeric metrics to show correlation.")

with tab3:
    st.subheader("Raw Metrics Data")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

st.caption("💡 Tip: Run evaluations from the Launch Evaluators page to populate metrics.")
