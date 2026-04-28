import streamlit as st
import pandas as pd
from sqlite_store import get_all_metrics_df

st.title("📊 Metrics Deep Dive")

df = get_all_metrics_df()

if df.empty:
    st.info("No metrics data yet.")
else:
    tab1, tab2 = st.tabs(["Distributions", "Correlation"])

    with tab1:
        metric = st.selectbox("Metric", ["judge_score", "bleu", "rouge", "bertscore"])
        st.bar_chart(df[metric].value_counts().sort_index())

    with tab2:
        st.dataframe(df[["judge_score", "bleu", "rouge", "bertscore"]].corr().round(2), use_container_width=True)