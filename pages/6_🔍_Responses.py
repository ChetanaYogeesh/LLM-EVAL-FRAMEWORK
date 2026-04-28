import streamlit as st
import pandas as pd
from sqlite_store import get_all_metrics_df

st.title("🔍 Browse Responses")

df = get_all_metrics_df()

if df.empty:
    st.info("No responses yet.")
else:
    fcol1, fcol2 = st.columns(2)
    sel_model = fcol1.selectbox("Model", ["All"] + sorted(df["model"].unique().tolist()))
    sel_category = fcol2.selectbox("Category", ["All"] + sorted(df["category"].unique().tolist()))

    filtered = df.copy()
    if sel_model != "All":
        filtered = filtered[filtered["model"] == sel_model]
    if sel_category != "All":
        filtered = filtered[filtered["category"] == sel_category]

    for _, row in filtered.iterrows():
        with st.expander(f"{row['model']} | {row['prompt'][:60]}..."):
            st.write("**Prompt:**", row["prompt"])
            st.write("**Response:**", row["response"])
            st.metric("Judge Score", row["judge_score"])