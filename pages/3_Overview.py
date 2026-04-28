import pandas as pd
import streamlit as st

from sqlite_store import get_experiments, get_leaderboard

st.title("🏠 Overview")

lb = get_leaderboard()
exps = get_experiments()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Models Evaluated", len(lb))
col2.metric("Experiments Run", len(exps))
col3.metric("Total Responses", "N/A")  # You can enhance this
col4.metric("Avg Judge Score", "—")

st.markdown("---")

col_a, col_b = st.columns([3, 2])

with col_a:
    st.subheader("📈 Model Performance")
    if lb:
        ldf = pd.DataFrame(lb)
        st.bar_chart(ldf.set_index("name")[["avg_judge_score", "avg_clarity", "avg_completeness"]])
    else:
        st.info("No data yet. Run an evaluation.")

with col_b:
    st.subheader("🕐 Recent Experiments")
    if exps:
        for e in exps[:5]:
            st.write(f"**{e['name']}** — {e.get('created_at', '')}")
    else:
        st.info("No experiments yet.")
