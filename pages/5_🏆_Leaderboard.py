import streamlit as st
import pandas as pd
from sqlite_store import get_leaderboard

st.title("🏆 Model Leaderboard")

lb = get_leaderboard()

if not lb:
    st.info("No results yet. Run an evaluation first.")
else:
    ldf = pd.DataFrame(lb)
    ldf.insert(0, "Rank", range(1, len(ldf) + 1))

    st.dataframe(
        ldf,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Judge Score": st.column_config.ProgressColumn(min_value=0, max_value=10),
        }
    )

    st.subheader("Score Comparison")
    chart_metric = st.selectbox("Metric", ["avg_judge_score", "avg_clarity", "avg_completeness"])
    st.bar_chart(ldf.set_index("name")[chart_metric])