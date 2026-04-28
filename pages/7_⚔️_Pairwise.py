import streamlit as st
import pandas as pd
from sqlite_store import get_pairwise_df

st.title("⚔️ Pairwise Comparisons")

pw = get_pairwise_df()

if pw.empty:
    st.info("No pairwise results yet.")
else:
    st.dataframe(pw, use_container_width=True)