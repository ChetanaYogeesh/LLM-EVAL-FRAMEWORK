import streamlit as st
import json
import pandas as pd
import plotly.express as px
from pathlib import Path

st.title("🔍 Evaluation Results")

def load_json_results():
    results = []
    for file in ["evaluation_results.json", "evaluation_history.json"]:
        if Path(file).exists():
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                results.extend(data)
            except:
                pass
    return results

all_results = load_json_results()

tab1, tab2, tab3 = st.tabs(["🟢 Ollama Results", "🔵 CrewAI Results", "🔴 Professional"])

with tab1:
    st.subheader("🟢 Ollama Evaluator Results")
    res = [r for r in all_results if "ollama" in str(r).lower()]
    if not res:
        st.info("No Ollama results yet. Run the Ollama Evaluator first.")
    else:
        selected = st.selectbox("Select Ollama run", [f"{r.get('test_case_id','N/A')} - {r.get('timestamp','')}" for r in res])
        current = next((r for r in res if f"{r.get('test_case_id')} - {r.get('timestamp','')}" == selected), res[0])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test Case", current.get("test_case_id", "N/A"))
        col2.metric("Pass/Fail", current.get("pass_fail", "UNKNOWN"))
        col3.metric("Release", current.get("release_decision", "N/A"))
        col4.metric("Failure Mode", current.get("failure_mode", "none"))

        metrics = current.get("metrics", {})
        if metrics:
            df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
            st.plotly_chart(px.bar(df, x="Metric", y="Value", title="Key Metrics"), use_container_width=True)

        st.subheader("Safety Analysis")
        c1, c2, c3 = st.columns(3)
        c1.metric("Hallucination", "Yes" if current.get("hallucination_detected") else "No")
        c2.metric("Bias", "Yes" if current.get("bias_detected") else "No")
        c3.metric("Toxicity", "Yes" if current.get("toxicity_detected") else "No")

with tab2:
    st.subheader("🔵 CrewAI Evaluator Results")
    res = [r for r in all_results if "crew" in str(r).lower()]
    if not res:
        st.info("No CrewAI results yet. Run the CrewAI Evaluator first.")
    else:
        selected = st.selectbox("Select CrewAI run", [f"{r.get('test_case_id','N/A')} - {r.get('timestamp','')}" for r in res])
        current = next((r for r in res if f"{r.get('test_case_id')} - {r.get('timestamp','')}" == selected), res[0])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test Case", current.get("test_case_id", "N/A"))
        col2.metric("Pass/Fail", current.get("pass_fail", "UNKNOWN"))
        col3.metric("Release", current.get("release_decision", "N/A"))
        col4.metric("Failure Mode", current.get("failure_mode", "none"))

        metrics = current.get("metrics", {})
        if metrics:
            df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
            st.plotly_chart(px.bar(df, x="Metric", y="Value", title="Key Metrics"), use_container_width=True)

        st.subheader("Safety Analysis")
        c1, c2, c3 = st.columns(3)
        c1.metric("Hallucination", "Yes" if current.get("hallucination_detected") else "No")
        c2.metric("Bias", "Yes" if current.get("bias_detected") else "No")
        c3.metric("Toxicity", "Yes" if current.get("toxicity_detected") else "No")

with tab3:
    st.subheader("🔴 Professional Pipeline Results")
    st.info("Professional results are available in **Leaderboard**, **Responses**, **Pairwise**, and **Metrics** pages.")