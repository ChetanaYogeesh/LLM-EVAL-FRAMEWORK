"""
pages/3_Overview.py — Framework Overview Dashboard
"""

import pandas as pd
import streamlit as st

from sqlite_store import get_all_metrics_df, get_experiments, get_leaderboard

st.set_page_config(page_title="Overview", page_icon="🏠", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.page-header { font-family:'Sora',sans-serif; font-size:1.8rem; font-weight:700; color:#e6edf3; margin-bottom:4px; }
.page-sub    { color:#6e7681; font-size:0.9rem; margin-bottom:32px; }

.stat-card {
    background:#0d1117; border:1px solid #21262d; border-radius:10px;
    padding:20px 20px 16px; text-align:center;
}
.stat-value { font-size:2.4rem; font-weight:700; color:#58a6ff; font-family:'JetBrains Mono',monospace; }
.stat-label { font-size:0.78rem; color:#6e7681; text-transform:uppercase; letter-spacing:0.08em; margin-top:4px; }

.section-title { font-weight:600; color:#e6edf3; font-size:1rem; margin:24px 0 12px; }

.exp-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:10px 0; border-bottom:1px solid #21262d; font-size:0.85rem;
}
.exp-name { color:#e6edf3; font-weight:500; }
.exp-time { color:#6e7681; font-family:'JetBrains Mono',monospace; font-size:0.72rem; }

.empty-hint { color:#6e7681; font-size:0.85rem; text-align:center; padding:32px 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="page-header">🏠 Overview</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">A snapshot of your evaluation framework activity.</div>',
    unsafe_allow_html=True,
)

lb = get_leaderboard()
exps = get_experiments()
df = get_all_metrics_df()

total_responses = len(df) if not df.empty else 0
avg_score = (
    f"{df['judge_score'].mean():.2f}" if not df.empty and "judge_score" in df.columns else "—"
)

c1, c2, c3, c4 = st.columns(4)
cards = [
    ("Models Evaluated", len(lb), "🤖"),
    ("Experiments Run", len(exps), "🧪"),
    ("Total Responses", total_responses, "💬"),
    ("Avg Judge Score", avg_score, "⭐"),
]
for col, (label, value, icon) in zip([c1, c2, c3, c4], cards, strict=False):
    col.markdown(
        f"""
    <div class="stat-card">
        <div style="font-size:1.6rem;margin-bottom:6px">{icon}</div>
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
col_a, col_b = st.columns([3, 2], gap="large")

with col_a:
    st.markdown('<div class="section-title">📈 Model Performance</div>', unsafe_allow_html=True)
    if lb:
        ldf = pd.DataFrame(lb)
        cols = [
            c for c in ["avg_judge_score", "avg_clarity", "avg_completeness"] if c in ldf.columns
        ]
        if cols:
            import plotly.express as px

            fig = px.bar(
                ldf,
                x="name",
                y=cols,
                barmode="group",
                labels={"name": "Model", "value": "Score", "variable": "Metric"},
                color_discrete_sequence=["#58a6ff", "#3fb950", "#d29922"],
            )
            fig.update_layout(
                plot_bgcolor="#0d1117",
                paper_bgcolor="#0d1117",
                font_color="#8b949e",
                legend_title_text="",
                margin=dict(t=10, b=0, l=0, r=0),
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(
            '<div class="empty-hint">No model data yet.<br>Run an evaluation to see performance charts.</div>',
            unsafe_allow_html=True,
        )

with col_b:
    st.markdown('<div class="section-title">🕐 Recent Experiments</div>', unsafe_allow_html=True)
    if exps:
        for e in exps[:6]:
            name = e.get("name", "unnamed")
            ts = e.get("created_at", "")[:16].replace("T", " ")
            st.markdown(
                f"""
            <div class="exp-row">
                <span class="exp-name">🧪 {name}</span>
                <span class="exp-time">{ts}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="empty-hint">No experiments yet.<br>Run a pipeline evaluation to see history here.</div>',
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
if not df.empty and "judge_score" in df.columns:
    st.markdown('<div class="section-title">📊 Score Distribution</div>', unsafe_allow_html=True)
    import plotly.express as px

    fig2 = px.histogram(
        df,
        x="judge_score",
        nbins=20,
        color_discrete_sequence=["#58a6ff"],
        labels={"judge_score": "Judge Score"},
    )
    fig2.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font_color="#8b949e",
        showlegend=False,
        margin=dict(t=10, b=0, l=0, r=0),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title="Count"),
    )
    st.plotly_chart(fig2, use_container_width=True)
