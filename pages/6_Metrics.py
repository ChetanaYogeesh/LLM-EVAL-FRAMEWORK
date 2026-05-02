"""pages/6_Metrics.py — Deep dive into NLP and quality metrics."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Metrics", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.ph  { font-size:1.6rem; font-weight:700; color:#e6edf3; margin-bottom:4px; }
.ps  { color:#6e7681; font-size:0.88rem; margin-bottom:24px; }
.sec { font-weight:600; color:#e6edf3; font-size:1rem; margin:20px 0 10px; }
.sc  { background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:14px 16px; text-align:center; }
.sv  { font-size:1.6rem; font-weight:700; color:#58a6ff; font-family:'JetBrains Mono',monospace; }
.sl  { font-size:0.7rem; color:#6e7681; text-transform:uppercase; letter-spacing:0.07em; margin-top:2px; }
.eb  { text-align:center; padding:48px 20px; color:#6e7681; font-size:0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="ph">📊 Metrics Deep Dive</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ps">NLP scores and quality metrics from the Professional Pipeline.</div>',
    unsafe_allow_html=True,
)

try:
    from sqlite_store import get_all_metrics_df

    df = get_all_metrics_df()
except Exception:
    df = None

if df is None or df.empty:
    st.markdown(
        '<div class="eb">📭 No metrics data yet.<br>Run the Professional Pipeline to generate scoring data.</div>',
        unsafe_allow_html=True,
    )
else:
    METRIC_COLS = [
        c
        for c in [
            "judge_score",
            "bleu",
            "rouge",
            "bertscore",
            "clarity",
            "completeness",
            "conciseness",
            "tone",
        ]
        if c in df.columns
    ]

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Averages</div>', unsafe_allow_html=True)
    kpi_cols = st.columns(len(METRIC_COLS))
    for col, m in zip(kpi_cols, METRIC_COLS, strict=False):
        avg = df[m].mean()
        col.markdown(
            f'<div class="sc"><div class="sv">{avg:.2f}</div><div class="sl">{m.replace("_", " ")}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2 = st.columns(2)
    models = ["All"] + sorted(df["model"].unique().tolist()) if "model" in df.columns else ["All"]
    cats = (
        ["All"] + sorted(df["category"].unique().tolist()) if "category" in df.columns else ["All"]
    )
    sel_model = f1.selectbox("Model", models)
    sel_cat = f2.selectbox("Category", cats)

    filtered = df.copy()
    if sel_model != "All":
        filtered = filtered[filtered["model"] == sel_model]
    if sel_cat != "All" and "category" in filtered.columns:
        filtered = filtered[filtered["category"] == sel_cat]

    st.caption(f"{len(filtered)} responses shown")

    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "By Model", "Correlation", "Raw Data"])

    # ── Distributions ─────────────────────────────────────────────────────────
    with tab1:
        metric = st.selectbox("Metric", METRIC_COLS, key="dist_metric")
        fig = px.histogram(
            filtered,
            x=metric,
            nbins=20,
            color_discrete_sequence=["#58a6ff"],
            labels={metric: metric.replace("_", " ").title()},
        )
        fig.update_layout(
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font_color="#8b949e",
            showlegend=False,
            margin=dict(t=10, b=0, l=0, r=0),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", title="Count"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Box plot per model
        if "model" in filtered.columns and filtered["model"].nunique() > 1:
            fig2 = px.box(
                filtered,
                x="model",
                y=metric,
                color="model",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"model": "Model", metric: metric.replace("_", " ").title()},
            )
            fig2.update_layout(
                plot_bgcolor="#0d1117",
                paper_bgcolor="#0d1117",
                font_color="#8b949e",
                showlegend=False,
                margin=dict(t=10, b=0, l=0, r=0),
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── By Model ──────────────────────────────────────────────────────────────
    with tab2:
        if "model" not in filtered.columns or filtered["model"].nunique() < 2:
            st.info("Need at least 2 models to compare. Run the pipeline with multiple models.")
        else:
            grouped = filtered.groupby("model")[METRIC_COLS].mean().reset_index()
            fig3 = px.bar(
                grouped.melt(id_vars="model", value_vars=METRIC_COLS),
                x="variable",
                y="value",
                color="model",
                barmode="group",
                labels={"variable": "Metric", "value": "Avg Score", "model": "Model"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig3.update_layout(
                plot_bgcolor="#0d1117",
                paper_bgcolor="#0d1117",
                font_color="#8b949e",
                margin=dict(t=10, b=0, l=0, r=0),
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ── Correlation ───────────────────────────────────────────────────────────
    with tab3:
        corr = filtered[METRIC_COLS].corr().round(2)
        fig4 = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="Blues",
                zmin=-1,
                zmax=1,
                text=corr.values,
                texttemplate="%{text}",
            )
        )
        fig4.update_layout(
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font_color="#8b949e",
            margin=dict(t=10, b=0, l=0, r=0),
            height=400,
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Raw Data ──────────────────────────────────────────────────────────────
    with tab4:
        show_cols = ["model", "prompt", "judge_score"] + [
            c for c in METRIC_COLS if c != "judge_score"
        ]
        show_cols = [c for c in show_cols if c in filtered.columns]
        st.dataframe(
            filtered[show_cols].sort_values("judge_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        csv = filtered[show_cols].to_csv(index=False)
        st.download_button("⬇ Download CSV", csv, "metrics.csv", "text/csv")
