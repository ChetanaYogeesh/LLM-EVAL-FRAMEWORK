"""pages/4_Professional_Pipeline.py — Run, leaderboard, pairwise, metrics in one place."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Professional Pipeline", page_icon="⚙️", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.ph  { font-size:1.6rem; font-weight:700; color:#e6edf3; margin-bottom:4px; }
.ps  { color:#6e7681; font-size:0.88rem; margin-bottom:24px; }
.sec { font-weight:600; color:#e6edf3; font-size:1rem; margin:16px 0 10px; }
.info-box { background:#0d1117; border:1px solid #21262d; border-radius:8px;
            padding:14px 16px; font-size:0.85rem; color:#8b949e; margin-bottom:16px; }
.sc  { background:#0d1117; border:1px solid #21262d; border-radius:8px;
       padding:14px; text-align:center; }
.sv  { font-size:1.6rem; font-weight:700; color:#58a6ff; font-family:'JetBrains Mono',monospace; }
.sl  { font-size:0.7rem; color:#6e7681; text-transform:uppercase; letter-spacing:0.07em; margin-top:2px; }
.pw-model { font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#6e7681; }
.pw-score { font-size:1.3rem; font-weight:700; color:#e6edf3; }
.win-a  { color:#3fb950; font-weight:700; }
.win-b  { color:#58a6ff; font-weight:700; }
.win-tie{ color:#d29922; font-weight:700; }
.eb  { text-align:center; padding:40px 20px; color:#6e7681; font-size:0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="ph">⚙️ Professional Pipeline</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ps">Full evaluation pipeline with NLP scoring, LLM judge, pairwise comparisons, and SQLite persistence.</div>',
    unsafe_allow_html=True,
)

# Load data once
try:
    from sqlite_store import get_all_metrics_df, get_experiments, get_leaderboard, get_pairwise_df

    lb = get_leaderboard()
    exps = get_experiments()
    df = get_all_metrics_df()
    pw = get_pairwise_df()
    db_ok = True
except Exception:
    lb = exps = []
    df = pw = None
    db_ok = False

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
    if df is not None and not df.empty and c in df.columns
]

tab_run, tab_lb, tab_pw, tab_metrics = st.tabs(
    ["▶ Run Pipeline", "🏆 Leaderboard", "⚔️ Pairwise", "📊 Metrics"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RUN
# ══════════════════════════════════════════════════════════════════════════════
with tab_run:
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown('<div class="sec">Configuration</div>', unsafe_allow_html=True)
        exp_name = st.text_input(
            "Experiment Name", value=f"run_{datetime.now().strftime('%H%M%S')}"
        )
        judge_mode = st.selectbox(
            "Judge Mode",
            ["heuristic", "auto", "openai", "claude"],
            help="heuristic = no API needed · auto = best available",
        )
        use_real = st.toggle(
            "Use Real Models (GPT-4.1 + Claude)",
            value=False,
            help="Requires OPENAI_API_KEY and ANTHROPIC_API_KEY",
        )

        st.markdown('<div class="sec">Prompt Dataset</div>', unsafe_allow_html=True)
        source = st.radio("Source", ["Built-in sample prompts", "Paste your own"], horizontal=True)

        prompts_data = []
        if source == "Built-in sample prompts":
            sample = Path("sample_prompts.json")
            if sample.exists():
                prompts_data = json.loads(sample.read_text())
                st.success(f"✅ Loaded {len(prompts_data)} sample prompts")
            else:
                prompts_data = [
                    {
                        "prompt": "What is the capital of France?",
                        "reference": "Paris is the capital of France.",
                        "category": "general",
                    }
                ]
                st.warning("sample_prompts.json not found — using 1 fallback prompt")
        else:
            raw = st.text_area("One prompt per line", height=120)
            if raw.strip():
                prompts_data = [
                    {"prompt": p.strip(), "reference": "", "category": "custom"}
                    for p in raw.splitlines()
                    if p.strip()
                ]

    with col2:
        st.markdown('<div class="sec">Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
        <div class="info-box">
            <b>Models:</b> {"GPT-4.1 + Claude" if use_real else "Mock A + Mock B"}<br>
            <b>Judge:</b> {judge_mode}<br>
            <b>Prompts:</b> {len(prompts_data)}<br>
            <b>Experiment:</b> {exp_name}
        </div>
        """,
            unsafe_allow_html=True,
        )

        if not use_real:
            st.info("Mock models run instantly — great for testing the pipeline without API keys.")
        else:
            st.warning("Real models require OPENAI_API_KEY and ANTHROPIC_API_KEY in secrets.")

        # Quick stats
        if db_ok:
            st.markdown('<div class="sec">Pipeline Stats</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.markdown(
                f'<div class="sc"><div class="sv">{len(lb)}</div><div class="sl">Models</div></div>',
                unsafe_allow_html=True,
            )
            c2.markdown(
                f'<div class="sc"><div class="sv">{len(exps)}</div><div class="sl">Experiments</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    if st.button("▶ Start Pipeline", type="primary"):
        if not prompts_data:
            st.error("Please provide at least one prompt.")
        else:
            tmp = Path("_tmp_prompts.json")
            tmp.write_text(json.dumps(prompts_data))
            with st.spinner(f"Running pipeline on {len(prompts_data)} prompt(s)..."):
                try:
                    from runner import run_evaluation

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(
                            run_evaluation(
                                dataset_path=str(tmp),
                                use_real_models=use_real,
                                judge_mode=judge_mode,
                                experiment_name=exp_name,
                                verbose=False,
                            )
                        )
                    finally:
                        loop.close()
                    tmp.unlink(missing_ok=True)
                    st.success(
                        f"✅ Done! {len(prompts_data)} prompts evaluated. Check the Leaderboard tab."
                    )
                    st.rerun()
                except Exception as e:
                    tmp.unlink(missing_ok=True)
                    st.error(f"Pipeline failed: {e}")
                    st.exception(e)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_lb:
    if not lb:
        st.markdown(
            '<div class="eb">📭 No pipeline results yet.<br>Run the pipeline above to see model rankings.</div>',
            unsafe_allow_html=True,
        )
    else:
        import pandas as pd

        ldf = pd.DataFrame(lb)
        ldf.insert(0, "Rank", range(1, len(ldf) + 1))

        st.markdown('<div class="sec">Rankings</div>', unsafe_allow_html=True)
        st.dataframe(
            ldf,
            use_container_width=True,
            hide_index=True,
            column_config={"Rank": st.column_config.NumberColumn(width="small")},
        )

        st.markdown('<div class="sec">Score Comparison</div>', unsafe_allow_html=True)
        avail = [
            c
            for c in [
                "avg_judge_score",
                "avg_clarity",
                "avg_completeness",
                "avg_bertscore",
                "total_responses",
            ]
            if c in ldf.columns
        ]
        metric = st.selectbox("Metric", avail, key="lb_metric")
        fig = px.bar(
            ldf,
            x="name",
            y=metric,
            color="name",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"name": "Model", metric: metric.replace("avg_", "").replace("_", " ").title()},
        )
        fig.update_layout(
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font_color="#8b949e",
            showlegend=False,
            margin=dict(t=10, b=0, l=0, r=0),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PAIRWISE
# ══════════════════════════════════════════════════════════════════════════════
with tab_pw:
    if pw is None or pw.empty:
        st.markdown(
            '<div class="eb">📭 No pairwise data yet.<br>Run the pipeline with at least 2 models to generate head-to-head comparisons.</div>',
            unsafe_allow_html=True,
        )
    else:
        total = len(pw)
        a_wins = int((pw["winner"] == "A").sum())
        b_wins = int((pw["winner"] == "B").sum())
        ties = int((pw["winner"] == "tie").sum())
        ma = pw["model_a"].iloc[0]
        mb = pw["model_b"].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Comparisons", total)
        c2.metric(f"{ma} Wins", a_wins)
        c3.metric(f"{mb} Wins", b_wins)
        c4.metric("Ties", ties)

        col_chart, _ = st.columns([1, 2])
        with col_chart:
            fig = px.pie(
                values=[a_wins, b_wins, ties],
                names=[ma, mb, "Tie"],
                color_discrete_sequence=["#3fb950", "#58a6ff", "#d29922"],
                hole=0.55,
            )
            fig.update_layout(
                paper_bgcolor="#0d1117",
                font_color="#8b949e",
                margin=dict(t=10, b=0, l=0, r=0),
                height=200,
                legend=dict(orientation="h", y=-0.2),
            )
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="sec">All Comparisons</div>', unsafe_allow_html=True)
        f1, f2 = st.columns(2)
        cats = (
            ["All"] + sorted(pw["category"].dropna().unique().tolist())
            if "category" in pw.columns
            else ["All"]
        )
        sel_cat = f1.selectbox("Category", cats, key="pw_cat")
        sel_win = f2.selectbox("Winner", ["All", "A", "B", "tie"], key="pw_win")

        filtered = pw.copy()
        if sel_cat != "All" and "category" in filtered.columns:
            filtered = filtered[filtered["category"] == sel_cat]
        if sel_win != "All":
            filtered = filtered[filtered["winner"] == sel_win]

        st.caption(f"Showing {len(filtered)} of {total} comparisons")

        for _, row in filtered.iterrows():
            winner = row.get("winner", "?")
            win_cls = "win-a" if winner == "A" else "win-b" if winner == "B" else "win-tie"
            with st.expander(f"{row.get('prompt', '')[:80]}…"):
                mc1, mc2, mc3 = st.columns([2, 2, 1])
                with mc1:
                    st.markdown(
                        f'<div class="pw-model">{row.get("model_a", "Model A")}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="pw-score">{row.get("score_a", "—")}/10</div>',
                        unsafe_allow_html=True,
                    )
                with mc2:
                    st.markdown(
                        f'<div class="pw-model">{row.get("model_b", "Model B")}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="pw-score">{row.get("score_b", "—")}/10</div>',
                        unsafe_allow_html=True,
                    )
                with mc3:
                    st.markdown('<div class="pw-model">Result</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="pw-score {win_cls}">{winner.upper()}</div>',
                        unsafe_allow_html=True,
                    )

                breakdown = row.get("breakdown")
                if breakdown:
                    try:
                        bd = json.loads(breakdown) if isinstance(breakdown, str) else breakdown
                        if isinstance(bd, dict):
                            bc = st.columns(len(bd))
                            for col, (crit, vals) in zip(bc, bd.items(), strict=False):
                                if isinstance(vals, list | tuple) and len(vals) == 3:
                                    w, a, b = vals
                                    col.metric(crit.capitalize(), f"A={a} B={b}", f"→ {w}")
                    except Exception:
                        pass

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    if df is None or df.empty or not METRIC_COLS:
        st.markdown(
            '<div class="eb">📭 No metrics data yet.<br>Run the pipeline to generate scoring data.</div>',
            unsafe_allow_html=True,
        )
    else:
        import pandas as pd

        # KPI averages
        st.markdown('<div class="sec">Averages</div>', unsafe_allow_html=True)
        kpi_cols = st.columns(len(METRIC_COLS))
        for col, m in zip(kpi_cols, METRIC_COLS, strict=False):
            col.markdown(
                f'<div class="sc"><div class="sv">{df[m].mean():.2f}</div>'
                f'<div class="sl">{m.replace("_", " ")}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # Filters
        f1, f2 = st.columns(2)
        models = (
            ["All"] + sorted(df["model"].unique().tolist()) if "model" in df.columns else ["All"]
        )
        cats = (
            ["All"] + sorted(df["category"].unique().tolist())
            if "category" in df.columns
            else ["All"]
        )
        sel_model = f1.selectbox("Model", models, key="m_model")
        sel_cat = f2.selectbox("Category", cats, key="m_cat")

        filtered = df.copy()
        if sel_model != "All":
            filtered = filtered[filtered["model"] == sel_model]
        if sel_cat != "All" and "category" in filtered.columns:
            filtered = filtered[filtered["category"] == sel_cat]

        st.caption(f"{len(filtered)} responses")

        mt1, mt2, mt3, mt4 = st.tabs(["Distributions", "By Model", "Correlation", "Raw Data"])

        with mt1:
            metric = st.selectbox("Metric", METRIC_COLS, key="dist_m")
            fig = px.histogram(filtered, x=metric, nbins=20, color_discrete_sequence=["#58a6ff"])
            fig.update_layout(
                plot_bgcolor="#0d1117",
                paper_bgcolor="#0d1117",
                font_color="#8b949e",
                showlegend=False,
                margin=dict(t=4, b=0, l=0, r=0),
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d", title="Count"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with mt2:
            if "model" not in filtered.columns or filtered["model"].nunique() < 2:
                st.info("Need at least 2 models to compare.")
            else:
                grouped = filtered.groupby("model")[METRIC_COLS].mean().reset_index()
                fig2 = px.bar(
                    grouped.melt(id_vars="model", value_vars=METRIC_COLS),
                    x="variable",
                    y="value",
                    color="model",
                    barmode="group",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    labels={"variable": "Metric", "value": "Avg Score", "model": "Model"},
                )
                fig2.update_layout(
                    plot_bgcolor="#0d1117",
                    paper_bgcolor="#0d1117",
                    font_color="#8b949e",
                    margin=dict(t=4, b=0, l=0, r=0),
                    xaxis=dict(gridcolor="#21262d"),
                    yaxis=dict(gridcolor="#21262d"),
                )
                st.plotly_chart(fig2, use_container_width=True)

        with mt3:
            corr = filtered[METRIC_COLS].corr().round(2)
            fig3 = go.Figure(
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
            fig3.update_layout(
                plot_bgcolor="#0d1117",
                paper_bgcolor="#0d1117",
                font_color="#8b949e",
                margin=dict(t=4, b=0, l=0, r=0),
                height=380,
            )
            st.plotly_chart(fig3, use_container_width=True)

        with mt4:
            show_cols = ["model", "prompt", "judge_score"] + [
                c for c in METRIC_COLS if c != "judge_score"
            ]
            show_cols = [c for c in show_cols if c in filtered.columns]
            st.dataframe(
                filtered[show_cols].sort_values("judge_score", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "⬇ Download CSV", filtered[show_cols].to_csv(index=False), "metrics.csv", "text/csv"
            )
