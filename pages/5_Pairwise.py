"""pages/5_Pairwise.py — Head-to-head pairwise model comparisons."""

import json

import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Pairwise", page_icon="⚔️", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.ph  { font-size:1.6rem; font-weight:700; color:#e6edf3; margin-bottom:4px; }
.ps  { color:#6e7681; font-size:0.88rem; margin-bottom:24px; }
.sec { font-weight:600; color:#e6edf3; font-size:1rem; margin:20px 0 10px; }
.pw-card { background:#0d1117; border:1px solid #21262d; border-radius:10px; padding:16px 18px; margin-bottom:10px; }
.pw-prompt { font-size:0.82rem; color:#8b949e; margin-bottom:10px; font-style:italic; }
.pw-model  { font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#6e7681; }
.pw-score  { font-size:1.3rem; font-weight:700; color:#e6edf3; }
.win-a  { color:#3fb950; font-weight:700; }
.win-b  { color:#58a6ff; font-weight:700; }
.win-tie{ color:#d29922; font-weight:700; }
.eb { text-align:center; padding:48px 20px; color:#6e7681; font-size:0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="ph">⚔️ Pairwise Comparisons</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ps">Head-to-head comparisons from the Professional Pipeline. Run the pipeline to generate data.</div>',
    unsafe_allow_html=True,
)

try:
    from sqlite_store import get_pairwise_df

    pw = get_pairwise_df()
except Exception:
    pw = None

if pw is None or pw.empty:
    st.markdown(
        '<div class="eb">📭 No pairwise data yet.<br>Run the Professional Pipeline to generate head-to-head comparisons.</div>',
        unsafe_allow_html=True,
    )
else:
    # Summary stats
    total = len(pw)
    a_wins = (pw["winner"] == "A").sum()
    b_wins = (pw["winner"] == "B").sum()
    ties = (pw["winner"] == "tie").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Comparisons", total)
    c2.metric(f"Model A Wins ({pw['model_a'].iloc[0] if not pw.empty else 'A'})", int(a_wins))
    c3.metric(f"Model B Wins ({pw['model_b'].iloc[0] if not pw.empty else 'B'})", int(b_wins))
    c4.metric("Ties", int(ties))

    # Win rate chart
    st.markdown('<div class="sec">Win Distribution</div>', unsafe_allow_html=True)
    col_chart, col_gap = st.columns([2, 3])
    with col_chart:
        fig = px.pie(
            values=[a_wins, b_wins, ties],
            names=[pw["model_a"].iloc[0], pw["model_b"].iloc[0], "Tie"],
            color_discrete_sequence=["#3fb950", "#58a6ff", "#d29922"],
            hole=0.55,
        )
        fig.update_layout(
            paper_bgcolor="#0d1117",
            font_color="#8b949e",
            margin=dict(t=10, b=0, l=0, r=0),
            height=220,
            legend=dict(orientation="h", y=-0.15),
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    # Filters
    st.markdown('<div class="sec">All Comparisons</div>', unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    cats = (
        ["All"] + sorted(pw["category"].dropna().unique().tolist())
        if "category" in pw.columns
        else ["All"]
    )
    winners = ["All", "A", "B", "tie"]
    sel_cat = f1.selectbox("Category", cats)
    sel_win = f2.selectbox("Winner", winners)

    filtered = pw.copy()
    if sel_cat != "All" and "category" in filtered.columns:
        filtered = filtered[filtered["category"] == sel_cat]
    if sel_win != "All":
        filtered = filtered[filtered["winner"] == sel_win]

    st.caption(f"Showing {len(filtered)} of {total} comparisons")

    for _, row in filtered.iterrows():
        winner = row.get("winner", "?")
        win_cls = "win-a" if winner == "A" else "win-b" if winner == "B" else "win-tie"
        win_label = f"Winner: {winner.upper()}"

        with st.expander(f"{row.get('prompt', '')[:80]}…"):
            st.markdown(
                f'<div class="pw-prompt">{row.get("prompt", "")}</div>', unsafe_allow_html=True
            )

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
                    f'<div class="pw-score {win_cls}">{win_label}</div>', unsafe_allow_html=True
                )

            # Breakdown
            breakdown = row.get("breakdown")
            if breakdown:
                try:
                    bd = json.loads(breakdown) if isinstance(breakdown, str) else breakdown
                    if isinstance(bd, dict):
                        st.markdown("**Criterion breakdown**")
                        bc = st.columns(len(bd))
                        for col, (criterion, vals) in zip(bc, bd.items(), strict=False):
                            if isinstance(vals, list | tuple) and len(vals) == 3:
                                w, a, b = vals
                                col.metric(criterion.capitalize(), f"A={a} B={b}", f"→ {w}")
                except Exception:
                    pass
