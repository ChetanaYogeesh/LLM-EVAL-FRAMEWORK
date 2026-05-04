"""pages/1_Launch.py — Run any evaluator."""

import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
st.set_page_config(page_title="Launch", page_icon="🚀", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.ph  { font-size:1.6rem; font-weight:700; color:#e6edf3; margin-bottom:4px; }
.ps  { color:#6e7681; font-size:0.88rem; margin-bottom:28px; }
.row { display:flex; align-items:center; gap:16px; background:#0d1117; border:1px solid #21262d; border-radius:10px; padding:16px 20px; margin-bottom:12px; }
.ri  { font-size:1.8rem; flex-shrink:0; }
.rt  { font-weight:700; color:#e6edf3; font-size:0.95rem; margin:0 0 2px; }
.rm  { font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#6e7681; }
.badge { display:inline-block; padding:1px 7px; border-radius:20px; font-size:0.65rem; font-family:'JetBrains Mono',monospace; font-weight:600; margin-left:6px; }
.bg { background:#0d2818; color:#3fb950; border:1px solid #238636; }
.bb { background:#0c1f3f; color:#58a6ff; border:1px solid #1f6feb; }
.ba { background:#2d1e0f; color:#d29922; border:1px solid #9e6a03; }
.bx { background:#161b22; color:#8b949e; border:1px solid #30363d; }
.err-box { background:#1a0a0a; border:1px solid #6e1a1a; border-radius:8px; padding:14px 16px; margin-top:8px; }
.err-title { color:#f85149; font-weight:700; font-size:0.95rem; margin-bottom:6px; }
.err-msg   { color:#e6edf3; font-size:0.85rem; margin-bottom:8px; }
.err-fix   { color:#3fb950; font-size:0.82rem; }
.err-detail { color:#6e7681; font-size:0.75rem; font-family:'JetBrains Mono',monospace; margin-top:8px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="ph">🚀 Launch Evaluators</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ps">Run any evaluator below. Results save automatically and appear in Results & Pipeline pages.</div>',
    unsafe_allow_html=True,
)


def _friendly_error(log: str) -> tuple[str, str, str] | None:
    """
    Parse raw output and return (title, message, fix) for known errors,
    or None if the error is not recognised.
    """
    low = log.lower()

    if "402" in log or "insufficient credits" in low or "never purchased credits" in low:
        return (
            "💳 OpenRouter — Insufficient Credits",
            "Your OpenRouter account has no credits. This evaluator requires paid OpenRouter credits.",
            "→ Add credits at https://openrouter.ai/settings/credits (even $5 goes a long way)",
        )
    if "401" in log or "invalid api key" in low or "authentication" in low:
        return (
            "🔑 Authentication Error",
            "Your OPENAI_API_KEY is invalid or not recognised by OpenRouter.",
            "→ Check your key at https://openrouter.ai/keys and update it in Streamlit Secrets",
        )
    if "404" in log or "no endpoints found" in low or "model not found" in low:
        return (
            "🔍 Model Not Found",
            "The requested model is not available on OpenRouter right now.",
            "→ Update the model name in crewai_evaluator.py or try a different free model",
        )
    if "connection" in low and "ollama" in low:
        return (
            "🔌 Ollama Not Running",
            "Could not connect to local Ollama at localhost:11434.",
            "→ Run: ollama serve  then  ollama pull llama3.2",
        )
    if "openai_api_key" in low or "api_key not set" in low or "key not found" in low:
        return (
            "🔑 Missing API Key",
            "OPENAI_API_KEY is not set in the environment.",
            "→ Add it to Streamlit Secrets: share.streamlit.io → App settings → Secrets",
        )
    if "modulenotfounderror" in low or "no module named" in low:
        import re
        m = re.search(r"no module named '([^']+)'", low)
        mod = m.group(1) if m else "unknown"
        return (
            f"📦 Missing Package: {mod}",
            f"The Python package '{mod}' is not installed.",
            f"→ Add '{mod}' to requirements.txt and push to redeploy",
        )
    if "timeout" in low:
        return (
            "⏱ Timeout",
            "The evaluator took too long and was stopped.",
            "→ Try again, or reduce max_tokens in crewai_evaluator.py",
        )
    return None


def show_result(ev: dict, result: subprocess.CompletedProcess) -> None:
    combined_out = (result.stdout or "").strip()
    combined_err = (result.stderr or "").strip()
    full_log = "\n".join(filter(None, [combined_out, combined_err])) or "No output captured."

    if result.returncode == 0:
        st.success(f"✅ {ev['title']} completed! See **Results** in the sidebar.")
        with st.expander("Output log"):
            st.code(full_log)
        return

    # Try to show a friendly error
    parsed = _friendly_error(full_log)
    if parsed:
        title, message, fix = parsed
        st.error(f"❌ {ev['title']} failed")
        st.markdown(
            f"""
            <div class="err-box">
                <div class="err-title">{title}</div>
                <div class="err-msg">{message}</div>
                <div class="err-fix">{fix}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Full log (for debugging)"):
            st.code(full_log)
    else:
        # Unknown error — show the last meaningful lines, not the full traceback
        lines = [ln for ln in full_log.splitlines() if ln.strip()]
        # Last non-traceback line is usually the most useful
        summary_lines = [ln for ln in lines if not ln.startswith("  ") and ln.strip()]
        summary = "\n".join(summary_lines[-5:]) if summary_lines else full_log[-800:]

        st.error(f"❌ {ev['title']} failed")
        st.code(summary)
        with st.expander("Full log (for debugging)"):
            st.code(full_log)


EVALUATORS = [
    {
        "key": "ollama",
        "icon": "🟢",
        "title": "Ollama Evaluator",
        "badges": [("Free", "bg"), ("Local → OpenRouter fallback", "bx")],
        "meta": "llama3.2 locally · falls back to gpt-4o-mini via OpenRouter",
        "script": "ollama_evaluator.py",
        "timeout": 90,
        "note": None,
    },
    {
        "key": "crew",
        "icon": "🔵",
        "title": "CrewAI Evaluator",
        "badges": [("Multi-agent", "bb"), ("Requires OPENAI_API_KEY", "bx")],
        "meta": "6 specialist agents · hierarchical · trace, quality, safety, cost, regression",
        "script": "crewai_evaluator.py",
        "timeout": 180,
        "note": "May take up to 3 minutes",
    },
    {
        "key": "pro",
        "icon": "🔴",
        "title": "Professional Pipeline",
        "badges": [("SQLite", "ba"), ("Multi-model · NLP scoring", "bx")],
        "meta": "Full pipeline · pairwise comparisons · LLM judge · leaderboard",
        "script": None,
        "timeout": None,
        "note": "→ Configure in the Pipeline page (sidebar)",
    },
]

for ev in EVALUATORS:
    badges_html = "".join(f'<span class="badge {c}">{t}</span>' for t, c in ev["badges"])
    col_info, col_btn = st.columns([5, 1])
    with col_info:
        st.markdown(
            f"""
        <div class="row">
            <div class="ri">{ev["icon"]}</div>
            <div>
                <div class="rt">{ev["title"]} {badges_html}</div>
                <div class="rm">{ev["meta"]}</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col_btn:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if ev["script"]:
            clicked = st.button("▶ Run", key=ev["key"], use_container_width=True, type="primary")
        else:
            clicked = False
            st.page_link(
                "pages/4_Professional_Pipeline.py",
                label="→ Open",
                use_container_width=True,
            )

    if ev["note"] and not ev["script"]:
        st.caption(ev["note"])

    if ev["script"] and clicked:
        note = f" — {ev['note']}" if ev["note"] else ""
        with st.spinner(f"Running {ev['title']}{note}..."):
            result = subprocess.run(
                [sys.executable, str(ROOT / ev["script"])],
                capture_output=True,
                text=True,
                cwd=ROOT,
                timeout=ev["timeout"],
            )
        show_result(ev, result)