"""
ollama_evaluator.py
Clean Local Evaluator with helpful messages.
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    import litellm

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


def _ollama_reachable() -> bool:
    try:
        import httpx

        httpx.get("http://localhost:11434", timeout=2.0)
        return True
    except Exception:
        return False


def _exit_with_error(msg: str) -> None:
    """Print to both stdout and stderr so the caller always sees the message."""
    print(msg, flush=True)
    print(msg, file=sys.stderr, flush=True)
    sys.exit(1)


def call_llm(prompt: str) -> str:
    if not LITELLM_AVAILABLE:
        _exit_with_error("❌ litellm is not installed.\n   Run: pip install litellm")

    if _ollama_reachable():
        print("🟢 Using local Ollama (llama3.2)", flush=True)
        try:
            response = litellm.completion(
                model="ollama/llama3.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2500,
            )
            return response.choices[0].message.content
        except Exception as e:
            _exit_with_error(
                f"❌ Ollama call failed: {e}\n"
                "   Make sure Ollama is running and llama3.2 is pulled:\n"
                "     ollama serve\n"
                "     ollama pull llama3.2"
            )
    else:
        print("⚠️  Ollama not detected — trying OpenRouter fallback...", flush=True)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            _exit_with_error(
                "❌ Ollama is not running and OPENAI_API_KEY is not set.\n"
                "\n"
                "   Option 1 — Run Ollama locally:\n"
                "     ollama serve\n"
                "     ollama pull llama3.2\n"
                "\n"
                "   Option 2 — Set your OpenRouter key:\n"
                "     export OPENAI_API_KEY=sk-or-v1-...\n"
                "   Or add it to Streamlit secrets (share.streamlit.io → App settings → Secrets)"
            )

        print("⚠️  Falling back to OpenRouter gpt-4o-mini", flush=True)
        try:
            response = litellm.completion(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.0,
                max_tokens=2500,
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e).lower()
            if "402" in err or "credits" in err or "insufficient" in err:
                _exit_with_error(
                    "❌ OpenRouter credit error — your account has no credits.\n"
                    "   Add credits at: https://openrouter.ai/settings/credits\n"
                    f"   Detail: {e}"
                )
            elif "401" in err or "auth" in err or "key" in err:
                _exit_with_error(
                    f"❌ OpenRouter authentication error — check your API key.\n   Detail: {e}"
                )
            elif "404" in err or "not found" in err:
                _exit_with_error(
                    "❌ OpenRouter model not found.\n"
                    "   The model 'openrouter/openai/gpt-4o-mini' may be unavailable.\n"
                    f"   Detail: {e}"
                )
            else:
                _exit_with_error(f"❌ OpenRouter call failed: {e}")


def run_evaluation() -> dict:
    print("🚀 Starting Ollama Evaluator...\n", flush=True)

    test_case = {
        "test_case_id": "TC-001",
        "expected_outcome": "Paris is the capital of France.",
        "context": "Paris is the capital of France.",
    }

    prompt = f"""Evaluate this agent execution:

Expected: {test_case["expected_outcome"]}
Trace: Agent researched and returned correct answer.

Return ONLY valid JSON with these exact keys:
{{
  "test_case_id": "{test_case["test_case_id"]}",
  "pass_fail": "PASS or FAIL",
  "failure_mode": "none or description",
  "release_decision": "approve or hold",
  "recommendations": [],
  "top_bottlenecks": [],
  "top_regressions": []
}}"""

    raw = call_llm(prompt)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    json_str = match.group(0) if match else raw

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        print("⚠️  Could not parse JSON from LLM — using defaults", flush=True)
        data = {
            "test_case_id": test_case["test_case_id"],
            "pass_fail": "UNKNOWN",
            "failure_mode": "json_parse_error",
            "release_decision": "hold",
            "recommendations": [],
            "top_bottlenecks": [],
            "top_regressions": [],
        }

    data.setdefault("test_case_id", test_case["test_case_id"])
    data.setdefault("hallucination_detected", False)
    data.setdefault("bias_detected", False)
    data.setdefault("toxicity_detected", False)
    data["timestamp"] = datetime.now().isoformat()

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(data, f, indent=2)

    history: list = []
    history_path = Path("evaluation_history.json")
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            history = []
    history.append(data)
    history_path.write_text(json.dumps(history, indent=2))

    print("\n✅ Evaluation completed!", flush=True)
    print(f"   Pass/Fail: {data.get('pass_fail', 'UNKNOWN')}", flush=True)
    print(f"   Release:   {data.get('release_decision', '—')}", flush=True)
    print("   Results saved to evaluation_results.json", flush=True)
    return data


if __name__ == "__main__":
    try:
        run_evaluation()
    except KeyboardInterrupt:
        print("\n⛔ Cancelled.", file=sys.stderr)
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        msg = f"❌ Unexpected error: {e}"
        print(msg, flush=True)
        print(msg, file=sys.stderr, flush=True)
        sys.exit(1)
