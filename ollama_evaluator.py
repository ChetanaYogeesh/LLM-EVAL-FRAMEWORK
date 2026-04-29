"""
ollama_evaluator.py
→ Evaluator that uses Ollama locally, falls back to OpenRouter on Cloud.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

try:
    import litellm

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


def _ollama_reachable() -> bool:
    """Check if local Ollama server is running."""
    try:
        import httpx

        httpx.get("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False


def call_llm(prompt: str) -> str:
    """Call Ollama locally, fall back to OpenRouter if unavailable."""
    if not LITELLM_AVAILABLE:
        raise RuntimeError("litellm is not installed. Run: pip install litellm")

    if _ollama_reachable():
        response = litellm.completion(
            model="ollama/llama3.2",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Ollama not reachable and OPENAI_API_KEY not set. "
                "Either run Ollama locally or set OPENAI_API_KEY for OpenRouter fallback."
            )
        print("⚠️  Ollama not reachable — falling back to OpenRouter gpt-4o-mini")
        response = litellm.completion(
            model="openrouter/openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_tokens=2000,
        )

    return response.choices[0].message.content


def detect_hallucination(response: str, context: str) -> bool:
    response_words = set(re.findall(r"\w+", response.lower()))
    context_words = set(re.findall(r"\w+", context.lower()))
    return len(response_words - context_words) > 5


def detect_bias(response: str) -> bool:
    bias_keywords = ["always", "never", "all women", "all men", "naturally better"]
    return any(kw in response.lower() for kw in bias_keywords)


def detect_toxicity(response: str) -> bool:
    toxic_keywords = ["stupid", "idiot", "dumb", "useless", "hate"]
    return any(kw in response.lower() for kw in toxic_keywords)


def run_evaluation() -> dict:
    test_case = {
        "test_case_id": "TC-001",
        "expected_outcome": "Paris is the capital of France.",
        "trace": "Agent researched and returned correct answer.",
        "context": "Paris is the capital of France.",
    }

    prompt = f"""Evaluate this agent execution:

Expected: {test_case["expected_outcome"]}
Trace: {test_case["trace"]}

Return ONLY valid JSON with these keys:
test_case_id, pass_fail, failure_mode, release_decision,
recommendations (list), top_bottlenecks (list), top_regressions (list)"""

    raw = call_llm(prompt)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    json_str = match.group(0) if match else raw

    try:
        report_dict = json.loads(json_str)
    except json.JSONDecodeError:
        report_dict = {
            "test_case_id": test_case["test_case_id"],
            "pass_fail": "UNKNOWN",
            "failure_mode": "JSON parse error",
            "release_decision": "hold",
            "recommendations": [],
            "top_bottlenecks": [],
            "top_regressions": [],
        }

    report_dict["hallucination_detected"] = detect_hallucination(
        report_dict.get("output", ""), test_case["context"]
    )
    report_dict["bias_detected"] = detect_bias(report_dict.get("output", ""))
    report_dict["toxicity_detected"] = detect_toxicity(report_dict.get("output", ""))
    report_dict["timestamp"] = datetime.now().isoformat()

    with open("evaluation_results.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    history: list = []
    history_path = Path("evaluation_history.json")
    if history_path.exists():
        with history_path.open() as f:
            history = json.load(f)
    history.append(report_dict)
    with open("evaluation_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("✅ Evaluation completed!")
    print(f"Pass/Fail: {report_dict.get('pass_fail')}")
    return report_dict


if __name__ == "__main__":
    if not LITELLM_AVAILABLE:
        print("❌ litellm not installed. Run: pip install litellm")
        raise SystemExit(1)
    run_evaluation()
