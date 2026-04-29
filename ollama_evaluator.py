"""
ollama_evaluator.py
→ Direct Ollama evaluator (lightweight, reliable, no CrewAI dependency)
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

try:
    import litellm

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


class EvaluationReport(BaseModel):
    test_case_id: str
    pass_fail: str
    metrics: dict[str, Any]
    failure_mode: str
    recommendations: list[str]
    release_decision: str
    top_bottlenecks: list[str]
    top_regressions: list[str]
    hallucination_detected: bool = False
    bias_detected: bool = False
    toxicity_detected: bool = False
    timestamp: str = ""


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


def call_ollama(prompt: str) -> str:
    if not LITELLM_AVAILABLE:
        raise RuntimeError("litellm is not installed. Run: pip install litellm")
    response = litellm.completion(
        model="ollama/llama3.2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2000,
    )
    return response.choices[0].message.content


def run_evaluation() -> dict:
    test_case = {
        "test_case_id": "TC-001",
        "expected_outcome": "Paris is the capital of France.",
        "trace": "Agent researched and returned correct answer.",
        "context": "Paris is the capital of France.",
    }

    prompt = f"""Evaluate this agent execution:

Expected: {test_case['expected_outcome']}
Trace: {test_case['trace']}

Return ONLY valid JSON matching the EvaluationReport schema."""

    raw = call_ollama(prompt)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    json_str = match.group(0) if match else raw
    report_dict = json.loads(json_str)

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

    print("✅ Ollama evaluation completed!")
    print(f"Pass/Fail: {report_dict.get('pass_fail')}")
    return report_dict


if __name__ == "__main__":
    if not LITELLM_AVAILABLE:
        print("❌ litellm not installed. Run: pip install litellm")
        raise SystemExit(1)
    run_evaluation()
