"""
ollama_evaluator.py
Clean Local-First LLM Evaluator with friendly error messages.
"""

import json
import os
import re
import sys
from datetime import datetime
from typing import Any

import litellm
from pydantic import BaseModel, Field

# Import shared tools


class EvaluationReport(BaseModel):
    test_case_id: str
    pass_fail: str = "UNKNOWN"
    failure_mode: str = "none"
    release_decision: str = "hold"
    metrics: dict[str, Any] = Field(default_factory=dict)
    hallucination_detected: bool = False
    bias_detected: bool = False
    toxicity_detected: bool = False
    recommendations: list[str] = Field(default_factory=list)
    top_bottlenecks: list[str] = Field(default_factory=list)
    top_regressions: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


def _ollama_reachable() -> bool:
    try:
        import httpx

        httpx.get("http://localhost:11434", timeout=2.0)
        return True
    except Exception:
        return False


def call_llm(prompt: str) -> str:
    """Call LLM with clean error handling."""
    try:
        if _ollama_reachable():
            print("🟢 Using local Ollama (llama3.2)")
            response = litellm.completion(
                model="ollama/llama3.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=3000,
            )
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("\n❌ Error: Ollama is not running and no OPENAI_API_KEY found.")
                print("   → Start Ollama or set your OpenRouter key.")
                raise SystemExit(1)

            print("⚠️  Ollama not reachable — using OpenRouter (gpt-4o-mini)")
            response = litellm.completion(
                model="openrouter/openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.0,
                max_tokens=3000,
            )
        return response.choices[0].message.content

    except Exception as e:
        error_str = str(e).lower()
        if (
            "402" in error_str
            or "insufficient credits" in error_str
            or "payment required" in error_str
        ):
            print("\n" + "=" * 65)
            print("❌ OPENROUTER CREDIT ERROR")
            print("=" * 65)
            print("You don't have enough credits on your OpenRouter account.")
            print("This account has never purchased credits.")
            print("\n✅ Recommended:")
            print("   1. Run Ollama locally (best option)")
            print("      ollama serve")
            print("      ollama pull llama3.2")
            print("   2. Add credits → https://openrouter.ai/settings/credits")
            print("=" * 65)
        else:
            print(f"\n❌ Unexpected error: {e}")

        raise SystemExit(1) from None


def run_evaluation() -> EvaluationReport:
    # ... (keeping the rest of your logic clean) ...
    test_case = {
        "test_case_id": "TC-001",
        "expected_outcome": "Paris is the capital of France.",
        "context": "Paris is the capital of France.",
        "trace": json.dumps({"steps": [{"name": "research", "latency_ms": 2450}]}),
        "actual_final_answer": "Paris is the capital of France.",
    }

    prompt = f"""Evaluate this execution and return ONLY valid JSON.

Expected: {test_case['expected_outcome']}
Answer: {test_case['actual_final_answer']}

Return JSON only."""

    raw = call_llm(prompt)

    # Clean JSON extraction
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    json_str = match.group(0) if match else raw

    try:
        data = json.loads(json_str)
    except Exception:
        data = {"pass_fail": "UNKNOWN", "failure_mode": "parse_error"}

    report = EvaluationReport(**data)
    report.timestamp = datetime.now().isoformat()

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(report.model_dump(), f, indent=2)

    print(f"\n✅ Evaluation Completed → {report.pass_fail}")
    return report


if __name__ == "__main__":
    try:
        run_evaluation()
    except SystemExit:
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
