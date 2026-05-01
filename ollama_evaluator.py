"""
ollama_evaluator.py
Clean Local Evaluator with helpful messages.
"""

import json
import os
import re
import sys
from datetime import datetime
from typing import Any

import litellm
from pydantic import BaseModel, Field

# Shared tools


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
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


def _ollama_reachable() -> bool:
    try:
        import httpx

        httpx.get("http://localhost:11434", timeout=2.0)
        return True
    except Exception:
        return False


def call_llm(prompt: str) -> str:
    """Call LLM with clear messaging."""
    try:
        if _ollama_reachable():
            print("🟢 Using local Ollama (llama3.2)")
            response = litellm.completion(
                model="ollama/llama3.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2500,
            )
        else:
            print("⚠️  Ollama not detected. Trying OpenRouter fallback...")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("❌ ERROR: Ollama is not running and OPENAI_API_KEY is not set.")
                print("   → Start Ollama and pull llama3.2, or set your OpenRouter key.")
                sys.exit(1)

            response = litellm.completion(
                model="openrouter/openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.0,
                max_tokens=2500,
            )
        return response.choices[0].message.content

    except Exception as e:
        err_str = str(e).lower()
        if "402" in err_str or "credits" in err_str or "payment" in err_str:
            print("\n❌ OpenRouter Credit Error")
            print("You need to add credits at https://openrouter.ai/settings/credits")
        elif "connection" in err_str or "ollama" in err_str:
            print("\n❌ Ollama Connection Error")
            print("Make sure Ollama is running:")
            print("   ollama serve")
            print("   ollama pull llama3.2")
        else:
            print(f"\n❌ Error: {e}")

        sys.exit(1)


def run_evaluation():
    print("🚀 Starting Ollama Evaluator...\n")

    prompt = "Evaluate this execution and return only valid JSON with pass_fail, failure_mode, etc."

    raw = call_llm(prompt)

    # Extract JSON
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    json_str = match.group(0) if match else raw

    try:
        data = json.loads(json_str)
    except Exception:
        data = {"pass_fail": "UNKNOWN", "failure_mode": "parse_error"}

    report = EvaluationReport(**data)
    report.timestamp = datetime.now().isoformat()

    # Save
    with open("evaluation_results.json", "w") as f:
        json.dump(report.model_dump(), f, indent=2)

    print(f"\n✅ Done! Result: {report.pass_fail} | Mode: {report.failure_mode}")


if __name__ == "__main__":
    try:
        run_evaluation()
    except KeyboardInterrupt:
        print("\n\n⛔ Evaluation cancelled by user.")
        sys.exit(0)
    except SystemExit:
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
