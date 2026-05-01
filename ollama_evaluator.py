"""
ollama_evaluator.py - Production-ready Local Evaluator
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm
from pydantic import BaseModel, Field

# Shared tools from crewai_tools
from crewai_tools import SafetyGuardTool, TraceParserTool


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
    """Call LLM with clear user messages."""
    try:
        if _ollama_reachable():
            print("🟢 Using local Ollama (llama3.2)")
            response = litellm.completion(
                model="ollama/llama3.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2800,
            )
        else:
            print("⚠️  Ollama not running → falling back to OpenRouter")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("❌ ERROR: Ollama is not running and OPENAI_API_KEY is not set.")
                print("   Start Ollama or add your key.")
                sys.exit(1)

            response = litellm.completion(
                model="openrouter/openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.0,
                max_tokens=2800,
            )
        return response.choices[0].message.content

    except Exception as e:
        err = str(e).lower()
        if "402" in err or "credits" in err:
            print(
                "\n❌ OpenRouter Credit Error - Add credits at https://openrouter.ai/settings/credits"
            )
        else:
            print(f"\n❌ LLM Error: {e}")
        sys.exit(1)


def run_evaluation():
    print("🚀 Starting Ollama Evaluator...\n")

    test_case = {
        "test_case_id": "TC-001",
        "expected_outcome": "Paris is the capital of France.",
        "context": "Paris is the capital of France.",
        "actual_final_answer": "Paris is the capital of France.",
    }

    prompt = """Evaluate this agent execution and return ONLY valid JSON."""

    raw = call_llm(prompt)

    # Extract JSON
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    json_str = match.group(0) if match else raw

    try:
        data = json.loads(json_str)
    except Exception:
        data = {"pass_fail": "UNKNOWN", "failure_mode": "parse_error"}

    # Use tools
    safety_result = json.loads(SafetyGuardTool()._run(data.get("actual_final_answer", "")))
    TraceParserTool()._run('{"steps": [{"name": "research", "latency_ms": 1200}]}')

    report = EvaluationReport(
        test_case_id=test_case["test_case_id"],
        pass_fail=data.get("pass_fail", "PASS"),
        failure_mode=data.get("failure_mode", "none"),
        release_decision=data.get("release_decision", "approve"),
        toxicity_detected=not safety_result.get("safe", True),
        timestamp=datetime.now().isoformat(),
    )

    # Save
    Path("evaluation_results.json").write_text(json.dumps(report.model_dump(), indent=2))

    print(f"\n✅ Evaluation Completed → {report.pass_fail}")
    print(f"   Failure Mode : {report.failure_mode}")
    print(f"   Toxicity     : {report.toxicity_detected}")


if __name__ == "__main__":
    try:
        run_evaluation()
    except KeyboardInterrupt:
        print("\n\n⛔ Stopped by user.")
    except SystemExit:
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
