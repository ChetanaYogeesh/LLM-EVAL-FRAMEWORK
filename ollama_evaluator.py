"""
ollama_evaluator.py
Improved Local-First LLM Evaluator
- Uses Ollama primarily
- Falls back gracefully to OpenRouter
- Integrates crewai_tools
- Clean user-friendly error messages
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

# Import shared tools for consistency with CrewAI evaluator
from crewai_tools import (
    MetricCalculatorTool,
    SafetyGuardTool,
    TraceParserTool,
)


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
    """Check if local Ollama server is running."""
    try:
        import httpx

        httpx.get("http://localhost:11434", timeout=2.0)
        return True
    except Exception:
        return False


def call_llm(prompt: str) -> str:
    """Call LLM with smart fallback: Ollama → OpenRouter."""
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
                raise RuntimeError("Ollama not reachable and OPENAI_API_KEY not set.")
            print("⚠️  Ollama unreachable — falling back to OpenRouter (gpt-4o-mini)")
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
            print("\n" + "=" * 70)
            print("❌ OPENROUTER CREDIT ERROR")
            print("=" * 70)
            print("Your OpenRouter account has insufficient credits.")
            print("This account has never purchased credits.")
            print("\n💡 Recommended Solutions:")
            print("   1. Run Ollama locally (recommended):")
            print("      → ollama serve")
            print("      → ollama pull llama3.2")
            print("   2. Add credits at → https://openrouter.ai/settings/credits")
            print("=" * 70)
        else:
            print(f"\n❌ LLM call failed: {e}")

        # Fixed: Use 'from None' to suppress original exception context (satisfies B904)
        raise SystemExit(1) from None


def run_evaluation(test_case: dict[str, Any] = None) -> EvaluationReport:
    if test_case is None:
        test_case = {
            "test_case_id": "TC-001",
            "expected_outcome": "Paris is the capital of France.",
            "context": "Paris is the capital of France.",
            "trace": json.dumps(
                {
                    "steps": [{"name": "research", "latency_ms": 2450}],
                    "loop_count": 0,
                    "retry_count": 1,
                }
            ),
            "actual_final_answer": "Paris is the capital of France and has approximately 2.1 million residents.",
        }

    # Strong evaluation prompt
    prompt = f"""You are a strict and accurate LLM evaluator.

Expected: {test_case['expected_outcome']}
Actual Answer: {test_case.get('actual_final_answer', '')}
Trace: {test_case.get('trace', '')}

Return ONLY valid JSON with this structure:
{{
  "pass_fail": "PASS" or "FAIL",
  "failure_mode": "none|hallucination|reasoning|safety",
  "release_decision": "approve|approve_with_caution|block",
  "recommendations": ["list of short suggestions"],
  "metrics": {{"reasoning_quality": 4.2, "step_efficiency": 4.0}}
}}
"""

    raw = call_llm(prompt)

    # Robust JSON extraction
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    json_str = match.group(0) if match else raw

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        print("⚠️ Failed to parse JSON from LLM, using safe fallback")
        data = {"pass_fail": "UNKNOWN", "failure_mode": "parse_error", "recommendations": []}

    # Use shared tools
    safety_tool = SafetyGuardTool()
    safety_result = json.loads(safety_tool._run(data.get("actual_final_answer", "")))

    trace_tool = TraceParserTool()
    trace_analysis = trace_tool._run(test_case.get("trace", "{}"))

    metric_tool = MetricCalculatorTool()
    metrics_result = metric_tool._run(
        trace_analysis=trace_analysis,
        quality_scores=data.get("metrics", {}),
        safety_result=safety_result,
        cost_latency_result={"cost_per_successful_task_usd": 0.0},
        regression_result={"flags": []},
        expected_outcome=test_case["expected_outcome"],
        actual_final_answer=test_case.get("actual_final_answer", ""),
    )

    # Build final report
    report = EvaluationReport(
        test_case_id=test_case["test_case_id"],
        pass_fail=data.get("pass_fail", "UNKNOWN"),
        failure_mode=data.get("failure_mode", "none"),
        release_decision=data.get("release_decision", "hold"),
        recommendations=data.get("recommendations", ["Improve prompt structure"]),
        metrics=json.loads(metrics_result).get("metrics", {}),
        hallucination_detected=len(test_case.get("actual_final_answer", ""))
        > len(test_case["expected_outcome"]) * 1.5,
        toxicity_detected=not safety_result.get("safe", True),
        top_bottlenecks=json.loads(trace_analysis).get("bottlenecks", []),
    )

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(report.model_dump(), f, indent=2)

    # Append to history
    history_path = Path("evaluation_history.json")
    history = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            history = []
    history.append(report.model_dump())
    history_path.write_text(json.dumps(history, indent=2))

    print(f"\n✅ Evaluation Completed → {report.pass_fail} | Mode: {report.failure_mode}")
    return report


if __name__ == "__main__":
    try:
        run_evaluation()
    except SystemExit:
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
