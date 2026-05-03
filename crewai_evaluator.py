"""
crewai_evaluator.py
→ Full hierarchical multi-agent CrewAI evaluator
"""

import json
import os
import traceback
from typing import Any

import yaml
from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv
from pydantic import BaseModel

from crewai_tools import (
    CostCalculatorTool,
    HumanReviewTool,
    MetricCalculatorTool,
    RegressionComparatorTool,
    SafetyGuardTool,
    TraceParserTool,
)

load_dotenv()

print("🚀 Starting CrewAI Multi-Agent Evaluator...")


def get_llm(agent_name: str) -> LLM:
    """Return a crewai.LLM object with smart model switching based on agent role.

    API key is read at call time so module import is always safe even when
    OPENAI_API_KEY is not set (e.g. during pytest collection of unit tests).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

    if "coordinator" in agent_name.lower() or "quality" in agent_name.lower():
        pass
    elif "safety" in agent_name.lower():
        pass
    else:
        pass

    # Change get_llm() to use Anthropic directly if you have an ANTHROPIC_API_KEY
    # OR use a genuinely free OpenRouter model

    # In get_llm(), replace ALL model names with this one line:
    return LLM(
        model="openrouter/free",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=800,
    )


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


class AgentEvaluatorCrew:
    def __init__(self) -> None:
        try:
            with open("config/agents.yaml") as f:
                self.agents_config = yaml.safe_load(f)
            with open("config/tasks.yaml") as f:
                self.tasks_config = yaml.safe_load(f)
            print("✅ Config files loaded successfully.")
        except Exception as e:
            print(f"❌ Config loading failed: {e}")
            raise

    def create_manager(self) -> Agent:
        """Create the coordinator as manager (must NOT be in agents list)."""
        return Agent(
            config=self.agents_config.get("evaluator_coordinator", {}),
            verbose=True,
            llm=get_llm("evaluator_coordinator"),
            allow_delegation=True,
        )

    def create_worker_agents(self) -> list[Agent]:
        """Create the five specialist worker agents."""
        return [
            Agent(
                config=self.agents_config.get("trace_analyst", {}),
                verbose=True,
                llm=get_llm("trace_analyst"),
                tools=[TraceParserTool()],
            ),
            Agent(
                config=self.agents_config.get("quality_judge", {}),
                verbose=True,
                llm=get_llm("quality_judge"),
            ),
            Agent(
                config=self.agents_config.get("safety_judge", {}),
                verbose=True,
                llm=get_llm("safety_judge"),
                tools=[SafetyGuardTool(), HumanReviewTool()],
            ),
            Agent(
                config=self.agents_config.get("cost_latency_analyst", {}),
                verbose=True,
                llm=get_llm("cost_latency_analyst"),
                tools=[CostCalculatorTool()],
            ),
            Agent(
                config=self.agents_config.get("regression_monitor", {}),
                verbose=True,
                llm=get_llm("regression_monitor"),
                tools=[RegressionComparatorTool()],
            ),
        ]

    def crew(self) -> Crew:
        manager = self.create_manager()
        workers = self.create_worker_agents()

        main_task = Task(
            config=self.tasks_config.get("coordinate_evaluation", {}),
            agent=manager,
            tools=[MetricCalculatorTool()],
        )

        return Crew(
            agents=workers,
            tasks=[main_task],
            process=Process.hierarchical,
            manager_agent=manager,
            verbose=True,
            memory=False,
            output_json=True,
            max_iter=10,
        )


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found in .env!")
        raise SystemExit(1)

    try:
        evaluator = AgentEvaluatorCrew()
        crew = evaluator.crew()

        test_inputs = {
            "test_case_id": "TC-001",
            "trace_json_here": json.dumps(
                {
                    "steps": [{"name": "research", "latency_ms": 2450}],
                    "loop_count": 0,
                    "retry_count": 1,
                }
            ),
            "expected_outcome": "Paris is the capital of France.",
            "baseline": {"p95_latency_ms": 3000, "safety_violation_rate": 0},
        }

        print("🚀 Kicking off hierarchical evaluation...")
        result = crew.kickoff(inputs=test_inputs)

        output = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        with open("evaluation_results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)

        print("✅ Evaluation completed successfully!")
        print("📄 Results saved to evaluation_results.json")
        if isinstance(output, dict):
            pass_fail = output.get("pass_fail") or output.get("EvaluationReport", {}).get(
                "pass_fail", "N/A"
            )
        elif hasattr(output, "pass_fail"):
            pass_fail = output.pass_fail
        else:
            pass_fail = "N/A"
        print(f"Pass/Fail: {pass_fail}")

    except Exception as e:
        print(f"❌ Error during crew execution: {e}")
        traceback.print_exc()
