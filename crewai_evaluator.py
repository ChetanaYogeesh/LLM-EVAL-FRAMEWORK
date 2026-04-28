"""
crewai_evaluator.py
→ Full hierarchical multi-agent CrewAI evaluator
"""

import json
import os
import traceback
from typing import Any

import yaml
from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from pydantic import BaseModel

# Import CrewAI tools from dedicated module
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

OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY")


def get_llm_config(agent_name: str):
    """Smart model switching based on agent role."""
    base = {
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0.0,
    }
    if "coordinator" in agent_name.lower() or "quality" in agent_name.lower():
        base["model"] = "openrouter/openai/gpt-4o"
        base["max_tokens"] = 4000
    elif "safety" in agent_name.lower():
        base["model"] = "openrouter/anthropic/claude-3-haiku"
        base["max_tokens"] = 3000
    else:
        base["model"] = "openrouter/google/gemini-flash-1.5"
        base["max_tokens"] = 2000
    return base


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
    def __init__(self):
        try:
            with open("config/agents.yaml") as f:
                self.agents_config = yaml.safe_load(f)
            with open("config/tasks.yaml") as f:
                self.tasks_config = yaml.safe_load(f)
            print("✅ Config files loaded successfully.")
        except Exception as e:
            print(f"❌ Config loading failed: {e}")
            raise

    def create_manager(self):
        """Create the coordinator as manager (DO NOT add it to agents list)."""
        return Agent(
            config=self.agents_config.get("evaluator_coordinator", {}),
            verbose=True,
            llm=get_llm_config("evaluator_coordinator"),
            allow_delegation=True,
        )

    def create_worker_agents(self):
        """Create the specialist agents."""
        return [
            Agent(
                config=self.agents_config.get("trace_analyst", {}),
                verbose=True,
                llm=get_llm_config("trace_analyst"),
                tools=[TraceParserTool()],
            ),
            Agent(
                config=self.agents_config.get("quality_judge", {}),
                verbose=True,
                llm=get_llm_config("quality_judge"),
            ),
            Agent(
                config=self.agents_config.get("safety_judge", {}),
                verbose=True,
                llm=get_llm_config("safety_judge"),
                tools=[SafetyGuardTool(), HumanReviewTool()],
            ),
            Agent(
                config=self.agents_config.get("cost_latency_analyst", {}),
                verbose=True,
                llm=get_llm_config("cost_latency_analyst"),
                tools=[CostCalculatorTool()],
            ),
            Agent(
                config=self.agents_config.get("regression_monitor", {}),
                verbose=True,
                llm=get_llm_config("regression_monitor"),
                tools=[RegressionComparatorTool()],
            ),
        ]

    def crew(self) -> Crew:
        manager = self.create_manager()
        workers = self.create_worker_agents()

        # Main coordination task assigned to manager
        main_task = Task(
            config=self.tasks_config.get("coordinate_evaluation", {}),
            agent=manager,
            tools=[MetricCalculatorTool()],  # Final aggregation tool
        )

        return Crew(
            agents=workers,  # Only workers here
            tasks=[main_task],
            process=Process.hierarchical,
            manager_agent=manager,
            verbose=True,
            memory=False,
            output_json=True,
            max_iter=10,
        )


if __name__ == "__main__":
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

        # Save result
        output = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        with open("evaluation_results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)

        print("✅ Evaluation completed successfully!")
        print("📄 Results saved to evaluation_results.json")
        print(f"Pass/Fail: {output.get('pass_fail', 'N/A')}")

    except Exception as e:
        print(f"❌ Error during crew execution: {e}")
        traceback.print_exc()
