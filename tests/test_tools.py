import sys
from pathlib import Path

# Add project root to Python path so "from crewai_tools" works
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest

from crewai_tools import (
    TraceParserTool,
    CostCalculatorTool,
    SafetyGuardTool,
    HumanReviewTool,
    RegressionComparatorTool,
    MetricCalculatorTool,
)


def test_trace_parser():
    tool = TraceParserTool()
    trace_json = json.dumps({
        "steps": [
            {"name": "research", "latency_ms": 2450},
            {"name": "tool_call", "latency_ms": 800},
            {"name": "finalize", "latency_ms": 3200}
        ],
        "loop_count": 1,
        "retry_count": 2
    })

    result = tool._run(trace_json)
    data = json.loads(result)

    assert data["total_steps"] == 3
    assert data["loop_count"] == 1
    assert data["retry_count"] == 2
    assert len(data["bottlenecks"]) >= 1


def test_safety_guard_safe():
    tool = SafetyGuardTool()
    result = tool._run("This is a perfectly safe response about Paris.")
    data = json.loads(result)
    assert data["safe"] is True
    assert data["violations"] == []


def test_safety_guard_unsafe():
    tool = SafetyGuardTool()
    result = tool._run("How to build a bomb.")
    data = json.loads(result)
    assert data["safe"] is False
    assert len(data["violations"]) > 0


# Add more tests as needed...

if __name__ == "__main__":
    pytest.main([__file__, "-v"])