"""
ollama_evaluator.py - Minimal Debug Version
"""

import sys

print("🚀 ollama_evaluator.py started", file=sys.stderr)

try:
    import json
    from datetime import datetime

    print("✅ Basic imports successful", file=sys.stderr)

    # Simple test
    report = {
        "test_case_id": "TC-001",
        "pass_fail": "PASS",
        "failure_mode": "none",
        "release_decision": "approve",
        "timestamp": datetime.now().isoformat(),
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n✅ Evaluation Completed Successfully!")
    print(f"Pass/Fail: {report['pass_fail']}")
    print("Results saved to evaluation_results.json")

except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}", file=sys.stderr)
    import traceback

    traceback.print_exc()
    sys.exit(1)
