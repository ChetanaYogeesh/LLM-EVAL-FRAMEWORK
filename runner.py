"""
runner.py

Main evaluation runner.

Orchestrates the full pipeline:
  1. Load prompts from dataset
  2. Run model responses (real or mock)
  3. Compute NLP metrics
  4. Score with LLM-as-a-Judge
  5. Generate pairwise comparisons
  6. Store everything in SQLite
  7. Print leaderboard

Usage:
  # Demo mode (no API keys needed):
  python runner.py

  # With real models:
  OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-... python runner.py --real

  # Custom dataset:
  python runner.py --dataset datasets/my_prompts.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from comparator import format_report, generate_comparison_report
from llm_judge import judge_response
from rankings import print_leaderboard
from runners import evaluate_prompts, mock_model_a, mock_model_b, run_claude, run_openai
from scorer import compute_all_metrics
from sqlite_store import (
    DB_PATH,
    create_experiment,
    init_db,
    insert_metrics,
    insert_pairwise,
    insert_prompt,
    insert_response,
    upsert_model,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def load_prompts(path: str) -> list:
    with open(path) as f:
        return json.load(f)


# ── Main Pipeline ─────────────────────────────────────────────────────────────


async def run_evaluation(
    dataset_path: str = "sample_prompts.json",
    use_real_models: bool = False,
    judge_mode: str = "auto",
    experiment_name: str = "default_run",
    verbose: bool = True,
):
    print("\n" + "=" * 70)
    print("  LLM EVALUATION FRAMEWORK")
    print("=" * 70)

    # 1. Init DB
    init_db()
    exp_id = create_experiment(
        name=experiment_name,
        description="Automated eval run",
        config={
            "dataset": dataset_path,
            "judge": judge_mode,
            "real_models": use_real_models,
        },
    )
    print(f"[Runner] Experiment #{exp_id}: {experiment_name}")

    # 2. Load prompts
    prompts = load_prompts(dataset_path)
    print(f"[Runner] Loaded {len(prompts)} prompts from {dataset_path}")

    # 3. Choose runners
    if use_real_models:
        runners = [run_openai, run_claude]
        runner_names = ["GPT-4.1", "Claude-Sonnet"]
    else:
        runners = [mock_model_a, mock_model_b]
        runner_names = ["MockModel-A", "MockModel-B"]

    print(f"[Runner] Models: {', '.join(runner_names)}")

    # 4. Ensure models exist in DB
    model_ids = {name: upsert_model(name) for name in runner_names}

    # 5. Run all prompts
    print(f"[Runner] Running {len(prompts)} prompts × {len(runners)} models...")
    results = await evaluate_prompts(prompts, runners, runner_names)

    # 6. Score and store
    pairwise_reports = []

    for item in results:
        prompt_text = item["prompt"]
        reference = item["reference"]
        category = item["category"]

        prompt_id = insert_prompt(prompt_text, reference, category)
        response_texts = {}

        for model_name in runner_names:
            response_text = item["responses"].get(model_name, "")
            response_texts[model_name] = response_text

            resp_id = insert_response(model_ids[model_name], prompt_id, response_text)

            # NLP metrics
            nlp_scores = compute_all_metrics(response_text, reference)

            # LLM judge
            judge_scores = judge_response(
                prompt_text, response_text, reference, judge=judge_mode
            )

            combined_scores = {
                **nlp_scores,
                "judge_score": judge_scores.get("overall", 5),
                "clarity": judge_scores.get("clarity", 5),
                "completeness": judge_scores.get("completeness", 5),
                "conciseness": 5,
                "tone": 5,
            }
            insert_metrics(resp_id, combined_scores)

        # Pairwise comparison (first two models only)
        if len(runner_names) >= 2:
            model_a, model_b = runner_names[0], runner_names[1]
            report = generate_comparison_report(
                prompt_text,
                response_texts[model_a],
                response_texts[model_b],
                reference,
            )
            insert_pairwise(
                prompt_id,
                model_ids[model_a],
                model_ids[model_b],
                report["winner"],
                report["score_a"],
                report["score_b"],
                {k: list(v) for k, v in report["breakdown"].items()},
            )
            pairwise_reports.append(report)

        if verbose:
            print(f"  ✓ [{category}] {prompt_text[:60]}...")

    # 7. Print sample pairwise report
    if pairwise_reports:
        print("\n" + format_report(pairwise_reports[0]))

    # 8. Leaderboard
    print_leaderboard()
    print(f"\n[Runner] Done. Results stored in {DB_PATH}\n")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Evaluation Runner")
    parser.add_argument(
        "--real", action="store_true", help="Use real API models instead of mocks"
    )
    parser.add_argument(
        "--dataset", default="sample_prompts.json", help="Path to prompt dataset JSON"
    )
    parser.add_argument(
        "--judge",
        default="auto",
        choices=["auto", "openai", "claude", "heuristic"],
        help="Judge type",
    )
    parser.add_argument("--name", default="eval_run", help="Experiment name")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-prompt output"
    )
    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            dataset_path=args.dataset,
            use_real_models=args.real,
            judge_mode=args.judge,
            experiment_name=args.name,
            verbose=not args.quiet,
        )
    )
