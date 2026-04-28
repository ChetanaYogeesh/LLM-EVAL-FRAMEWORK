"""
runner.py - Main evaluation runner
"""

import argparse
import asyncio
import json

from llm_judge import judge_response
from rankings import print_leaderboard
from runners import evaluate_prompts, mock_model_a, mock_model_b, run_claude, run_openai
from scorer import compute_all_metrics
from sqlite_store import (
    DB_PATH,
    create_experiment,
    init_db,
    insert_metrics,
    insert_prompt,
    insert_response,
    upsert_model,
)


async def run_evaluation(
    dataset_path: str = "sample_prompts.json",
    use_real_models: bool = False,
    judge_mode: str = "auto",
    experiment_name: str = "default_run",
):
    init_db()
    create_experiment(experiment_name, config={"real": use_real_models, "judge": judge_mode})

    with open(dataset_path) as f:
        prompts = json.load(f)

    runners = [run_openai, run_claude] if use_real_models else [mock_model_a, mock_model_b]
    runner_names = ["GPT-4.1", "Claude-Sonnet"] if use_real_models else ["Mock-A", "Mock-B"]

    model_ids = {name: upsert_model(name) for name in runner_names}

    results = await evaluate_prompts(prompts, runners, runner_names)

    for item in results:
        prompt_id = insert_prompt(
            item["prompt"], item.get("reference", ""), item.get("category", "general")
        )
        for name in runner_names:
            resp = item["responses"].get(name, "")
            resp_id = insert_response(model_ids[name], prompt_id, resp)

            nlp = compute_all_metrics(resp, item.get("reference", ""))
            judged = judge_response(item["prompt"], resp, item.get("reference", ""), judge_mode)
            scores = {**nlp, "judge_score": judged.get("overall", 5)}
            insert_metrics(resp_id, scores)

    print_leaderboard()
    print(f"✅ Evaluation completed. Results saved to {DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--dataset", default="sample_prompts.json")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.dataset, args.real))
