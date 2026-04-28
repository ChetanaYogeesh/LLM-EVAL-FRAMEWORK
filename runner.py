"""
runner.py - Main evaluation runner (Professional Pipeline)
"""

import asyncio
import argparse
from pathlib import Path

from llm_judge import judge_response
from scorer import compute_all_metrics
from sqlite_store import init_db, create_experiment, upsert_model, insert_prompt, insert_response, insert_metrics, insert_pairwise
from runners import evaluate_prompts, mock_model_a, mock_model_b, run_openai, run_claude
from rankings import print_leaderboard

async def run_evaluation(dataset_path="sample_prompts.json", use_real=False, judge_mode="auto", name="eval_run"):
    init_db()
    create_experiment(name, config={"real": use_real, "judge": judge_mode})

    with open(dataset_path) as f:
        prompts = json.load(f)

    runners = [run_openai, run_claude] if use_real else [mock_model_a, mock_model_b]
    runner_names = ["GPT-4o", "Claude"] if use_real else ["Mock-A", "Mock-B"]

    model_ids = {name: upsert_model(name) for name in runner_names}

    results = await evaluate_prompts(prompts, runners, runner_names)

    for item in results:
        prompt_id = insert_prompt(item["prompt"], item.get("reference", ""), item.get("category", "general"))
        for name in runner_names:
            resp = item["responses"].get(name, "")
            resp_id = insert_response(model_ids[name], prompt_id, resp)
            metrics = compute_all_metrics(resp, item.get("reference", ""))
            judge_scores = judge_response(item["prompt"], resp, item.get("reference", ""), judge_mode)
            metrics["judge_score"] = judge_scores.get("overall", 5)
            insert_metrics(resp_id, metrics)

    print_leaderboard()
    print("✅ Evaluation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--dataset", default="sample_prompts.json")
    args = parser.parse_args()
    asyncio.run(run_evaluation(args.dataset, args.real))