#!/bin/bash
set -e

rm -f EvalDash.py

# Run ruff once on everything upfront, then stage ALL files together
ruff check . --fix --unsafe-fixes
ruff format .

# Stage everything at once — no unstaged files left to conflict
git add -A

# Now commit in logical groups using git commit with pathspecs
git commit scorer.py -m "feat(scorer): add BLEU, ROUGE, BERTScore with Jaccard fallback"
git commit comparator.py -m "feat(comparator): add pairwise clarity/accuracy/helpfulness scorer"
git commit llm_judge.py -m "feat(judge): add LLM-as-a-judge with GPT, Claude, heuristic fallback"
git commit sqlite_store.py rankings.py -m "feat(store): add SQLite persistence and leaderboard generator"
git commit runners.py runner.py ollama_evaluator.py crewai_evaluator.py -m "feat(evaluators): add Ollama, CrewAI, and professional pipeline runners"
git commit dashboard.py -m "feat(dashboard): integrate all evaluators into unified Streamlit UI"

git push origin main
