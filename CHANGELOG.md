# Changelog

All notable changes to **Agent Evaluator Crew** are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---
## [Unreleased]

### Fixed
- Ollama evaluator falls back to OpenRouter gpt-4o-mini when localhost:11434 unreachable
- Python pinned to 3.11 via runtime.txt for Streamlit Cloud compatibility
- sqlite_store missing get_all_metrics_df, get_pairwise_df, get_experiments functions
- dashboard ImportError on startup from wrong sqlite_store imports
- pages/4_Run_Eval asyncio loop conflict with Streamlit runtime
- ollama_evaluator pydantic import removed — returns plain dict
- requirements.txt missing litellm, pyyaml, nltk, rouge-score

### Added  
- 8 Streamlit pages: Launch, Results, Overview, Run Eval, Leaderboard, Responses, Pairwise, Metrics
- OpenRouter fallback in ollama_evaluator with OPENAI_API_KEY secret

## [Unreleased]

### Added
- Initial multi-agent evaluation framework with six specialized agents
- Five production tools: `TraceParserTool`, `MetricCalculatorTool`, `SafetyGuardTool`, `HumanReviewTool`, `RegressionComparatorTool`
- Streamlit dashboard with KPI overview, safety analysis, bottleneck view
- Simple (Ollama) and full CrewAI evaluation runners
- Hallucination, bias, and toxicity detectors
- Evaluation history persistence (`evaluation_history.json`)
- GitHub Actions CI/CD pipeline (lint, unit tests, integration tests, GHCR publish)
- Streamlit Community Cloud deployment (staging on `main`, prod on tag)
- Dockerfile with multi-stage build and GHCR publishing
