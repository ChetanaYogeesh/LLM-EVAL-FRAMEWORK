### Big Picture Overview

Your project has **three main evaluation systems**:

1. **Ollama Evaluator** (`ollama_evaluator.py`) — Fast, local, no API keys
2. **CrewAI Evaluator** (`crewai_evaluator.py`) — Multi-agent, more intelligent
3. **Professional Pipeline** — Full SQLite + multi-model + advanced scoring (your original big system)

The other files support these three systems.

---

### How All Files Work Together (Simple Diagram)

```
User runs → dashboard.py
                ↓
        Launch Evaluators page
                ↓
   ┌────────────┬────────────┬────────────────────┐
   │            │            │                    │
Ollama    CrewAI     Professional Pipeline
Evaluator  Evaluator     (runner.py)
   │            │            │
   └────────────┴────────────┘
                ↓
         Saves results
                ↓
   evaluation_results.json   +   evals.db (SQLite)
                ↓
         dashboard.py shows results
                ↓
   Results page (separate tabs for each)
```

---

### Role of Each File

| File                    | Purpose | Used By                          | Required? |
|-------------------------|--------|----------------------------------|---------|
| **dashboard.py**        | Main UI, buttons, result viewer | User (Streamlit)                | Yes (Main) |
| **ollama_evaluator.py** | Simple local evaluator using Ollama | Launch button                   | Yes |
| **crewai_evaluator.py** | Multi-agent CrewAI evaluator     | Launch button                   | Yes |
| **runner.py**           | Full professional evaluation pipeline | "Professional Pipeline" button  | Yes |
| **runners.py**          | Model runners (OpenAI, Claude, Mock) | runner.py                       | Yes |
| **llm_judge.py**        | LLM-as-a-Judge + heuristic fallback | runner.py + evaluators          | Yes |
| **scorer.py**           | BLEU, ROUGE, BERTScore metrics   | runner.py                       | Yes |
| **sqlite_store.py**     | Saves everything to `evals.db`   | runner.py + dashboard           | Yes |
| **rankings.py**         | Prints leaderboard               | runner.py                       | Optional |

---

### Data Flow Explained (Step by Step)

1. **User opens dashboard** → `dashboard.py`
2. User clicks **"Run Ollama Evaluator"** → calls `ollama_evaluator.py`
   - Uses `litellm` to call local Ollama
   - Uses simple heuristic detectors (hallucination, bias, toxicity)
   - Saves result to `evaluation_results.json`

3. User clicks **"Run CrewAI Evaluator"** → calls `crewai_evaluator.py`
   - Uses CrewAI with multiple specialized agents
   - Saves result to `evaluation_results.json`

4. User clicks **"Run Professional Pipeline"** → calls `runner.py`
   - Loads prompts
   - Runs models via `runners.py`
   - Scores with `scorer.py` (NLP metrics)
   - Scores with `llm_judge.py` (LLM judge)
   - Saves **everything** to SQLite (`evals.db`) using `sqlite_store.py`
   - Shows leaderboard using `rankings.py`

5. **Viewing Results**:
   - **Ollama & CrewAI** results → shown in "🔍 Results" page (JSON files)
   - **Professional Pipeline** results → shown in Leaderboard, Responses, Pairwise, Metrics pages (from SQLite)

---

### Simple Summary (Like explaining to a 10-year-old)

Think of it like this:

- **dashboard.py** = The control room with big buttons
- **ollama_evaluator.py** = The quick local tester (uses your home computer)
- **crewai_evaluator.py** = The smart team of agents working together
- **runner.py** = The full factory production line (most powerful, uses database)
- **llm_judge.py** = The strict teacher who grades the answers
- **scorer.py** = The math tools that measure how good the answers are
- **sqlite_store.py** = The filing cabinet that remembers everything
- **runners.py** = The actual robots that generate answers (GPT, Claude, or fake ones)

All of them work together so you can test AI responses in different ways and compare the results easily.

