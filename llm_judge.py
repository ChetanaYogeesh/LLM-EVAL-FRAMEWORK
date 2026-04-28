"""
judges/llm_judge.py

LLM-as-a-Judge evaluator.

Uses a powerful LLM (GPT-4 or Claude) to score responses on:
  - Correctness
  - Completeness
  - Clarity
  - Reasoning quality

Falls back to heuristic scoring when no API key is available.
"""

import json
import os
import re

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing AI-generated responses.

Evaluate the following response to the given prompt.

Score it from 1 to 10 on each of these dimensions:
1. Correctness   – Is the information accurate?
2. Completeness  – Does it fully address the prompt?
3. Clarity       – Is it easy to understand?
4. Reasoning     – Is the logic sound and well-explained?

Prompt:
{prompt}

Model Response:
{response}

Reference Answer (ground truth):
{reference}

Return your evaluation as JSON only, with no additional text:
{{
  "correctness":  <1-10>,
  "completeness": <1-10>,
  "clarity":      <1-10>,
  "reasoning":    <1-10>,
  "overall":      <1-10>,
  "reason":       "<brief explanation>"
}}"""


# ── OpenAI Judge ──────────────────────────────────────────────────────────────


def llm_judge_openai(prompt: str, response: str, reference: str = "") -> dict:
    """
    Use GPT-4 to judge a response.

    Requires: OPENAI_API_KEY environment variable + pip install openai
    """
    try:
        import openai

        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            prompt=prompt, response=response, reference=reference
        )
        result = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=400,
        )
        raw = result.choices[0].message.content or ""
        return _parse_judge_output(raw)
    except (ImportError, KeyError, Exception) as e:
        return heuristic_judge(response, reference, error=str(e))


# ── Claude Judge ──────────────────────────────────────────────────────────────


def llm_judge_claude(prompt: str, response: str, reference: str = "") -> dict:
    """
    Use Claude to judge a response.

    Requires: ANTHROPIC_API_KEY environment variable + pip install anthropic
    """
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            prompt=prompt, response=response, reference=reference
        )
        result = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        raw = result.content[0].text or ""
        return _parse_judge_output(raw)
    except (ImportError, KeyError, Exception) as e:
        return heuristic_judge(response, reference, error=str(e))


# ── Heuristic Fallback Judge ──────────────────────────────────────────────────


def heuristic_judge(response: str, reference: str = "", error: str = "") -> dict:
    """
    Fallback judge using simple heuristics when no LLM API is available.
    Returns scores based on length, keyword overlap, and structure.
    """
    from comparator import score_clarity, score_completeness, score_tone

    stop = {"the", "a", "an", "is", "are", "and", "or"}
    ref_words = set(reference.lower().split()) - stop
    resp_words = set(response.lower().split()) - stop
    overlap = len(ref_words & resp_words) / len(ref_words) if ref_words else 0.5

    correctness = max(1, min(10, int(overlap * 10)))
    completeness = score_completeness(response)
    clarity = score_clarity(response)
    tone = score_tone(response)
    overall = round((correctness + completeness + clarity + tone) / 4, 1)

    return {
        "correctness": correctness,
        "completeness": completeness,
        "clarity": clarity,
        "reasoning": tone,
        "overall": overall,
        "reason": f"Heuristic evaluation (no LLM judge available{': ' + error if error else ''}).",
        "is_heuristic": True,
    }


# ── Parser ────────────────────────────────────────────────────────────────────


def _parse_judge_output(raw: str) -> dict:
    """Extract JSON from the judge's response, with graceful fallback."""
    try:
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find a JSON block anywhere in the text
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {
            "correctness": 5,
            "completeness": 5,
            "clarity": 5,
            "reasoning": 5,
            "overall": 5,
            "reason": "Could not parse judge output.",
            "raw": raw,
        }


# ── Convenience wrapper ───────────────────────────────────────────────────────


def judge_response(
    prompt: str,
    response: str,
    reference: str = "",
    judge: str = "auto",
) -> dict:
    """
    Judge a response using the best available method.

    judge options: 'openai', 'claude', 'heuristic', 'auto'
    'auto' tries openai → claude → heuristic.
    """
    if judge == "openai":
        return llm_judge_openai(prompt, response, reference)
    elif judge == "claude":
        return llm_judge_claude(prompt, response, reference)
    elif judge == "heuristic":
        return heuristic_judge(response, reference)
    else:  # auto
        api_key_openai = os.environ.get("OPENAI_API_KEY", "")
        api_key_claude = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key_openai:
            return llm_judge_openai(prompt, response, reference)
        elif api_key_claude:
            return llm_judge_claude(prompt, response, reference)
        else:
            return heuristic_judge(response, reference)
