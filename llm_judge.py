"""
llm_judge.py - LLM-as-a-Judge with heuristic fallback
"""

import json
import os
import re

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator.

Score the response from 1-10 on:
- Correctness
- Completeness
- Clarity
- Reasoning

Prompt: {prompt}
Response: {response}
Reference: {reference}

Return ONLY JSON:
{
  "correctness": <1-10>,
  "completeness": <1-10>,
  "clarity": <1-10>,
  "reasoning": <1-10>,
  "overall": <1-10>,
  "reason": "<brief explanation>"
}"""


def _parse_judge_output(raw: str) -> dict:
    """Extract JSON from judge output with graceful fallback."""
    try:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
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
    }


def heuristic_judge(response: str, reference: str = "") -> dict:
    """Simple heuristic fallback when no LLM API is available."""
    stop = {"the", "a", "an", "is", "are", "and", "or", "but", "in", "on", "to", "for", "of"}
    r1 = set(response.lower().split()) - stop
    r2 = set(reference.lower().split()) - stop
    overlap = len(r1 & r2) / len(r1) if r1 else 0.5

    score = max(1, min(10, int(overlap * 10)))

    return {
        "correctness": score,
        "completeness": score,
        "clarity": score,
        "reasoning": score,
        "overall": score,
        "reason": "Heuristic evaluation (no API key available)",
        "is_heuristic": True,
    }


def judge_response(prompt: str, response: str, reference: str = "", judge: str = "auto") -> dict:
    """Main judge function with fallback logic."""
    if judge == "heuristic":
        return heuristic_judge(response, reference)

    # Try OpenAI first
    if os.getenv("OPENAI_API_KEY"):
        try:
            import openai

            client = openai.OpenAI()
            result = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": JUDGE_PROMPT_TEMPLATE.format(
                            prompt=prompt, response=response, reference=reference
                        ),
                    }
                ],
                max_tokens=300,
            )
            return _parse_judge_output(result.choices[0].message.content)
        except Exception:
            pass

    # Try Claude
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            client = anthropic.Anthropic()
            result = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": JUDGE_PROMPT_TEMPLATE.format(
                            prompt=prompt, response=response, reference=reference
                        ),
                    }
                ],
            )
            return _parse_judge_output(result.content[0].text)
        except Exception:
            pass

    # Final fallback
    return heuristic_judge(response, reference)
