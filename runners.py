"""
models/runners.py

Model runners for OpenAI and Anthropic (Claude).

Each runner accepts a prompt string and returns the model's text response.
Set API keys via environment variables:
  - OPENAI_API_KEY
  - ANTHROPIC_API_KEY
"""

import asyncio
import os
from collections.abc import Callable

# ── OpenAI Runner ─────────────────────────────────────────────────────────────


async def run_openai(prompt: str, model: str = "gpt-4.1") -> str:
    """
    Send a prompt to an OpenAI chat model and return the text response.

    Requires: OPENAI_API_KEY environment variable.
    """
    try:
        import openai

        client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
        )
        return response.choices[0].message.content or ""
    except ImportError as err:
        raise ImportError("Install openai: pip install openai") from err
    except KeyError as err:
        raise OSError("Set the OPENAI_API_KEY environment variable.") from err


# ── Anthropic (Claude) Runner ─────────────────────────────────────────────────


async def run_claude(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """
    Send a prompt to a Claude model and return the text response.

    Requires: ANTHROPIC_API_KEY environment variable.
    """
    try:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = await client.messages.create(
            model=model,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text or ""
    except ImportError as err:
        raise ImportError("Install anthropic: pip install anthropic") from err
    except KeyError as err:
        raise OSError("Set the ANTHROPIC_API_KEY environment variable.") from err


# ── Mock Runners (for testing without API keys) ───────────────────────────────


async def mock_model_a(prompt: str) -> str:
    """Mock model that returns a clear, example-rich answer."""
    return (
        f"Here is a clear explanation of '{prompt[:40]}...'. "
        "For example, consider how this concept applies in practice. "
        "Therefore, the key takeaway is that clarity and structure improve understanding. "
        "This approach is widely used because it reduces ambiguity and helps readers follow along."
    )


async def mock_model_b(prompt: str) -> str:
    """Mock model that returns a verbose but less structured answer."""
    return (
        f"Regarding your query about '{prompt[:40]}', it is important to note that "
        "there are multitudinous considerations and ramifications that one must take into account "
        "when attempting to formulate an appropriately comprehensive response to such an inquiry. "
        "The utilization of sophisticated terminological constructs can sometimes obfuscate meaning."
    )


# ── Batch Evaluation Runner ───────────────────────────────────────────────────


async def evaluate_prompts(
    prompts: list[dict],
    runners: list[Callable],
    runner_names: list[str] | None = None,
) -> list[dict]:
    """
    Run all prompts through all model runners concurrently.

    Args:
        prompts:       List of dicts with keys: 'prompt', 'reference' (optional)
        runners:       List of async callables (e.g. [run_openai, run_claude])
        runner_names:  Optional labels for each runner

    Returns:
        List of result dicts with prompt, reference, and per-model responses
    """
    if runner_names is None:
        runner_names = [f"model_{i}" for i in range(len(runners))]

    results = []

    for item in prompts:
        prompt_text = item["prompt"]
        reference = item.get("reference", "")

        tasks = [runner(prompt_text) for runner in runners]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        result = {
            "prompt": prompt_text,
            "reference": reference,
            "category": item.get("category", "general"),
            "responses": {},
        }

        for name, resp in zip(runner_names, responses, strict=False):
            if isinstance(resp, Exception):
                result["responses"][name] = f"[ERROR] {resp}"
            else:
                result["responses"][name] = resp

        results.append(result)

    return results
