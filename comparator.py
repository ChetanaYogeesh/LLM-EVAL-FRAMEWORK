"""
judges/comparator.py

Pairwise LLM Response Comparator
Compares two model responses across multiple criteria and generates reports.
"""


# ============================================================================
# PAIRWISE COMPARISON FUNCTIONS
# ============================================================================


def compare_clarity(response_a: str, response_b: str) -> tuple[str, int, int]:
    """
    Compares two responses based on clarity and readability.

    Scoring evaluates:
    - Average sentence length (shorter = more readable)
    - Word complexity (fewer long words = better)
    - Structural clarity (use of paragraphs or line breaks)

    Returns:
        tuple: (winner, score_a, score_b)
    """

    def clarity_score(text: str) -> int:
        score = 5  # Neutral baseline

        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if sentences:
            avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_len < 15:
                score += 3
            elif avg_len < 25:
                score += 1
            else:
                score -= 1

        words = text.split()
        long_words = [w for w in words if len(w) > 10]
        complexity_ratio = len(long_words) / len(words) if words else 0

        if complexity_ratio < 0.1:
            score += 2
        elif complexity_ratio > 0.3:
            score -= 2

        return max(1, min(10, score))

    score_a = clarity_score(response_a)
    score_b = clarity_score(response_b)
    winner = "A" if score_a > score_b else ("B" if score_b > score_a else "tie")

    return winner, score_a, score_b


def compare_accuracy(
    response_a: str, response_b: str, reference: str
) -> tuple[str, int, int]:
    """
    Compares responses against a reference answer.

    Scoring:
    - Keyword overlap with reference
    - Concept matching
    - Penalizes low overlap

    Returns:
        tuple: (winner, score_a, score_b)
    """

    def accuracy_score(text: str, ref: str) -> int:
        text, ref = text.lower(), ref.lower()
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
        }
        ref_words = set(ref.split()) - stop_words
        text_words = set(text.split()) - stop_words
        overlap = len(ref_words & text_words) / len(ref_words) if ref_words else 0
        return max(1, min(10, int(overlap * 10)))

    score_a = accuracy_score(response_a, reference)
    score_b = accuracy_score(response_b, reference)
    winner = "A" if score_a > score_b else ("B" if score_b > score_a else "tie")

    return winner, score_a, score_b


def compare_helpfulness(response_a: str, response_b: str) -> tuple[str, int, int]:
    """
    Compares responses based on usefulness.

    Helpful responses include examples, explanations, and practical info.

    Returns:
        tuple: (winner, score_a, score_b)
    """

    def helpfulness_score(text: str) -> int:
        score = 5
        word_count = len(text.lower().split())

        if word_count > 100:
            score += 2
        elif word_count < 30:
            score -= 1

        if "example" in text.lower() or "for example" in text.lower():
            score += 2

        explanation_terms = ["because", "therefore", "so that", "in order to"]
        if any(term in text.lower() for term in explanation_terms):
            score += 1

        return max(1, min(10, score))

    score_a = helpfulness_score(response_a)
    score_b = helpfulness_score(response_b)
    winner = "A" if score_a > score_b else ("B" if score_b > score_a else "tie")

    return winner, score_a, score_b


# ============================================================================
# INDIVIDUAL SCORING FUNCTIONS
# ============================================================================


def score_clarity(text: str) -> int:
    score = 5
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if sentences:
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_len < 15:
            score += 3
        elif avg_len < 25:
            score += 1
        else:
            score -= 1
    if "\n" in text:
        score += 1
    return max(1, min(10, score))


def score_completeness(text: str) -> int:
    words = len(text.split())
    if words < 20:
        score = 3
    elif words < 50:
        score = 6
    elif words < 150:
        score = 8
    else:
        score = 9
    if "example" in text.lower():
        score += 1
    return max(1, min(10, score))


def score_conciseness(text: str) -> int:
    word_count = len(text.split())
    if 50 <= word_count <= 200:
        score = 9
    elif word_count < 30:
        score = 5
    else:
        score = 7
    words = text.lower().split()
    unique_ratio = len(set(words)) / len(words) if words else 1
    if unique_ratio < 0.5:
        score -= 2
    return max(1, min(10, score))


def score_tone(text: str) -> int:
    score = 7
    if any(w in text.lower() for w in ["yeah", "nah", "gonna", "wanna"]):
        score -= 2
    if "!!!" in text:
        score -= 1
    if any(t in text.lower() for t in ["however", "therefore", "moreover"]):
        score += 1
    return max(1, min(10, score))


def score_response(
    response: str, criteria: list[str], reference: str = ""
) -> dict[str, int]:
    """Score a single response across multiple criteria."""
    scores = {}
    for criterion in criteria:
        if criterion == "clarity":
            scores["clarity"] = score_clarity(response)
        elif criterion == "completeness":
            scores["completeness"] = score_completeness(response)
        elif criterion == "conciseness":
            scores["conciseness"] = score_conciseness(response)
        elif criterion == "tone":
            scores["tone"] = score_tone(response)
        else:
            scores[criterion] = 5
    return scores


def calculate_overall_score(
    criterion_scores: dict[str, int], weights: dict[str, float] = None
) -> float:
    """Combine individual criterion scores into a weighted overall score."""
    if not criterion_scores:
        return 0.0
    if weights is None:
        weights = {k: 1 / len(criterion_scores) for k in criterion_scores}
    total = sum(
        score * weights.get(crit, 0) for crit, score in criterion_scores.items()
    )
    total_weight = sum(weights.get(crit, 0) for crit in criterion_scores)
    return total / total_weight if total_weight else 0.0


# ============================================================================
# REPORT GENERATION
# ============================================================================


def generate_comparison_report(
    prompt: str, response_a: str, response_b: str, reference: str = ""
) -> dict:
    """Generate a full comparison report with pairwise comparisons and scores."""
    breakdown = {
        "clarity": compare_clarity(response_a, response_b),
        "accuracy": compare_accuracy(response_a, response_b, reference),
        "helpfulness": compare_helpfulness(response_a, response_b),
    }

    criteria = ["clarity", "completeness", "conciseness", "tone"]
    scores_a = score_response(response_a, criteria)
    scores_b = score_response(response_b, criteria)

    overall_a = calculate_overall_score(scores_a)
    overall_b = calculate_overall_score(scores_b)

    winner = "A" if overall_a > overall_b else ("B" if overall_b > overall_a else "tie")
    recommendation = (
        f"Use Response {winner} for production."
        if winner != "tie"
        else "Both responses are equivalent."
    )

    return {
        "prompt": prompt,
        "winner": winner,
        "score_a": round(overall_a, 2),
        "score_b": round(overall_b, 2),
        "breakdown": breakdown,
        "detailed_scores_a": scores_a,
        "detailed_scores_b": scores_b,
        "recommendation": recommendation,
    }


def format_report(report: dict) -> str:
    """Convert report dict into readable formatted output."""
    lines = [
        "=" * 70,
        "PAIRWISE COMPARISON REPORT",
        "=" * 70,
        f"Prompt: {report['prompt']}\n",
        f"Response A Score: {report['score_a']}/10",
        f"Response B Score: {report['score_b']}/10\n",
        f"Winner: {report['winner']}\n",
        "Detailed Breakdown:",
        "-" * 70,
    ]
    for criterion, (winner, a, b) in report["breakdown"].items():
        lines.append(f"  {criterion.capitalize():15s} A={a}  B={b}  → Winner: {winner}")
    lines += [
        "\nDetailed Scores:",
        "-" * 70,
        f"  {'Criterion':15s} {'Response A':12s} {'Response B':12s}",
    ]
    all_criteria = set(
        list(report["detailed_scores_a"].keys())
        + list(report["detailed_scores_b"].keys())
    )
    for crit in sorted(all_criteria):
        a_val = report["detailed_scores_a"].get(crit, "-")
        b_val = report["detailed_scores_b"].get(crit, "-")
        lines.append(f"  {crit.capitalize():15s} {str(a_val):12s} {str(b_val):12s}")
    lines += [
        "\nRecommendation:",
        report["recommendation"],
        "=" * 70,
    ]
    return "\n".join(lines)
