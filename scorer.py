"""
metrics/scorer.py

NLP metrics for evaluating model responses against reference answers.

Supports:
  - BLEU  (nltk)
  - ROUGE (rouge-score)
  - BERTScore (bert-score)
  - Lightweight fallback scorer (no external dependencies)
"""


# ── BLEU ──────────────────────────────────────────────────────────────────────


def compute_bleu(response: str, reference: str) -> float:
    """
    Compute sentence-level BLEU score.

    Requires: pip install nltk
    Returns value in [0, 1].
    """
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        ref_tokens = reference.lower().split()
        hyp_tokens = response.lower().split()
        smoother = SmoothingFunction().method1
        return round(
            sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother), 4
        )
    except ImportError:
        return _fallback_overlap(response, reference)


# ── ROUGE ─────────────────────────────────────────────────────────────────────


def compute_rouge(response: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score.

    Requires: pip install rouge-score
    Returns value in [0, 1].
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, response)
        return round(scores["rougeL"].fmeasure, 4)
    except ImportError:
        return _fallback_overlap(response, reference)


# ── BERTScore ─────────────────────────────────────────────────────────────────


def compute_bertscore(response: str, reference: str) -> float:
    """
    Compute BERTScore F1.

    Requires: pip install bert-score torch
    Returns value in [0, 1].
    """
    try:
        from bert_score import score as bert_score

        _, _, f1 = bert_score([response], [reference], lang="en", verbose=False)
        return round(f1[0].item(), 4)
    except ImportError:
        return _fallback_overlap(response, reference)


# ── Fallback ──────────────────────────────────────────────────────────────────


def _fallback_overlap(response: str, reference: str) -> float:
    """
    Lightweight token-overlap similarity when NLP libraries are unavailable.
    Returns Jaccard similarity as a proxy metric.
    """
    stop = {
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
        "to",
        "for",
        "of",
    }
    r1 = set(response.lower().split()) - stop
    r2 = set(reference.lower().split()) - stop
    if not r1 and not r2:
        return 1.0
    if not r1 or not r2:
        return 0.0
    return round(len(r1 & r2) / len(r1 | r2), 4)


# ── Combined Scorer ───────────────────────────────────────────────────────────


def compute_all_metrics(response: str, reference: str) -> dict[str, float]:
    """
    Compute BLEU, ROUGE, and BERTScore in one call.

    Returns a dict with keys: bleu, rouge, bertscore.
    """
    return {
        "bleu": compute_bleu(response, reference),
        "rouge": compute_rouge(response, reference),
        "bertscore": compute_bertscore(response, reference),
    }
