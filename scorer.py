"""
scorer.py - NLP metrics with fallback
"""

def _fallback_overlap(response: str, reference: str) -> float:
    stop = {"the","a","an","is","are","and","or","but","in","on","to","for","of"}
    r1 = set(response.lower().split()) - stop
    r2 = set(reference.lower().split()) - stop
    if not r1 or not r2:
        return 0.5
    return round(len(r1 & r2) / len(r1 | r2), 4)


def compute_all_metrics(response: str, reference: str) -> dict:
    """Returns bleu, rouge, bertscore with fallback."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        bleu = round(sentence_bleu([reference.lower().split()], response.lower().split(), smoothing_function=SmoothingFunction().method1), 4)
    except:
        bleu = _fallback_overlap(response, reference)

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge = round(scorer.score(reference, response)["rougeL"].fmeasure, 4)
    except:
        rouge = _fallback_overlap(response, reference)

    try:
        from bert_score import score as bert_score
        _, _, f1 = bert_score([response], [reference], lang="en", verbose=False)
        bert = round(f1[0].item(), 4)
    except:
        bert = _fallback_overlap(response, reference)

    return {"bleu": bleu, "rouge": rouge, "bertscore": bert}