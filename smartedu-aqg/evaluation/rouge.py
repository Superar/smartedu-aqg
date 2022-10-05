from typing import Union
from rouge_score.rouge_scorer import RougeScorer


def score(gold: str, generated: str) -> dict:
    scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
    return scorer.score(gold, generated)
