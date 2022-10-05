from typing import Union

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def score(gold: str, generated: str) -> Union[float, int]:
    """
    Calculate the BLEU score between the two sentences.

    Args:
        gold (str): Reference sentence
        generated (str): Automatically generated sentence to compare to `gold`

    Returns:
        Union[float, int]: BLEU score
    """
    gold_toks = word_tokenize(gold, language='portuguese')
    generated_toks = word_tokenize(generated, language='portuguese')
    smoothing = SmoothingFunction()
    return sentence_bleu([gold_toks], generated_toks,
                         smoothing_function=smoothing.method1)
