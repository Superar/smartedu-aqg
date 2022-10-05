import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim


def create_context(text: list, answers: list, model: str) -> np.ndarray:
    """
    Creates the context for each answer in `answers` using five
    sentences from `text` by using a BERT model for sentence similarity.

    Args:
        text (list): List of sentences for the context
        answers (list): List of answers from which to select the context
        model (str): Sentence similarity BERT model

    Returns:
        np.ndarray: 2D array with the indices from `text` corresponding to
                  the contexts for each answer
    """
    model = SentenceTransformer(model)
    text_embeddings = model.encode(text)
    answers_embeddings = model.encode(answers)
    sims = pytorch_cos_sim(answers_embeddings, text_embeddings)
    context_indices = np.argsort(sims, axis=1)[:, -5:]
    return context_indices.numpy()
