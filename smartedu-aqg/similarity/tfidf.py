from pathlib import Path

import numpy as np
import tqdm
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities.docsim import Similarity
from gensim.test.utils import get_tmpfile
from nltk.tokenize import word_tokenize


def create_context(idfpath: Path, text: list, answers: list) -> np.ndarray:
    """
    Creates the context for each answer in `answers` using five
    sentences from `text` by using TF-IDF similarity measure.

    Args:
        idfpath (Path): Path to directory to calculate IDF values
        text (list): List of sentences for the context
        answers (list): List of answers from which to select the context

    Returns:
        np.ndarray: 2D array with the indices from `text` corresponding to
                  the contexts for each answer
    """
    text_tokens = [word_tokenize(l, language='portuguese') for l in text]
    answers_tokens = [word_tokenize(l, language='portuguese') for l in answers]

    tf_idf_corpus = list()
    for filepath in tqdm.tqdm(list(idfpath.iterdir())):
        with filepath.open('rU', encoding='utf-8') as file_:
            tfidf_tokens = word_tokenize(file_.read(), language='portuguese')
            tf_idf_corpus.append(tfidf_tokens)

    dct = Dictionary(tf_idf_corpus)
    text_bow = [dct.doc2bow(t) for t in text_tokens]
    answers_bow = [dct.doc2bow(a) for a in answers_tokens]
    tf_idf_bow = [dct.doc2bow(t) for t in tf_idf_corpus]
    tfidf = TfidfModel(tf_idf_bow)

    # Convert corpus to TF-IDF values
    text_tfidf = tfidf[text_bow]
    answers_tfidf = tfidf[answers_bow]

    # Create index to query similarity values
    index = Similarity(get_tmpfile('index'), text_tfidf, num_features=len(dct))
    sims = index[answers_tfidf]
    context_indices = np.argsort(sims, axis=1)[:, -5:]
    return context_indices
