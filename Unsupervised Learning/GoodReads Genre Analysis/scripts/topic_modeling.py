"""
Topic modeling with Latent Dirichlet Allocation (LDA) using Gensim.
"""

import logging
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models

logger = logging.getLogger(__name__)

try:
    _STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _STOPWORDS = set(stopwords.words("english"))

EXTRA_STOPS = {
    "book", "books", "read", "reading", "novel", "story", "author",
    "one", "first", "new", "like", "also", "would", "us", "may",
    "edition", "published", "isbn", "page", "pages", "chapter",
}
_STOPWORDS |= EXTRA_STOPS


def _tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    return [w for w in text.split() if len(w) > 2 and w not in _STOPWORDS]


def build_lda_model(
    corpus_texts: pd.Series,
    n_topics: int = 5,
    passes: int = 15,
    chunksize: int = 50,
) -> tuple:
    """
    Train an LDA model on the text corpus.
    Returns (lda_model, gensim_corpus, dictionary, tokenized_docs).
    """
    tokenized = corpus_texts.apply(_tokenize).tolist()

    freq = defaultdict(int)
    for doc in tokenized:
        for token in doc:
            freq[token] += 1
    tokenized = [[t for t in doc if freq[t] > 1] for doc in tokenized]

    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    lda = models.LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=42,
        passes=passes,
        chunksize=chunksize,
        alpha="auto",
        eta="auto",
    )

    coherence = models.CoherenceModel(
        model=lda, texts=tokenized, dictionary=dictionary, coherence="c_v"
    )
    score = coherence.get_coherence()
    logger.info("LDA (%d topics): coherence c_v=%.3f", n_topics, score)

    return lda, bow_corpus, dictionary, tokenized


def find_optimal_topics(corpus_texts: pd.Series, topic_range=range(3, 12)) -> dict:
    """Evaluate coherence for different topic counts."""
    tokenized = corpus_texts.apply(_tokenize).tolist()

    freq = defaultdict(int)
    for doc in tokenized:
        for token in doc:
            freq[token] += 1
    tokenized = [[t for t in doc if freq[t] > 1] for doc in tokenized]

    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    results = {"n_topics": [], "coherence": []}
    for n in topic_range:
        lda = models.LdaModel(
            corpus=bow_corpus, id2word=dictionary, num_topics=n,
            random_state=42, passes=10, alpha="auto", eta="auto",
        )
        cm = models.CoherenceModel(model=lda, texts=tokenized, dictionary=dictionary, coherence="c_v")
        c = cm.get_coherence()
        results["n_topics"].append(n)
        results["coherence"].append(c)
        logger.info("Topics=%d  coherence=%.3f", n, c)

    return results


def get_topic_words(lda_model, n_words: int = 10) -> list[list[tuple[str, float]]]:
    """Get top words per topic with their probabilities."""
    return [lda_model.show_topic(t, topn=n_words) for t in range(lda_model.num_topics)]


def assign_topics(lda_model, bow_corpus) -> np.ndarray:
    """Return the dominant topic index for each document."""
    dominant = []
    for bow in bow_corpus:
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0.0)
        dominant.append(max(topic_dist, key=lambda x: x[1])[0])
    return np.array(dominant)
