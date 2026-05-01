"""
Transform raw Goodreads book data into feature matrices for unsupervised learning.
"""

import re
import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import nltk
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

try:
    _STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _STOPWORDS = set(stopwords.words("english"))


def _clean_text(text: str) -> str:
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)  # strip HTML
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return " ".join(text.lower().split())


def build_text_corpus(df: pd.DataFrame) -> pd.Series:
    """
    Combine all textual fields into a single document per book.
    Uses: title, author, shelves, OL subjects, OL description, OL first sentence, my_review.
    """
    parts = []
    for col in ["title", "author", "shelves", "ol_subjects", "ol_description", "ol_first_sentence", "my_review"]:
        if col in df.columns:
            cleaned = df[col].fillna("").astype(str).apply(_clean_text)
            parts.append(cleaned)
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))

    return parts[0].str.cat(parts[1:], sep=" ").str.strip()


def build_tfidf_features(corpus: pd.Series, max_features: int = 500, ngram_range=(1, 2)) -> tuple:
    """Return a TF-IDF matrix and the fitted vectorizer."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.9,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    logger.info("TF-IDF matrix: %s, vocab size: %d", tfidf_matrix.shape, len(vectorizer.vocabulary_))
    return tfidf_matrix, vectorizer


def build_genre_features(df: pd.DataFrame) -> tuple:
    """
    One-hot encode shelves (Goodreads user tags) and OL subjects.
    Returns the binary matrix and the binarizer.
    """
    genre_lists = []
    for _, row in df.iterrows():
        genres = set()
        if "shelves" in df.columns and isinstance(row.get("shelves"), str):
            for s in row["shelves"].split(","):
                s = s.strip().lower().replace("-", " ")
                if s and s not in ("to-read", "currently-reading", "read", "default"):
                    genres.add(s)
        if "ol_subjects" in df.columns and isinstance(row.get("ol_subjects"), str):
            for s in row["ol_subjects"].split("|"):
                s = s.strip().lower()
                if s and len(s) < 40:
                    genres.add(s)
        genre_lists.append(list(genres))

    if not any(genre_lists):
        return np.zeros((len(df), 0)), None

    mlb = MultiLabelBinarizer(sparse_output=True)
    genre_matrix = mlb.fit_transform(genre_lists)

    min_count = max(2, int(0.05 * len(df)))
    col_sums = np.array(genre_matrix.sum(axis=0)).flatten()
    keep = col_sums >= min_count
    genre_matrix = genre_matrix[:, keep]
    kept_classes = np.array(mlb.classes_)[keep]

    logger.info("Genre features: %d books x %d genres (min_count=%d)", genre_matrix.shape[0], genre_matrix.shape[1], min_count)
    return genre_matrix, kept_classes


def build_numeric_features(df: pd.DataFrame) -> np.ndarray:
    """Scale numeric columns: avg_rating, pages, original_year."""
    num_cols = []
    for col in ["avg_rating", "pages", "original_year"]:
        if col in df.columns:
            num_cols.append(df[col].fillna(df[col].median()))

    if not num_cols:
        return np.zeros((len(df), 0))

    numeric = pd.concat(num_cols, axis=1).values
    scaler = StandardScaler()
    return scaler.fit_transform(numeric)


def build_combined_features(df: pd.DataFrame, max_tfidf: int = 500) -> tuple:
    """
    Build a combined feature matrix from text TF-IDF, genre one-hot, and numeric features.
    Returns (combined_matrix, corpus, tfidf_vectorizer, genre_classes).
    """
    corpus = build_text_corpus(df)
    tfidf_matrix, vectorizer = build_tfidf_features(corpus, max_features=max_tfidf)
    genre_matrix, genre_classes = build_genre_features(df)
    numeric_matrix = build_numeric_features(df)

    from scipy.sparse import hstack, issparse, csr_matrix

    parts = [tfidf_matrix]
    if genre_matrix.shape[1] > 0:
        if not issparse(genre_matrix):
            genre_matrix = csr_matrix(genre_matrix)
        parts.append(genre_matrix.astype(float) * 0.5)
    if numeric_matrix.shape[1] > 0:
        parts.append(csr_matrix(numeric_matrix) * 0.3)

    combined = hstack(parts)
    logger.info("Combined feature matrix: %s", combined.shape)

    return combined, corpus, vectorizer, genre_classes
