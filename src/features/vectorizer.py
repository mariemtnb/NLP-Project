"""TF-IDF feature extraction with unigrams and bigrams."""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")


def build_tfidf(
    texts: pd.Series,
    max_features: int = 50_000,
    ngram_range: tuple = (1, 2),
    min_df: int = 3,
    sublinear_tf: bool = True,
) -> tuple[TfidfVectorizer, spmatrix]:
    """Fit a TF-IDF vectorizer and return (vectorizer, matrix)."""
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        analyzer="word",
    )
    X = vec.fit_transform(texts)
    joblib.dump(vec, TFIDF_PATH)
    return vec, X


def load_tfidf() -> TfidfVectorizer:
    return joblib.load(TFIDF_PATH)


def transform(vec: TfidfVectorizer, texts: pd.Series) -> spmatrix:
    return vec.transform(texts)
