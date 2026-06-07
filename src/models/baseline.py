"""VADER lexical baseline for sentiment classification."""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def vader_predict(text: str) -> str:
    scores = _analyzer.polarity_scores(str(text))
    compound = scores["compound"]
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


def vader_predict_batch(texts: pd.Series) -> pd.Series:
    return texts.apply(vader_predict)


def vader_scores(text: str) -> dict:
    """Return raw VADER scores dict for a single text."""
    return _analyzer.polarity_scores(str(text))
