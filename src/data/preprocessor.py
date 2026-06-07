"""Text preprocessing pipeline for NLP sentiment analysis."""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg in ["stopwords", "wordnet", "punkt_tab"]:
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg != "punkt_tab" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

_EMOJI_MAP = {
    r":[-]?\)": " happy ",
    r":[-]?\(": " sad ",
    r":[Dd]": " laugh ",
    r"<3": " love ",
    r":\*": " kiss ",
    r";\)": " wink ",
}


def clean_text(text: str, remove_stops: bool = True, lemmatize: bool = True) -> str:
    """Full text cleaning pipeline."""
    if not isinstance(text, str):
        return ""

    for pattern, replacement in _EMOJI_MAP.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()

    if remove_stops:
        tokens = [t for t in tokens if t not in _stop_words and len(t) > 2]

    if lemmatize:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Apply cleaning pipeline to a DataFrame column."""
    df = df.copy()
    df["clean_text"] = df[text_col].apply(clean_text)
    df["text_length"] = df[text_col].apply(lambda x: len(str(x).split()))
    df["clean_length"] = df["clean_text"].apply(lambda x: len(x.split()))
    return df
