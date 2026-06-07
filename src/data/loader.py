"""Data loading and sampling utilities for the Yelp dataset."""

import json
import os
import pandas as pd
from typing import Optional


DATA_RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
DATA_PROCESSED = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def load_sample(path: Optional[str] = None, n: Optional[int] = None) -> pd.DataFrame:
    """Load the pre-sampled 50k review dataset."""
    if path is None:
        path = os.path.join(DATA_PROCESSED, "sample_50k.json")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if n and len(records) >= n:
                    break

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def load_business(path: Optional[str] = None) -> pd.DataFrame:
    """Load Yelp business metadata."""
    if path is None:
        path = os.path.join(DATA_RAW, "yelp_academic_dataset_business.json")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records.append({
                    "business_id": r["business_id"],
                    "name": r["name"],
                    "city": r.get("city", ""),
                    "state": r.get("state", ""),
                    "stars": r.get("stars", 0),
                    "review_count": r.get("review_count", 0),
                    "categories": r.get("categories", ""),
                })
    return pd.DataFrame(records)


def sentiment_label(stars: int) -> str:
    if stars in [1, 2]:
        return "negative"
    elif stars == 3:
        return "neutral"
    return "positive"
