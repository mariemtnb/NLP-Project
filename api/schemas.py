from pydantic import BaseModel
from typing import Optional


class PredictRequest(BaseModel):
    text: str
    model: str = "svm"  # "svm" | "nb" | "vader"


class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: Optional[float] = None
    scores: Optional[dict] = None
    model_used: str


class BatchPredictRequest(BaseModel):
    texts: list[str]
    model: str = "svm"


class ModelMetrics(BaseModel):
    model: str
    accuracy: float
    f1_macro: float
    f1_negative: float
    f1_neutral: float
    f1_positive: float
