"""FastAPI application — Sentiment Analysis API
Group 5: Mariem Tanabene, Abir Mhamdi, Nouha Laaroussi
"""

import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from api.schemas import PredictRequest, PredictResponse, BatchPredictRequest

# ── Lazy model cache ──────────────────────────────────────────────
_cache: dict = {}


def get_vec():
    if "vec" not in _cache:
        from src.features.vectorizer import load_tfidf
        _cache["vec"] = load_tfidf()
    return _cache["vec"]


def get_ml_model(name: str):
    if name not in _cache:
        from src.models.ml_models import load_model
        _cache[name] = load_model(name)
    return _cache[name]


def get_results() -> dict:
    if "results" not in _cache:
        p = ROOT / "reports" / "evaluation_results.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                _cache["results"] = json.load(f)
        else:
            _cache["results"] = {}
    return _cache["results"]


app = FastAPI(
    title="Sentiment Analysis API",
    description="NLP Sentiment Analysis — Yelp Reviews · Project 5 CRISP-DM + Agile",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = ROOT / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=(frontend_dir / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    models_loaded = [k for k in _cache if k in ("svm", "nb")]
    return {
        "status": "ok",
        "models_loaded": models_loaded,
        "tfidf_loaded": "vec" in _cache,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text cannot be empty.")

    model_name = req.model.lower()

    if model_name == "vader":
        from src.models.baseline import vader_predict, vader_scores
        sentiment = vader_predict(text)
        scores = vader_scores(text)
        return PredictResponse(
            text=text,
            sentiment=sentiment,
            confidence=round(abs(scores["compound"]), 4),
            scores=scores,
            model_used="VADER Baseline",
        )

    if model_name not in ("svm", "nb"):
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'. Use: svm, nb, vader.")

    from src.data.preprocessor import clean_text
    cleaned = clean_text(text) or text.lower()

    vec = get_vec()
    X = vec.transform([cleaned])
    model = get_ml_model(model_name)
    sentiment = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    scores_dict = {c: round(float(p), 4) for c, p in zip(classes, proba)}
    conf = float(max(proba))

    label_map = {"svm": "LinearSVC", "nb": "Naive Bayes"}
    return PredictResponse(
        text=text,
        sentiment=sentiment,
        confidence=round(conf, 4),
        scores=scores_dict,
        model_used=label_map[model_name],
    )


@app.post("/predict/batch")
async def predict_batch(req: BatchPredictRequest):
    if len(req.texts) > 100:
        raise HTTPException(status_code=400, detail="Batch size limit is 100.")
    results = []
    for text in req.texts:
        r = await predict(PredictRequest(text=text, model=req.model))
        results.append(r)
    return {"results": results, "count": len(results)}


@app.get("/metrics")
async def get_metrics():
    data = get_results()
    if not data:
        raise HTTPException(status_code=404, detail="No evaluation results found. Run pipeline_train.py.")
    return data


@app.get("/models")
async def list_models():
    return {
        "available": [
            {"id": "svm", "name": "LinearSVC", "description": "Linear SVC + TF-IDF (best F1)"},
            {"id": "nb",  "name": "Naive Bayes", "description": "MultinomialNB + TF-IDF (fast)"},
            {"id": "vader", "name": "VADER", "description": "Lexicon baseline, no training"},
        ],
        "recommended": "svm",
    }