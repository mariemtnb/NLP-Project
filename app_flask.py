"""
Flask + Waitress server — SentiYelp Sentiment Analysis
Group 5: Mariem Tanabene, Abir Mhamdi, Nouha Laaroussi
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from flask import Flask, request, jsonify, send_from_directory, Response

app = Flask(__name__, static_folder=str(ROOT / "frontend" / "static"), static_url_path="/static")

# Lazy model cache
_cache: dict = {}

def get_results():
    if "results" not in _cache:
        p = ROOT / "reports" / "evaluation_results.json"
        _cache["results"] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _cache["results"]


@app.route("/")
def index():
    return (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": list(k for k in _cache if k in ("svm","nb")), "tfidf_loaded": "vec" in _cache})


@app.route("/models")
def models():
    return jsonify({"available": [
        {"id": "svm",   "name": "LinearSVC",    "description": "Best F1 — TF-IDF + LinearSVC"},
        {"id": "nb",    "name": "Naive Bayes",  "description": "Fast — TF-IDF + MultinomialNB"},
        {"id": "vader", "name": "VADER",        "description": "Lexicon baseline, no training needed"},
    ], "recommended": "svm"})


@app.route("/metrics")
def metrics():
    data = get_results()
    if not data:
        return jsonify({"error": "No results found. Run pipeline_train.py first."}), 404
    return jsonify(data)


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    text = (body.get("text") or "").strip()
    model_name = (body.get("model") or "svm").lower()
    if not text:
        return jsonify({"error": "text cannot be empty"}), 422

    if model_name == "vader":
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        a = SentimentIntensityAnalyzer()
        scores = a.polarity_scores(text)
        c = scores["compound"]
        sentiment = "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
        return jsonify({"text": text, "sentiment": sentiment, "confidence": round(abs(c), 4), "scores": scores, "model_used": "VADER Baseline"})

    if model_name not in ("svm", "nb"):
        return jsonify({"error": f"Unknown model '{model_name}'"}), 400

    # Lazy-load ML models
    if "vec" not in _cache:
        from src.features.vectorizer import load_tfidf
        _cache["vec"] = load_tfidf()
    if model_name not in _cache:
        from src.models.ml_models import load_model
        _cache[model_name] = load_model(model_name)

    from src.data.preprocessor import clean_text
    cleaned = clean_text(text) or text.lower()
    X = _cache["vec"].transform([cleaned])
    model = _cache[model_name]
    sentiment = model.predict(X)[0]
    proba = list(model.predict_proba(X)[0])
    classes = list(model.classes_)
    scores_dict = {c: round(float(p), 4) for c, p in zip(classes, proba)}
    conf = float(max(proba))
    label_map = {"svm": "LinearSVC", "nb": "Naive Bayes"}
    return jsonify({"text": text, "sentiment": sentiment, "confidence": round(conf, 4), "scores": scores_dict, "model_used": label_map[model_name]})


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    body = request.get_json(force=True)
    texts = (body.get("texts") or [])[:100]
    model = body.get("model", "svm")
    results = []
    for t in texts:
        with app.test_request_context():
            pass
        import flask
        with app.test_request_context(json={"text": t, "model": model}):
            r = predict()
            if hasattr(r, "get_json"):
                results.append(r.get_json())
            else:
                results.append(r[0].get_json())
    return jsonify({"results": results, "count": len(results)})


if __name__ == "__main__":
    try:
        from waitress import serve
        print("Starting SentiYelp on http://0.0.0.0:8000")
        serve(app, host="0.0.0.0", port=8000, threads=4)
    except ImportError:
        app.run(host="0.0.0.0", port=8000, debug=False)
