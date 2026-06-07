"""
Full training pipeline:
  1. Load 50k sample
  2. Preprocess text
  3. VADER baseline
  4. TF-IDF + SVM + Naive Bayes
  5. Save models & evaluation results
"""

import json
import os
import sys
import time

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from src.data.loader import load_sample
from src.data.preprocessor import preprocess_dataframe
from src.features.vectorizer import build_tfidf, transform
from src.models.baseline import vader_predict_batch
from src.models.ml_models import train_svm, train_naive_bayes
from src.evaluation.metrics import evaluate, plot_confusion_matrix, compare_models
from src.visualization.plots import (
    plot_sentiment_distribution,
    plot_wordcloud,
    plot_text_length_distribution,
    plot_temporal_sentiment,
    plot_model_comparison,
    plot_burndown,
)

RESULTS_PATH = os.path.join("reports", "evaluation_results.json")
os.makedirs("reports/figures", exist_ok=True)


def main():
    print("=" * 60)
    print("  NLP Sentiment Analysis — Training Pipeline")
    print("  Group: Mariem Tanabene, Abir Mhamdi, Nouha Laaroussi")
    print("=" * 60)

    # ── 1. Load data ────────────────────────
    print("\n[1/7] Loading dataset...")
    t0 = time.time()
    df = load_sample()
    print(f"  Loaded {len(df):,} reviews in {time.time()-t0:.1f}s")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Sentiment counts:\n{df['sentiment'].value_counts()}")

    # ── 2. Preprocess ───────────────────────────
    print("\n[2/7] Preprocessing text...")
    t0 = time.time()
    df = preprocess_dataframe(df, text_col="text")
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── 3. EDA Visualisations ──────────────────────────
    print("\n[3/7] Generating EDA visualisations...")
    plot_sentiment_distribution(df)
    plot_text_length_distribution(df)
    plot_temporal_sentiment(df)
    for sentiment in ["negative", "neutral", "positive"]:
        print(f"  Wordcloud: {sentiment}...")
        plot_wordcloud(df, sentiment)
    print("  Figures saved to reports/figures/")

    # ── 4. Train/test split ────────────────────────────
    print("\n[4/7] Splitting dataset (80/20)...")
    X_text = df["clean_text"]
    y = df["sentiment"]
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train_text):,} | Test: {len(X_test_text):,}")

    # ── 5. VADER baseline ──────────────────────────────
    print("\n[5/7] Running VADER baseline...")
    t0 = time.time()
    X_test_raw = df.loc[X_test_text.index, "text"]
    y_pred_vader = vader_predict_batch(X_test_raw)
    vader_result = evaluate(y_test, y_pred_vader, "VADER Baseline")
    plot_confusion_matrix(y_test, y_pred_vader, "VADER Baseline")
    print(f"  F1 Macro: {vader_result['f1_macro']:.4f}  |  Accuracy: {vader_result['accuracy']:.4f}  ({time.time()-t0:.1f}s)")

    # ── 6. TF-IDF vectorisation ──────────────────────────────
    print("\n[6/7] Vectorising with TF-IDF (max 50k features, 1-2grams)...")
    t0 = time.time()
    vec, X_train_tfidf = build_tfidf(X_train_text)
    X_test_tfidf = transform(vec, X_test_text)
    print(f"  Matrix shape: {X_train_tfidf.shape}  ({time.time()-t0:.1f}s)")

    # ── 7. ML models ───────────────────────────────────
    print("\n[7/7] Training ML models...")

    print("  Training Naive Bayes...")
    t0 = time.time()
    nb_model = train_naive_bayes(X_train_tfidf, y_train)
    y_pred_nb = nb_model.predict(X_test_tfidf)
    nb_result = evaluate(y_test, y_pred_nb, "Naive Bayes")
    plot_confusion_matrix(y_test, y_pred_nb, "Naive Bayes")
    print(f"  F1 Macro: {nb_result['f1_macro']:.4f}  |  Accuracy: {nb_result['accuracy']:.4f}  ({time.time()-t0:.1f}s)")

    print("  Training LinearSVC (calibrated)...")
    t0 = time.time()
    svm_model = train_svm(X_train_tfidf, y_train)
    y_pred_svm = svm_model.predict(X_test_tfidf)
    svm_result = evaluate(y_test, y_pred_svm, "LinearSVC")
    plot_confusion_matrix(y_test, y_pred_svm, "LinearSVC")
    print(f"  F1 Macro: {svm_result['f1_macro']:.4f}  |  Accuracy: {svm_result['accuracy']:.4f}  ({time.time()-t0:.1f}s)")

    # ── Comparison ────────────────────────────────────
    results = [vader_result, nb_result, svm_result]
    comparison = compare_models(results)
    print("\n-- Model Comparison --")
    print(comparison.to_string(index=False))
    plot_model_comparison(comparison)

    # ── Burn-down chart ───────────────────────────────────
    sprints = [
        {
            "name": "Sprint 1",
            "days": list(range(0, 15)),
            "ideal": [40 - i * (40 / 14) for i in range(15)],
            "actual": [40, 38, 36, 33, 30, 28, 25, 22, 20, 17, 14, 11, 8, 4, 0],
        },
        {
            "name": "Sprint 2",
            "days": list(range(0, 15)),
            "ideal": [50 - i * (50 / 14) for i in range(15)],
            "actual": [50, 48, 45, 42, 39, 36, 33, 29, 26, 22, 18, 13, 9, 4, 0],
        },
        {
            "name": "Sprint 3",
            "days": list(range(0, 15)),
            "ideal": [45 - i * (45 / 14) for i in range(15)],
            "actual": [45, 44, 42, 40, 37, 34, 30, 27, 23, 19, 15, 11, 7, 3, 0],
        },
    ]
    plot_burndown(sprints)

    # ── Save results ────────────────────────────────────────
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "models": results,
                "comparison": comparison.to_dict(orient="records"),
                "sprints": sprints,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {RESULTS_PATH}")
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
