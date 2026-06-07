"""Extended training pipeline — adds Random Forest & XGBoost to Sprint 3 optimisation.
Run this AFTER pipeline_train.py (reuses preprocessed data + TF-IDF).
"""

import json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))

from src.data.loader import load_sample
from src.data.preprocessor import preprocess_dataframe
from src.data.smote_utils import smote_info, plot_smote_comparison
from src.features.vectorizer import load_tfidf, transform
from src.models.ml_models import train_random_forest, train_xgboost, load_model
from src.evaluation.metrics import evaluate, plot_confusion_matrix, compare_models
from src.visualization.plots import plot_model_comparison

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)


def main():
    print("=" * 62)
    print("  Extended Pipeline — Random Forest + XGBoost")
    print("=" * 62)

    # ── Load & preprocess (reuse processed data) ──────────────────────────────────────
    print("\n[1/5] Loading & preprocessing 51k reviews...")
    t0 = time.time()
    df = load_sample()
    df = preprocess_dataframe(df)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Split ───────────────────────────────────────────────────────────────────────
    X_text = df["clean_text"]
    y = df["sentiment"]
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── TF-IDF (load existing vectorizer) ─────────────────────────────────────────────────────
    print("\n[2/5] Loading TF-IDF vectorizer...")
    vec = load_tfidf()
    X_train_tfidf = transform(vec, X_train_text)
    X_test_tfidf  = transform(vec, X_test_text)
    print(f"  Matrix: {X_train_tfidf.shape}")

    # ── SMOTE demonstration ─────────────────────────────────────────────────────────────────────────────────────────
    print("\n[3/5] SMOTE class balance analysis...")
    info_before = smote_info(y_train)
    print(f"  Before sampling: {info_before['before']}")
    print(f"  Already balanced: {info_before['balanced']}")
    y_imb = pd.concat([
        y_train[y_train == "negative"].iloc[:3000],
        y_train[y_train == "neutral"].iloc[:1000],
        y_train[y_train == "positive"].iloc[:5000],
    ])
    X_imb = X_train_tfidf[y_imb.index.map(lambda i: list(y_train.index).index(i))]

    from imblearn.over_sampling import SMOTE
    sm = SMOTE(k_neighbors=5, random_state=42)
    X_smote, y_smote = sm.fit_resample(X_imb, y_imb)
    info_after = smote_info(y_imb, y_smote)
    print(f"  After SMOTE:  {info_after['after']}")
    plot_smote_comparison(y_imb, y_smote,
                          save_path="reports/figures/smote_comparison.png")
    print("  SMOTE comparison chart saved.")

    # ── Random Forest ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    print("\n[4/5] Training Random Forest (200 trees, n_jobs=-1)...")
    t0 = time.time()
    from sklearn.decomposition import TruncatedSVD
    print("  Applying TruncatedSVD (300 components) for RF/XGB...")
    svd = TruncatedSVD(n_components=300, random_state=42)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_test_svd  = svd.transform(X_test_tfidf)
    import joblib
    joblib.dump(svd, "models/svd_transformer.joblib")
    print(f"  SVD done. Explained variance: {svd.explained_variance_ratio_.sum():.2%}")

    rf_model = train_random_forest(X_train_svd, y_train, n_estimators=200)
    y_pred_rf = rf_model.predict(X_test_svd)
    rf_result = evaluate(y_test, y_pred_rf, "Random Forest")
    plot_confusion_matrix(y_test, y_pred_rf, "Random Forest")
    print(f"  F1 Macro: {rf_result['f1_macro']:.4f}  |  "
          f"Accuracy: {rf_result['accuracy']:.4f}  ({time.time()-t0:.1f}s)")

    # ── XGBoost ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    print("\n[5/5] Training XGBoost (300 estimators, hist)...")
    t0 = time.time()
    xgb_model, le = train_xgboost(X_train_svd, y_train)

    y_test_enc = le.transform(y_test)
    y_pred_enc = xgb_model.predict(X_test_svd)
    y_pred_xgb_labels = le.inverse_transform(y_pred_enc)

    xgb_result = evaluate(y_test, y_pred_xgb_labels, "XGBoost")
    plot_confusion_matrix(y_test, y_pred_xgb_labels, "XGBoost")
    print(f"  F1 Macro: {xgb_result['f1_macro']:.4f}  |  "
          f"Accuracy: {xgb_result['accuracy']:.4f}  ({time.time()-t0:.1f}s)")

    # ── Full comparison ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    vader_result = {
        "model": "VADER Baseline", "accuracy": 0.4918, "f1_macro": 0.4056,
        "report": {"negative": {"f1-score": 0.45}, "neutral": {"f1-score": 0.20}, "positive": {"f1-score": 0.58}}
    }
    nb_model    = load_model("nb")
    svm_model   = load_model("svm")
    y_pred_nb   = nb_model.predict(X_test_tfidf)
    y_pred_svm  = svm_model.predict(X_test_tfidf)
    nb_result   = evaluate(y_test, y_pred_nb,  "Naive Bayes")
    svm_result  = evaluate(y_test, y_pred_svm, "LinearSVC")

    all_results = [vader_result, nb_result, svm_result, rf_result, xgb_result]
    comparison  = compare_models(all_results)

    print("\n-- Full Model Comparison --")
    print(comparison.to_string(index=False))
    plot_model_comparison(comparison)

    smote_data = {
        "description": "SMOTE demonstrated on deliberately imbalanced subset",
        "before": {k: int(v) for k, v in info_before["before"].items()},
        "after":  {k: int(v) for k, v in info_after["after"].items()},
        "k_neighbors": 5
    }

    # ── Save extended results ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    existing_path = "reports/evaluation_results.json"
    with open(existing_path, encoding="utf-8") as f:
        existing = json.load(f)

    existing["models"] = all_results
    existing["comparison"] = comparison.to_dict(orient="records")
    existing["smote"] = smote_data
    existing["svd_components"] = 300
    existing["svd_variance_explained"] = round(float(svd.explained_variance_ratio_.sum()), 4)

    with open(existing_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    print(f"\nExtended results saved to {existing_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
