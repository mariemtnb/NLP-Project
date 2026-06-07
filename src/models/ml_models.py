"""Machine Learning models: LinearSVC, Naive Bayes, Random Forest, XGBoost."""

import os
import joblib
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_svm(X_train, y_train, C: float = 1.0) -> CalibratedClassifierCV:
    svc = LinearSVC(C=C, max_iter=2000, class_weight="balanced")
    model = CalibratedClassifierCV(svc, cv=3)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODELS_DIR, "svm_model.joblib"))
    return model


def train_naive_bayes(X_train, y_train, alpha: float = 1.0) -> MultinomialNB:
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODELS_DIR, "nb_model.joblib"))
    return model


def train_random_forest(X_train, y_train, n_estimators: int = 200, max_depth: int = None) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODELS_DIR, "rf_model.joblib"))
    return model


def train_xgboost(X_train, y_train):
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    joblib.dump(le, os.path.join(MODELS_DIR, "xgb_label_encoder.joblib"))

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_enc, verbose=False)
    joblib.dump(model, os.path.join(MODELS_DIR, "xgb_model.joblib"))
    return model, le


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}. Run pipeline_train.py first.")
    return joblib.load(path)
