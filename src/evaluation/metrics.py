"""Evaluation utilities: metrics, confusion matrix, classification report."""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

LABELS = ["negative", "neutral", "positive"]


def evaluate(y_true, y_pred, model_name: str = "model") -> dict:
    """Compute and return evaluation metrics dict."""
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    result = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "report": report,
    }
    return result


def plot_confusion_matrix(y_true, y_pred, model_name: str = "model", save: bool = True) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS, ax=ax,
        linewidths=0.5
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"confusion_{model_name.lower().replace(' ', '_')}.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from a list of evaluate() results."""
    rows = []
    for r in results:
        rows.append({
            "Model": r["model"],
            "Accuracy": r["accuracy"],
            "F1 Macro": r["f1_macro"],
            "F1 Negative": round(r["report"].get("negative", {}).get("f1-score", 0), 4),
            "F1 Neutral": round(r["report"].get("neutral", {}).get("f1-score", 0), 4),
            "F1 Positive": round(r["report"].get("positive", {}).get("f1-score", 0), 4),
        })
    return pd.DataFrame(rows).sort_values("F1 Macro", ascending=False)
