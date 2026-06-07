"""Visualization helpers: wordclouds, temporal trends, sentiment distributions."""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

PALETTE = {"negative": "#e74c3c", "neutral": "#f39c12", "positive": "#2ecc71"}


def plot_sentiment_distribution(df: pd.DataFrame, save: bool = True) -> str:
    counts = df["sentiment"].value_counts().reindex(["negative", "neutral", "positive"])
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[s] for s in counts.index], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200, f"{val:,}", ha="center", fontsize=11)
    ax.set_title("Sentiment Distribution (50k Sample)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Sentiment")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "sentiment_distribution.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_wordcloud(df: pd.DataFrame, sentiment: str, save: bool = True) -> str:
    texts = " ".join(df[df["sentiment"] == sentiment]["clean_text"].dropna().tolist())
    color = {"negative": "Reds", "neutral": "YlOrBr", "positive": "Greens"}[sentiment]
    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap=color,
        max_words=100,
        collocations=False,
    ).generate(texts)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Most Frequent Words — {sentiment.capitalize()} Reviews", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"wordcloud_{sentiment}.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_text_length_distribution(df: pd.DataFrame, save: bool = True) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    for sentiment, color in PALETTE.items():
        subset = df[df["sentiment"] == sentiment]["text_length"]
        ax.hist(subset.clip(upper=500), bins=50, alpha=0.55, color=color, label=sentiment.capitalize())
    ax.set_title("Review Length Distribution by Sentiment", fontsize=14, fontweight="bold")
    ax.set_xlabel("Word Count (capped at 500)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "text_length_distribution.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_temporal_sentiment(df: pd.DataFrame, save: bool = True) -> str:
    monthly = (
        df.groupby(["year", "month", "sentiment"])
        .size()
        .reset_index(name="count")
    )
    monthly["period"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    for sentiment, color in PALETTE.items():
        subset = monthly[monthly["sentiment"] == sentiment].sort_values("period")
        ax.plot(subset["period"], subset["count"], color=color, label=sentiment.capitalize(), linewidth=2)
    ax.set_title("Sentiment Trend Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Period")
    ax.set_ylabel("Number of Reviews")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "temporal_sentiment.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(comparison_df: pd.DataFrame, save: bool = True) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comparison_df))
    w = 0.35
    ax.bar(x - w / 2, comparison_df["Accuracy"], w, label="Accuracy", color="#3498db", edgecolor="white")
    ax.bar(x + w / 2, comparison_df["F1 Macro"], w, label="F1 Macro", color="#9b59b6", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["Model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Accuracy & F1 Macro", fontsize=14, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "model_comparison.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_burndown(sprints: list[dict], save: bool = True) -> str:
    """Generate sprint burn-down chart from sprint data."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#3498db", "#e67e22", "#2ecc71"]
    for i, sprint in enumerate(sprints):
        days = sprint["days"]
        ideal = sprint["ideal"]
        actual = sprint["actual"]
        ax.plot(days, ideal, "--", color=colors[i], alpha=0.6, label=f"{sprint['name']} (ideal)")
        ax.plot(days, actual, "-o", color=colors[i], linewidth=2, markersize=5, label=f"{sprint['name']} (actual)")
    ax.set_xlabel("Day of Sprint")
    ax.set_ylabel("Story Points Remaining")
    ax.set_title("Burn-Down Chart — All Sprints", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "burndown_chart.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
