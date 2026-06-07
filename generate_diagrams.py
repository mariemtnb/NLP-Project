"""Generate project diagrams: CRISP-DM workflow, architecture, pipeline."""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

os.makedirs("diagrams", exist_ok=True)


# ── 1. CRISP-DM Cycle ──────────────────────────────────────────────
def crisp_dm_diagram():
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    phases = [
        ("Compréhension\nMétier", "#6c63ff", "S1-S2"),
        ("Compréhension\nDonnées", "#3498db", "S1-S2"),
        ("Préparation\nDonnées", "#e67e22", "S3-S4"),
        ("Modélisation", "#e74c3c", "S3-S4"),
        ("Évaluation", "#2ecc71", "S5-S6"),
        ("Déploiement", "#00d4aa", "S7"),
    ]

    n = len(phases)
    r_outer, r_inner = 1.1, 0.45
    angles = [2 * np.pi * i / n - np.pi / 2 for i in range(n)]

    for i, (label, color, sprint) in enumerate(phases):
        a1, a2 = angles[i], angles[(i + 1) % n]
        theta = np.linspace(a1, a2, 60)

        xs_outer = np.cos(theta) * r_outer
        ys_outer = np.sin(theta) * r_outer
        xs_inner = np.cos(theta[::-1]) * r_inner
        ys_inner = np.sin(theta[::-1]) * r_inner

        xs = np.concatenate([xs_outer, xs_inner])
        ys = np.concatenate([ys_outer, ys_inner])
        ax.fill(xs, ys, color=color, alpha=0.85, zorder=2)
        ax.plot(xs_outer, ys_outer, color="#0f1117", lw=2, zorder=3)

        mid_angle = (a1 + a2) / 2
        rx, ry = np.cos(mid_angle) * 0.8, np.sin(mid_angle) * 0.8
        ax.text(rx, ry, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white",
                multialignment="center", zorder=4)

        sx, sy = np.cos(mid_angle) * 1.32, np.sin(mid_angle) * 1.32
        ax.text(sx, sy, sprint, ha="center", va="center",
                fontsize=8, color=color, fontweight="600", zorder=4)

    circle = plt.Circle((0, 0), r_inner, color="#1a1d27", zorder=5)
    ax.add_patch(circle)
    ax.text(0, 0.06, "CRISP-DM", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white", zorder=6)
    ax.text(0, -0.1, "Yelp NLP", ha="center", va="center",
            fontsize=9, color="#8890b0", zorder=6)
    ax.text(0, -0.28, "Project 5", ha="center", va="center",
            fontsize=8, color="#6c63ff", fontweight="600", zorder=6)

    ax.set_title("Méthodologie CRISP-DM — Projet 5 NLP", color="white",
                 fontsize=14, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig("diagrams/crisp_dm_cycle.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  crisp_dm_cycle.png")


def architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)

    def box(x, y, w, h, color, label, sublabel="", alpha=0.9):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="white",
                                        linewidth=1.2, alpha=alpha, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.15 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=4)
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.22, sublabel,
                    ha="center", va="center", fontsize=7.5, color="#ccc", zorder=4)

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#8890b0", lw=1.5), zorder=2)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.15, label, ha="center", va="center",
                    fontsize=7, color="#8890b0")

    box(0.3, 5.8, 2.2, 1.2, "#2c3e50", "Yelp Dataset", "6.9M reviews")
    box(0.3, 4.2, 2.2, 1.2, "#2c3e50", "Kaggle", "yelp-dataset")
    box(3.2, 5.2, 2.2, 1.2, "#6c63ff", "Sampler", "50k balanced")
    box(3.2, 3.4, 2.2, 1.2, "#6c63ff", "Preprocessor", "NLTK / spaCy")
    box(3.2, 1.6, 2.2, 1.2, "#6c63ff", "TF-IDF", "50k features")
    box(6.4, 5.8, 2.0, 1.0, "#e74c3c", "VADER", "Baseline")
    box(6.4, 4.4, 2.0, 1.0, "#e67e22", "Naive Bayes", "F1: 0.725")
    box(6.4, 3.0, 2.0, 1.0, "#2ecc71", "LinearSVC ★", "F1: 0.743")
    box(9.5, 4.0, 2.2, 1.8, "#3498db", "Flask API", "Waitress WSGI\n/predict /metrics")
    box(12.3, 5.0, 1.5, 1.0, "#00d4aa", "Dashboard", "Chart.js")
    box(12.3, 3.5, 1.5, 1.0, "#00d4aa", "Predict UI", "HTML/JS/CSS")

    arrow(2.5, 6.4, 3.2, 5.8, "extract")
    arrow(2.5, 4.8, 3.2, 4.2)
    arrow(4.3, 5.2, 4.3, 4.6)
    arrow(4.3, 3.4, 4.3, 2.8)
    arrow(5.4, 2.2, 6.4, 6.3, "transform")
    arrow(5.4, 2.2, 6.4, 4.9)
    arrow(5.4, 2.2, 6.4, 3.5)
    arrow(8.4, 6.3, 9.5, 5.0, "predict")
    arrow(8.4, 4.9, 9.5, 4.9)
    arrow(8.4, 3.5, 9.5, 4.6)
    arrow(11.7, 4.9, 12.3, 5.5)
    arrow(11.7, 4.5, 12.3, 4.0)

    ax.text(7.0, 0.6, "Models saved as .joblib  \u00b7  TF-IDF vectorizer saved  \u00b7  Evaluation JSON exported",
            ha="center", fontsize=8, color="#8890b0")
    ax.set_title("Architecture Système — SentiYelp NLP Pipeline", color="white",
                 fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig("diagrams/system_architecture.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  system_architecture.png")


def pipeline_flowchart():
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)

    steps = [
        (6, 8.2, "#6c63ff", "1. Chargement Dataset", "51,000 avis Yelp (JSON)"),
        (6, 6.8, "#3498db", "2. Nettoyage & Prétraitement", "HTML, URLs, stop-words, lemmatisation"),
        (6, 5.4, "#e67e22", "3. Vectorisation TF-IDF", "Unigrammes + Bigrammes, max 50k features"),
        (2.5, 3.8, "#e74c3c", "4a. Baseline VADER", "F1 Macro: 0.406"),
        (6, 3.8, "#f39c12", "4b. Naive Bayes", "F1 Macro: 0.725"),
        (9.5, 3.8, "#2ecc71", "4c. LinearSVC ★", "F1 Macro: 0.743"),
        (6, 2.2, "#9b59b6", "5. Évaluation & Comparaison", "Accuracy, F1-macro, Matrice confusion"),
        (6, 0.8, "#00d4aa", "6. Déploiement", "Flask API + Dashboard Frontend"),
    ]

    for x, y, color, title, desc in steps:
        rect = mpatches.FancyBboxPatch((x - 2.2, y - 0.45), 4.4, 0.9,
                                        boxstyle="round,pad=0.08",
                                        facecolor=color, alpha=0.85,
                                        edgecolor="white", lw=1, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y + 0.12, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white", zorder=4)
        ax.text(x, y - 0.15, desc, ha="center", va="center",
                fontsize=8, color="#ddd", zorder=4)

    for i in range(3):
        _, y_from, *_ = steps[i]
        _, y_to, *_ = steps[i + 1]
        ax.annotate("", xy=(6, y_to + 0.45), xytext=(6, y_from - 0.45),
                    arrowprops=dict(arrowstyle="->", color="#8890b0", lw=1.8), zorder=2)

    for step_item in [steps[3], steps[4], steps[5]]:
        mx, my = step_item[0], step_item[1]
        ax.annotate("", xy=(mx, my + 0.45), xytext=(6, steps[2][1] - 0.45),
                    arrowprops=dict(arrowstyle="->", color="#8890b0", lw=1.5,
                                    connectionstyle="arc3,rad=0"), zorder=2)

    for step_item in [steps[3], steps[4], steps[5]]:
        mx, my = step_item[0], step_item[1]
        ax.annotate("", xy=(6, steps[6][1] + 0.45), xytext=(mx, my - 0.45),
                    arrowprops=dict(arrowstyle="->", color="#8890b0", lw=1.5), zorder=2)

    ax.annotate("", xy=(6, steps[7][1] + 0.45), xytext=(6, steps[6][1] - 0.45),
                arrowprops=dict(arrowstyle="->", color="#8890b0", lw=1.8), zorder=2)

    ax.set_title("Pipeline NLP — Flux de Traitement Complet", color="white",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("diagrams/pipeline_flowchart.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  pipeline_flowchart.png")


if __name__ == "__main__":
    print("Generating diagrams...")
    crisp_dm_diagram()
    architecture_diagram()
    pipeline_flowchart()
    print("All diagrams saved to diagrams/")
