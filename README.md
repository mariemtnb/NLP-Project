# SentiYelp — Analyse des Sentiments NLP
### Projet 5 · Data Science Project Management · CRISP-DM + Agile

**Groupe :** Mariem Tanabene · Abir Mhamdi · Nouha Laaroussi  
**Dataset :** [Yelp Review Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) — 6,9 millions d'avis

---

## Objectif

Classifier automatiquement les avis Yelp en **négatif / neutre / positif** pour alerter les équipes Qualité et Marketing d'une enseigne e-commerce. Projet conduit en méthodologie **CRISP-DM** avec **3 sprints Agile**.

---

## Résultats Clés

| Modèle | F1 Macro | Accuracy |
|---|---|---|
| **LinearSVC ★** | **0.743** | **74.45%** |
| Naive Bayes | 0.725 | 72.21% |
| VADER Baseline | 0.406 | 49.18% |

---

## Structure du Projet

```
ds-project-management/
├── data/
│   ├── raw/                          # Fichiers Yelp JSON extraits
│   └── processed/sample_50k.json    # 51,000 avis équilibrés
├── src/
│   ├── data/
│   │   ├── loader.py                 # Chargement du dataset
│   │   └── preprocessor.py          # Pipeline de nettoyage NLP
│   ├── features/
│   │   └── vectorizer.py            # TF-IDF (50k features, bigrammes)
│   ├── models/
│   │   ├── baseline.py              # VADER lexical baseline
│   │   └── ml_models.py             # LinearSVC + Naive Bayes
│   ├── evaluation/
│   │   └── metrics.py               # F1, accuracy, confusion matrix
│   └── visualization/
│       └── plots.py                  # Wordclouds, tendances, burn-down
├── api/
│   ├── main.py                       # FastAPI (uvicorn)
│   └── main_simple.py               # Flask + Waitress (production)
├── frontend/
│   ├── index.html                    # Dashboard SentiYelp
│   └── static/
│       ├── style.css
│       └── app.js
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Sprint 1 — EDA + VADER
│   ├── 02_modeling.ipynb            # Sprint 2 — TF-IDF + SVM + NB
│   └── 03_evaluation_deployment.ipynb  # Sprint 3 — Eval + Deploy
├── reports/
│   ├── CRISP_DM_Report.md / .docx   # Rapport CRISP-DM (markdown + Word illustré)
│   ├── Retrospective_Report.md / .docx  # Rapport de rétrospective Agile
│   ├── Rapport_PFE_SentiYelp_Groupe5.docx  # Rapport PFE illustré (24 figures)
│   ├── evaluation_results.json      # Métriques JSON
│   └── figures/                     # Graphiques générés
├── tableau/
│   ├── data/                        # 9 extraits CSV pour Tableau
│   ├── Tableau_Dashboard_Guide.md / .docx  # Guide de construction du tableau de bord
│   └── README.md                    # Démarrage rapide Tableau
├── diagrams/
│   ├── crisp_dm_cycle.png           # Cycle CRISP-DM
│   ├── system_architecture.png      # Architecture système
│   └── pipeline_flowchart.png       # Flux de traitement
├── trello/
│   └── sprint_export.json           # Export Trello (3 sprints)
├── models/                          # Modèles .joblib sauvegardés
├── app_flask.py                     # Point d'entrée serveur
├── pipeline_train.py                # Pipeline d'entraînement complet
├── generate_diagrams.py             # Génération des diagrammes
├── generate_pfe_report.js           # Génère le rapport PFE illustré (.docx)
├── generate_crispdm_report.js       # Convertit n'importe quel rapport .md → .docx
├── generate_tableau_data.py         # Génère les 9 extraits CSV pour Tableau
└── requirements.txt
```

---

## Démarrage Rapide

### 1. Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # optionnel
```

### 2. Entraînement (si modèles non existants)

```bash
python pipeline_train.py
```

Ce script :
- Charge le dataset (51,000 avis)
- Prétraite les textes (NLTK)
- Entraîne VADER, Naive Bayes et LinearSVC
- Génère toutes les figures dans `reports/figures/`
- Sauvegarde les modèles dans `models/`

### 3. Lancer l'API + Dashboard

```bash
python app_flask.py
```

Accéder à : [http://localhost:8000](http://localhost:8000)

---

## API REST

| Endpoint | Méthode | Corps | Description |
|---|---|---|---|
| `/` | GET | — | Dashboard web |
| `/health` | GET | — | Statut API |
| `/predict` | POST | `{"text": "...", "model": "svm"}` | Prédiction unitaire |
| `/predict/batch` | POST | `{"texts": [...], "model": "nb"}` | Batch (max 100) |
| `/metrics` | GET | — | Résultats évaluation |
| `/models` | GET | — | Modèles disponibles |

**Modèles :** `svm` (recommandé), `nb`, `vader`

---

## Sprints & Agile

| Sprint | Semaines | Livrable principal | Vélocité |
|---|---|---|---|
| Sprint 0 | S1 | DoD + 12 User Stories (MoSCoW) | 6 pts |
| Sprint 1 | S1-S2 | EDA + VADER baseline (F1=0.406) | 11 pts |
| Sprint 2 | S3-S4 | SVM + NB (F1=0.743) | 16 pts |
| Sprint 3 | S5-S7 | API + Dashboard + Rapport | 18 pts |

---

## Livrables

- ✅ Dépôt Git structuré avec README
- ✅ Rapport CRISP-DM (`reports/CRISP_DM_Report.md`)
- ✅ 3 Notebooks Jupyter annotés
- ✅ API REST Flask + Dashboard web SentiYelp
- ✅ Export Trello 3 sprints + burn-down charts
- ✅ Diagrammes : CRISP-DM, architecture, pipeline
- ✅ Figures EDA : wordclouds, distribution, temporal, confusion matrices

---

## Équipe

| Membre | Rôle |
|---|---|
| **Mariem Tanabene** | Data Scientist — modélisation, feature engineering, évaluation |
| **Abir Mhamdi** | Data Analyst / BI Developer — EDA, visualisations, dashboard |
| **Nouha Laaroussi** | Project Manager / Data Engineer — Trello, pipeline, rapport |
