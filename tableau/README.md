# Tableau de Bord SentiYelp — Dossier Tableau

Ce dossier contient tout le nécessaire pour construire et publier le **tableau de bord des sentiments** du Projet 5 dans **Tableau Public** (gratuit).

## Contenu

```
tableau/
├── data/                          # 9 extraits CSV (sources du tableau de bord)
│   ├── model_comparison.csv
│   ├── model_f1_by_class.csv
│   ├── sentiment_distribution.csv
│   ├── temporal_sentiment.csv
│   ├── category_sentiment.csv
│   ├── word_frequencies.csv
│   ├── sprint_burndown.csv
│   ├── sprint_velocity.csv
│   └── reviews_sample.csv
├── Tableau_Dashboard_Guide.md     # Guide de construction pas à pas (5 feuilles + dashboard)
├── Tableau_Dashboard_Guide.docx   # Même guide, version Word
├── sprint_export.json (../trello)  # rappel : backlog/sprints Trello
└── README.md
```

## Démarrage rapide

1. **Régénérer les données** (optionnel, déjà fournies) :
   ```bash
   python generate_tableau_data.py
   ```
2. **Ouvrir Tableau Public Desktop** → *Connect → Text File* → choisir les CSV de `data/`.
3. **Suivre le guide** `Tableau_Dashboard_Guide.md` : il décrit, feuille par feuille, les champs, les types de repères, les couleurs et les filtres.
4. **Assembler** les 5 feuilles en un tableau de bord, ajouter les filtres globaux `Sentiment` et `Year`.
5. **Publier** sur Tableau Public et coller le lien dans le rapport CRISP-DM.

## Palette de couleurs (réutiliser partout)

| Sentiment | Couleur | Hex |
|---|---|---|
| Négatif | rouge | `#FF4D6D` |
| Neutre | orange | `#FFA726` |
| Positif | vert | `#4CAF8A` |

## Les 5 feuilles du tableau de bord

1. **Vue exécutive (KPIs)** — meilleurs F1/accuracy, nombre d'avis, nombre de modèles.
2. **Évolution temporelle** — volume d'avis par sentiment et par année.
3. **Répartition par catégorie** — top catégories de business par sentiment.
4. **Nuage de mots par sentiment** — vocabulaire discriminant de chaque classe.
5. **Comparaison des modéles** — F1 par classe et par modèle.

> Pourquoi pas de fichier `.twb` livré ? Un classeur Tableau n'est pas fiablement créable en dehors de Tableau ; ce paquet (données + guide détaillé) permet de le reconstruire à l'identique en ~20–30 min, puis de le publier sur Tableau Public.
