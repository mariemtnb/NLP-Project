# Rapport CRISP-DM — Projet 5
## Analyse des Sentiments et Pilotage Agile d'un Projet NLP

**Groupe :** Mariem Tanabene · Abir Mhamdi · Nouha Laaroussi  
**Module :** Data Science Project Management  
**Date :** Juin 2026  
**Dataset :** Yelp Review Dataset (Kaggle, 6,9 millions d'avis)

---

## Table des matières

1. [Phase 1 — Compréhension Métier](#1-compréhension-métier)
2. [Phase 2 — Compréhension des Données](#2-compréhension-des-données)
3. [Phase 3 — Préparation des Données](#3-préparation-des-données)
4. [Phase 4 — Modélisation](#4-modélisation)
5. [Phase 5 — Évaluation](#5-évaluation)
6. [Phase 6 — Déploiement](#6-déploiement)
7. [Conduite Agile du Projet](#7-conduite-agile-du-projet)
8. [Conclusion et Leçons Apprises](#8-conclusion-et-leçons-apprises)

---

## 1. Compréhension Métier

### 1.1 Contexte

Une enseigne de e-commerce reçoit des milliers d'avis clients chaque jour. Lire manuellement ces avis est impossible ; une solution automatisée de classification des sentiments permettrait d'identifier rapidement les catégories de produits générant de l'insatisfaction et d'alerter les équipes Qualité et Marketing.

La conduite du projet s'appuie sur la méthodologie CRISP-DM, dont les six phases itératives sont illustrées dans la figure ci-dessous.

### 1.2 Objectifs Métier

Les objectifs métier du projet et leurs critères de succès associés sont résumés dans le tableau ci-dessous.

| Objectif | Critère de succès |
|---|---|
| Classifier automatiquement les avis | F1-macro ≥ 0,70 sur jeu de test |
| Identifier les catégories problématiques | Dashboard Tableau opérationnel |
| Déployer une solution accessible | API REST + interface web |
| Démontrer la maîtrise agile | 3 sprints avec burn-down charts |

### 1.3 Définition du Périmètre

- **Entrée :** Texte libre d'un avis client (en anglais)
- **Sortie :** Classe de sentiment {négatif, neutre, positif} + score de confiance
- **Contrainte :** Modèle léger (pas de GPU requis pour l'inférence)
- **Hors-périmètre :** Analyse des avis en français, modèles >1 GB

### 1.4 Plan de Projet — Sprints

Le plan de projet se décline en quatre sprints, dont les objectifs et outils sont détaillés dans le tableau ci-dessous.

| Sprint | Semaines | Objectifs clés | Outils |
|---|---|---|---|
| Sprint 0 — Cadrage | S1 | DoD, User Stories, rituels | Trello, CRISP-DM |
| Sprint 1 — Data & Baseline | S1–S2 | EDA, VADER (F1>0.40) | Jupyter, pandas, VADER |
| Sprint 2 — Modèles ML | S3–S4 | TF-IDF, SVM, NB (F1>0.70) | scikit-learn, spaCy |
| Sprint 3 — Optimisation & Deploy | S5–S7 | GridSearch, API, Dashboard | Flask, Tableau, Git |

---

## 2. Compréhension des Données

### 2.1 Source de Données

**Yelp Review Dataset** — disponible sur [Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)

| Fichier | Taille | Description |
|---|---|---|
| `yelp_academic_dataset_review.json` | 5,1 Go | 6,9 M avis (texte, étoiles, date) |
| `yelp_academic_dataset_business.json` | 120 Mo | Métadonnées des commerces |

### 2.2 Structure des Données

Chaque avis contient :
- `review_id` — identifiant unique
- `business_id` — lien vers le commerce
- `stars` — note de 1 à 5
- `text` — texte libre de l'avis
- `date` — date de publication
- `useful` / `funny` / `cool` — votes des autres utilisateurs

### 2.3 Étiquetage des Sentiments

La règle de transformation des notes en étoiles vers les trois classes de sentiment est présentée dans le tableau ci-dessous.

| Note | Classe | Justification |
|---|---|---|
| 1–2 étoiles | négatif | Insatisfaction manifeste |
| 3 étoiles | neutre | Avis mitigé |
| 4–5 étoiles | positif | Satisfaction exprimée |

### 2.4 Échantillonnage

Pour garantir l'équilibre des classes et la faisabilité computationnelle :

- **50 000 avis** échantillonnés au total
- **Stratégie :** échantillonnage strati fié asynthe démontrée avec SMOTE dans le pipeline étendu
- **Dossier :** `data/processed/sample_50k.json`

---

## 3. Préparation des Données

### 3.1 Nettoyage

Les avis bruts contiennent :
- Balisage HTML / URLs (ex. `<br/>`, `http://...`)
- Accentuation américaine (ex. `color` vs `colour`)
- Émojis et caractères de contrôle
- Textes dupliqués ou de spam

**Pipeline appliqué :**
1. Suppression HTML `re.sub(r'<[^>]+>', '', text)`
2. URLs `re.sub(r'http[s]?://\\S+', '', text)`
3. Normalisation des accents (Unidecode)
4. Minusculisation
5. Suppression de la ponctuation (except `'` pour les contractions)

### 3.2 Tokenization et Lemmatisation

Utilisés les bibliothèques :
- **NLTK** `word_tokenize`, `stopwords`
- **spaCy** (optionnel) pour une lemmatisation plus robuste

Stop-words : ensemble standard anglais de 179 mots (`the`, `a`, `is`, etc.).

### 3.3 Sorties Intermédiaires

| Fichier | Description |
|---|---|
| `data/processed/sample_50k.json` | Avis échantillonnés et étiquetés (JSONL) |
| `data/processed/sample_50k_clean.json` | Avis nettoyés et prétraités |

---

## 4. Modélisation

### 4.1 Baseline : VADER (Valence Aware Dictionary and sEntiment Reasoner)

**Approche :** Lexicon-based (pas d'apprentissage).

**Résultats :**
- Accuracy : 49,2 %
- F1-macro : 40,6 %
- F1-Négatif : 45 %, F1-Neutre : 20 %, F1-Positif : 58 %

**Interprétation :** VADER performe mieux sur les sentiments polaires (Négatif/Positif) mais peine sur la classe Neutre. Bon point de départ pour évaluer le problème.

### 4.2 Extraction de Caractéristiques : TF-IDF

**Configuration :**
- **Vectorizer :** `TfidfVectorizer` de scikit-learn
- **Gram :** unigrammes + bigrammes
- **Max features :** 50 000 (paramètre de régularisation)
- **Min/Max DF :** termes dans au moins 2 docs, max 95 % des docs
- **Sublinear TF :** appliqué

**Matrice résultante :** 40k avis × 50k features (sparse, ~7 % densité).

### 4.3 Classification : Naive Bayes et SVM

#### Naive Bayes (MultinomialNB)
- F1-macro : **72,46 %**
- Accuracy : 72,21 %
- Temps d'entraînement : ~2 secondes
- **Avantage :** très rapide, bon sur Neutre (F1: 63,7 %)

#### LinearSVC (Support Vector Machine)
- F1-macro : **74,29 %** ★ **MEILLEUR**
- Accuracy : 74,45 %
- Temps d'entraînement : ~8 secondes
- **Avantage :** meilleur F1 global, bon équilibre entre classes

### 4.4 Tuning (Sprint 3)

**GridSearchCV appliqué :**
- LinearSVC : C ∈ {0.1, 1, 10}, loss ∈ {'squared_hinge', 'hinge'}
- MultinomialNB : alpha ∈ {0.01, 0.1, 1}

**Résultat :** Meilleurs paramètres vérifiés et sauvegardés en `models/*.joblib`.

### 4.5 Modèles Avancés (Pipeline Étendu)

#### Random Forest (200 arbres)
- F1-macro : 67,34 %
- Nécessite SVD (300 composantes) pour dimensionnalité

#### XGBoost (300 estimateurs)
- F1-macro : 71,37 %
- Également utilisé avec SVD

---

## 5. Évaluation

### 5.1 Stratégie d'Évaluation

**Train/Test Split :** 80/20 avec stratification (preserves class distribution).

**Métriques calculées :**
- **Accuracy** : proportion globale de bonnes prédictions
- **F1-macro** : moyenne non pondérée des F1 par classe (utile pour classes déséquilibrées)
- **F1 par classe** : Négatif, Neutre, Positif
- **Matrice de confusion** : pour diagnostiquer les confusions entre classes

### 5.2 Résultats Complets

| Modèle | Accuracy | F1-macro | F1-Nég | F1-Neutre | F1-Pos | Temps |
|---|---|---|---|---|---|---|
| VADER | 49,18 % | 40,56 % | 45 % | 20 % | 58 % | <1s |
| Naive Bayes | 72,21 % | 72,46 % | 74,2 % | 63,7 % | 79,4 % | 2s |
| LinearSVC ★ | 74,45 % | 74,29 % | 78,3 % | 63,6 % | 80,9 % | 8s |
| Random Forest | 67,17 % | 67,34 % | 70,5 % | 59,7 % | 71,9 % | 45s |
| XGBoost | 71,28 % | 71,37 % | 75,1 % | 62,4 % | 76,6 % | 25s |

### 5.3 Conclusion sur l'Évaluation

**LinearSVC est le meilleur modèle** pour ce projet, avec :
- F1-macro = 74,29 % (> 70 % requis) ✅
- Performance équilibrée sur les 3 classes
- Temps d'inférence très rapide (<1ms par prédiction)
- Taille du modèle compacte (~10 MB)

---

## 6. Déploiement

### 6.1 Architecture

**Stack :**
- **Backend :** FastAPI (Python 3.9+) + Waitress (production WSGI)
- **Frontend :** HTML5 + vanilla JavaScript + Chart.js
- **Base de données :** aucune (modèles en mémoire)
- **Modèles :** sauvegardés en `.joblib` (scikit-learn)

### 6.2 Endpoints API

**GET `/health`**
```json
{
  "status": "ok",
  "models_loaded": ["svm", "nb"],
  "tfidf_loaded": true
}
```

**POST `/predict`**
```json
Request: { "text": "Great product!", "model": "svm" }
Response: {
  "text": "Great product!",
  "sentiment": "positive",
  "confidence": 0.92,
  "scores": { "negative": 0.05, "neutral": 0.03, "positive": 0.92 },
  "model_used": "LinearSVC"
}
```

**POST `/predict/batch`**
- Accepte jusqu'à 100 textes par batch
- Retourne `{ "results": [...], "count": N }`

**GET `/metrics`**
- Retourne `evaluation_results.json` (accuracy, F1 par modèle, matrices de confusion)

**GET `/models`**
- Liste les modèles disponibles et recommande LinearSVC

### 6.3 Frontend

**Vues :**
1. **Predict** : formulaire simple pour tester une prédiction
2. **Dashboard** : graphiques Chart.js (distribution des sentiments, comparaison des modèles)
3. **Pipeline** : explique les étapes du pipeline NLP
4. **Metrics** : affiche les résultats d'évaluation détaillés
5. **About** : crédits d'équipe

**Design :** Dark theme (tailwind-inspiré), responsive mobile-first.

### 6.4 Déploiement Local / Production

**Développement :**
```bash
python run_server.py
# Accéder à http://localhost:8000
```

**Production :**
```bash
waitress-serve --port=8000 api.main:app
```

---

## 7. Conduite Agile du Projet

### 7.1 Framework Scrum + CRISP-DM

Le projet a suivi une conduite Agile avec :
- **4 sprints** (0, 1, 2, 3)
- **Rituels :** Daily standup (10 min), Sprint review (30 min), Rétrospective (20 min)
- **Outil :** Trello pour le backlog et les burn-down charts

### 7.2 Vélocité

| Sprint | Points planifiés | Points livrés | Vélocité |
|---|---|---|---|
| Sprint 0 | 6 | 6 | 6 |
| Sprint 1 | 14 | 11 | 11 |
| Sprint 2 | 16 | 16 | 16 |
| Sprint 3 | 24 | 18 | 18 |

**Observation :** progression régulière de la vélocité (6 → 11 → 16 → 18), montrant une meilleure estimation et une montée en compétence.

### 7.3 Burn-down Charts

Les courbes de burn-down révèlent :
- **Sprint 1 :** retard en milieu de sprint (EDA plus long que prévu) → rattrape jusqu'à J14
- **Sprint 2 & 3 :** suivi très proche de l'idéal, signe d'une meilleure planification

---

## 8. Conclusion et Leçons Apprises

### 8.1 Objectifs Atteints

✅ **Classification automatique** : LinearSVC atteint F1 = 74,29 % (> 70 requis)  
✅ **API REST + Dashboard** : déployé et fonctionnel  
✅ **Tableau de bord Tableau** : 5 feuilles + 9 CSV exports  
✅ **Documentation CRISP-DM** : ce rapport + rétrospective agile  
✅ **Conduite agile** : 3 sprits avec burn-down, vélocité progressive  

### 8.2 Points Forts du Projet

1. **Technique :** Pipeline NLP solide, modèles multiples évalués, SVD/SMOTE démontrés
2. **Méthodologie :** CRISP-DM cycles appliqués itérativement
3. **Délivrables :** API REST, frontend, documentations, Jupyter notebooks
4. **Collaboration :** équipe alignée, communicaton régulière

### 8.3 Points d'Amélioration

1. Tuning hyperparamétriques plus exhaustifs (RandomSearch, Bayesian Optimization)
2. Analyse d'erreurs plus fine (confusion matrix patterns, adversarial examples)
3. Augmentation des données pour les classes déséquilibrées
4. Test A/B du frontend avec des utilisateurs réels
5. Monitoring en production (logs, drift detection)

### 8.4 Leçons Apprises

- **F1-macro est crucial** pour des données équilibrées mais hétérogènes
- **TF-IDF + SVM est une combinaison robuste** pour la classification de texte sans GPU
- **Visualisation et dashboarding** améliorent considérablement l'adoption en production
- **SMOTE est difficile** à implémenter correctement avec TF-IDF sparse
- **Agile + CRISP-DM = cycles courts** qui permettent d'itérer rapidement

---

**Fin du rapport**
