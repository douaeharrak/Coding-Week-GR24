# Application d'Aide à la Décision Médicale
## Prévision du Succès des Greffes de Moelle Osseuse Pédiatriques

**Projet 4 — Coding Week 09–15 Mars 2026**  
**Groupe 24 — École Centrale Casablanca**

---

## Présentation du projet

Ce projet propose une application web interactive permettant aux cliniciens d'estimer la probabilité de succès d'une greffe de moelle osseuse pédiatrique à partir de données cliniques concernant le patient et le donneur.

L'application s'appuie sur des modèles de Machine Learning entraînés sur un dataset médical réel de 187 patients pédiatriques, et intègre une analyse d'explicabilité SHAP afin de justifier chaque prédiction de manière transparente et interprétable.

---

## Dataset

| Propriété | Valeur |
|-----------|--------|
| Nom | Bone Marrow Transplant in Children |
| Source | UCI Machine Learning Repository |
| Lien | https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children |
| Format | `bone-marrow.arff` |
| Observations | 187 patients pédiatriques |
| Attributs | 37 variables cliniques |
| Types de données | 21 int64, 15 float64, 1 str |
| Mémoire brute | 54.2 KB |

---

## Structure du projet

```
Coding-Week-GR24/
│
├── .github/
│   └── workflows/                  # Pipeline CI/CD GitHub Actions
│
├── app/
│   └── app.py                      # Application Streamlit
│
├── data/
│   ├── raw/                        # Données brutes (bone-marrow.arff)
│   └── processed/                  # Données nettoyées
│       ├── bone_marrow_clean.csv
│       └── bone_marrow_clean_pregreffe.csv
│
├── models/                         # Modèles entraînés (.pkl)
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── randomforest_model.pkl
│   ├── svm_model.pkl
│   └── scaler.pkl
│
├── Notebooks/
│   └── Edaa.ipynb                  # Analyse exploratoire des données
│
├── outputs/                        # Résultats et visualisations
│   ├── comparaison_metriques.png
│   ├── confusion_matrices.png
│   ├── cv_auc_comparison.png
│   ├── feature_importance_randomforest.png
│   ├── feature_importance_xgboost.png
│   ├── metrics_comparison.png
│   ├── pr_curves.png
│   ├── rapport_greffe.html
│   └── roc_curves.png
│
├── README/
│   └── README.md
│
├── src/
│   ├── data_processing.py          # Pipeline de traitement des données
│   ├── evaluate_model.py           # Évaluation des modèles
│   └── train_model.py              # Entraînement des modèles
│
├── tests/
│   ├── __init__.py
│   └── test_data_processing.py     # Tests unitaires
│
├── .gitignore
└── requirements.txt
```

---

## Installation

### Prérequis

- Python 3.13.7
- pip

### Étapes

```bash
# Cloner le dépôt
git clone https://github.com/douaeharrak/Coding-Week-GR24.git
cd Coding-Week-GR24

# Créer et activer un environnement virtuel
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

---

## Utilisation

**1. Entraîner le modèle**

```bash
python src/train_model.py
```

**2. Lancer l'application**

```bash
streamlit run app/app.py --server.port 8507
```

Ouvrir ensuite dans le navigateur : http://localhost:8507

**3. Exécuter les tests**

```bash
python -m pytest tests/
```

---

## Modèles de Machine Learning

Quatre modèles ont été entraînés et évalués selon les métriques ROC-AUC, précision, recall et F1-score.

| Modèle | Description | Résultat |
|--------|-------------|----------|
| XGBoost | Gradient boosting optimisé | **Meilleur modèle** |
| LightGBM | Gradient boosting rapide et léger | — |
| Random Forest | Ensemble d'arbres de décision | — |
| SVM | Machine à vecteurs de support | — |

L'application charge automatiquement le premier modèle disponible dans le dossier `models/`, XGBoost étant prioritaire.

---

## Questions essentielles

### Le dataset était-il équilibré ?

Non. Le dataset présente un déséquilibre modéré : environ 60 % de survivants et 40 % de non-survivants. Trois stratégies ont été documentées et évaluées pour y remédier :

- Suréchantillonnage (SMOTE)
- Sous-échantillonnage
- Ajustement du poids des classes (`class_weight`)

### Quel modèle a donné les meilleurs résultats ?

XGBoost a obtenu les meilleures performances globales sur l'ensemble des métriques. Les comparaisons détaillées sont disponibles dans `outputs/comparaison_metriques.png`, `outputs/roc_curves.png` et `outputs/cv_auc_comparison.png`.

### Quelles caractéristiques médicales ont le plus influencé les prédictions ?

D'après l'analyse SHAP, les variables les plus déterminantes sont les suivantes :

| Rang | Variable | Signification clinique |
|------|----------|------------------------|
| 1 | `CD34kgx10d6` | Dose de cellules souches CD34+/kg — facteur #1 selon la littérature |
| 2 | `HLAmatch` | Compatibilité HLA |
| 3 | `Recipientage` | Âge du receveur |
| 4 | `Donorage` | Âge du donneur |
| 5 | `CMVstatus` | Statut CMV combiné donneur/receveur |

Les visualisations sont disponibles dans `outputs/feature_importance_xgboost.png`.

### Ingénierie des invites (Prompt Engineering)

**Tâche sélectionnée :** Développement de la fonction `optimize_memory(df)` dans `src/data_processing.py`.

**Invite utilisée :**
> *"Écris une fonction Python `optimize_memory(df)` qui parcourt les colonnes d'un DataFrame pandas et réduit l'utilisation mémoire en convertissant les types float64 en float32 et int64 en int32. Affiche la mémoire avant et après optimisation avec le pourcentage de gain."*

**Résultat :** La fonction a réduit l'utilisation mémoire de 27 %, tel que démontré dans `Notebooks/Edaa.ipynb`.

**Analyse :** L'invite était précise et orientée résultat. Une amélioration possible consisterait à inclure la conversion des colonnes catégorielles vers le type `category` de pandas, ce qui permettrait un gain supplémentaire.

---

## Pipeline de traitement des données

```
Données brutes (bone-marrow.arff)
        |
        v
handle_missing_values()   -- Imputation des 81 cellules manquantes (1.17 % global)
        |
        v
encode_features()         -- Encodage des variables catégorielles
        |
        v
handle_outliers()         -- Détection et traitement des valeurs aberrantes
        |
        v
optimize_memory()         -- Conversion float64/int64 -> float32/int32 (gain : 27 %)
        |
        v
scaler.transform()        -- Normalisation des features numériques
        |
        v
model.predict_proba()     -- Probabilité de succès de la greffe
```

### Valeurs manquantes identifiées

| Colonne | Valeurs manquantes | Pourcentage |
|---------|--------------------|-------------|
| extcGvHD | 31 | 16.58 % |
| CMVstatus | 16 | 8.56 % |
| RecipientCMV | 14 | 7.49 % |
| CD3dCD34 | 5 | 2.67 % |
| CD3dkgx10d8 | 5 | 2.67 % |
| Rbodymass | 2 | 1.07 % |
| DonorCMV | 2 | 1.07 % |
| **Total** | **81** | **1.17 %** |

---

## Variables cliniques utilisées

### Receveur (Patient)

| Variable | Description |
|----------|-------------|
| `Recipientage` | Âge (0–20 ans) |
| `Recipientgender` | Sexe (0 = Féminin, 1 = Masculin) |
| `Rbodymass` | Poids corporel (kg) |
| `RecipientABO` | Groupe sanguin (0 = O, 1 = A, 2 = B, 3 = AB) |
| `RecipientCMV` | Statut CMV (0 = Absent, 1 = Présent) |
| `Recipientage10` | Receveur < 10 ans — calculé automatiquement |

### Donneur

| Variable | Description |
|----------|-------------|
| `Donorage` | Âge (18–70 ans) |
| `Donorage35` | Donneur < 35 ans — calculé automatiquement |
| `DonorABO` | Groupe sanguin |
| `DonorCMV` | Statut CMV |

### Compatibilité Immunologique

| Variable | Description |
|----------|-------------|
| `HLAmatch` | Compatibilité HLA (/10) |
| `HLAmismatch` | Nombre d'antigènes HLA différents |
| `HLAgrI` | Score HLA global (HLAmatch / 10) |
| `ABOmatch` | Compatibilité ABO |
| `Antigen` | Antigènes incompatibles |
| `Alel` | Allèles incompatibles |
| `Gendermatch` | Concordance de sexe |
| `CMVstatus` | Statut CMV combiné donneur/receveur |

### Maladie et Protocole

| Variable | Description |
|----------|-------------|
| `Disease` | Diagnostic (ALL, AML, CML, MDS, SAA, Fanconi) |
| `Riskgroup` | Groupe de risque (0 = Faible, 1 = Élevé) |
| `Stemcellsource` | Source des cellules souches |
| `Txpostrelapse` | Greffe après rechute (0 = Non, 1 = Oui) |
| `Diseasegroup` | Maladie maligne vs non-maligne |
| `IIIV` | Stade avancé II à IV |

### Données de la Greffe

| Variable | Description |
|----------|-------------|
| `CD34kgx10d6` | Dose CD34+ (x10^6/kg) |
| `CD3dkgx10d8` | CD3+ (x10^8/kg) |
| `CD3dCD34` | Ratio CD3/CD34 — calculé automatiquement (CD3 / CD34) |

---

## Explicabilité SHAP

Chaque prédiction est accompagnée d'une analyse SHAP (SHapley Additive exPlanations) qui identifie les huit variables ayant le plus influencé la décision du modèle.

- Valeur SHAP positive : le facteur augmente la probabilité de succès
- Valeur SHAP négative : le facteur diminue la probabilité de succès

---

## Interprétation des résultats

| Score | Pronostic | Recommandation |
|-------|-----------|----------------|
| >= 75 % | Favorable | Indicateurs globalement positifs |
| 50 – 74 % | Modéré | Surveillance rapprochée recommandée |
| < 50 % | Défavorable | Consultation pluridisciplinaire conseillée |

---

## Intégration continue — GitHub Actions

Le projet inclut un pipeline CI/CD via GitHub Actions (`.github/workflows/`) qui exécute automatiquement les tests suivants à chaque push :

- Vérification de la gestion des valeurs manquantes (`handle_missing_values`)
- Vérification de la fonction `optimize_memory(df)`
- Vérification du chargement et de la prédiction du modèle

---

## Dépendances principales

```
streamlit
pandas==3.0.1
numpy==2.0.1
seaborn==0.13.2
scikit-learn
xgboost
lightgbm
matplotlib
shap
pytest
```

Liste complète disponible dans `requirements.txt`.

---

## Équipe

**Groupe 24 — Coding Week 09–15 Mars 2026 — École Centrale Casablanca**

| Membre | Email |
|--------|-------|
| DIONE Amina | amina.dione@centrale-casablanca.ma |
| GUIMBONE Daouda | daouda.guimbone@centrale-casablanca.ma |
| HARRAK Douae | douae.harrak@centrale-casablanca.ma |
| NADA Hassar | nada.hassar@centrale-casablanca.ma |
| WANDAOGO Zenabou | zenabou.wandaogo@centrale-casablanca.ma— |

**Encadrement :** Kawtar ZERHOUNI et Équipe pédagogique — École Centrale Casablanca

---

## Liste de contrôle des livrables

- [x] Code structuré de manière professionnelle
- [x] Analyse exploratoire documentée (`Notebooks/Edaa.ipynb`)
- [x] Gestion du déséquilibre des classes documentée
- [x] Intégration fonctionnelle de l'explicabilité SHAP
- [x] Interface web intuitive (Streamlit)
- [x] Pipeline CI/CD fonctionnel (GitHub Actions)
- [x] Fonction `optimize_memory()` implémentée et démontrée
- [x] Documentation complète incluant l'ingénierie des invites
- [x] Projet entièrement reproductible

---

## Avertissement

Cet outil est une aide à la décision et ne remplace en aucun cas l'avis d'un professionnel de santé qualifié. Les prédictions sont fournies à titre indicatif uniquement.
