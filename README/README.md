*README*

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)
[![Tests](https://github.com/douaeharrak/Coding-Week-GR24/actions/workflows/ci.yml/badge.svg)](https://github.com/douaeharrak/Coding-Week-GR24/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://docker.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# Description du projet 

Application d'aide à la décision médicale développée pour prédire le **taux de réussite des greffes de moelle osseuse chez les patients pédiatriques**. 

Notre solution utilise des modèles de **Machine Learning explicables (SHAP)** pour fournir aux médecins des prédictions précises et interprétables, facilitant ainsi la prise de décision clinique.

# Objectifs 

- Prédire le succès des greffes avec une haute précision
- Fournir des explications transparentes via SHAP
- Offrir une interface intuitive pour les médecins
- Garantir la reproductibilité via Docker et CI/CD

# Dataset

Nous utilisons le dataset **"Bone Marrow Transplant: Children"** de l'UCI Machine Learning Repository :
- **Source** : [UCI Repository - ID 565](https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children)
- **Instances** : 187 patients pédiatriques
- **Features** : 36 variables cliniques
- **Target** : Survie post-greffe (0 = vivant, 1 = décédé)

# Prérequis pour l'installation 

- Python 3.11
- Docker 
- Git

# Cloner le repository 

git clone https://github.com/douaeharrak/Coding-Week-GR24.git
cd Coding-Week-GR24

# Créer et activer un environnement virtuel 

python -m venv venv
venv\Scripts\activate

# Installer les dépendances

pip install -r requirements.txt

# Lancer l'application

streamlit run app/app.py

# Entrainer les modèles

python src/train_model.py

# Evaluer les performances 

python src/evaluate_model.py

# Exécuter les tests 

pytest tests/ -v

# Architecture du projet 

Coding-Week-GR24/
├── 📁 .github/
│   └── 📁 workflows/
│       └── 📄 ci.yml                 # CI/CD avec GitHub Actions
├── 📁 src/
│   ├── 📄 data_processing.py          # Prétraitement des données
│   ├── 📄 train_model.py              # Entraînement des modèles
│   └── 📄 evaluate_model.py           # Évaluation et métriques
├── 📁 tests/
│   ├── 📄 __init__.py
│   ├── 📄 test_data_processing.py     # Tests preprocessing
│   └── 📄 test_model.py                # Tests modèles
├── 📁 app/
│   └── 📄 app.py                       # Interface Streamlit
├── 📁 data/
│   └── 📄 bone_marrow_clean.csv        # Dataset (ignoré par git)
├── 📁 notebooks/
│   └── 📄 eda.ipynb                     # Analyse exploratoire
├── 📄 Dockerfile                         # Configuration Docker
├── 📄 .dockerignore                       # Fichiers ignorés par Docker
├── 📄 requirements.txt                    # Dépendances Python
├── 📄 .gitignore                          # Fichiers ignorés par Git
└── 📄 README.md                           # Documentation

# Discussion

1) le dataset est-il équilibré ? 
Non, déséquilibre modéré (60/40)
Solution : SMOTE avec random_state=42
Impact : Amélioration du recall pour la classe minoritaire

2) Quel modèle à donné les meilleurs résultats ? 
