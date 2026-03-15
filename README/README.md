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

- Déséquilibre modéré (60/40)
Solution : SMOTE avec random_state=42
Impact : Amélioration du recall pour la classe minoritaire

- Le modèle ML avec les meilleurs résultats est XGBoost. 
Indicateurs de performance :

ROC-AUC : 0.900 (Excellente capacité à distinguer les survivants des non-survivants)
Précision : 0.933 (93.3% des patients prédits à risque le sont réellement)
Rappel : 0.737 (73.7% des patients à risque sont correctement identifiés)
F1-Score : 0.828 (Excellent équilibre entre précision et rappel )

Sur le plan médical, ces résultats signifient que dans un contexte clinique :
. Le modèle identifie correctement **9 patients à risque sur 10**
. **Moins de 7% des alertes sont fausses**, évitant un stress inutile
. La décision médicale peut s'appuyer sur une prédiction fiable

# Prompt engineering

**tache sélectionnée** : Création des tests pytest pour les fonctions de preprocessing (test_data_processing.py)

- promp initial utilisé : 
"Génère des tests pytest pour les fonctions handle_missing_values, 
balance_classes et handle_outliers en Python. Inclus la gestion 
des cas normaux, des cas limites et des erreurs."

- ce que l'IA m'a fourni : 
Une structure de base pour les tests
Des exemples de DataFrames synthétiques
Des assertions pour vérifier les résultats
La gestion des cas d'erreur (edge cases)

- améliorations apportées
J'ai dû adapter le code généré pour qu'il corresponde exactement aux fonctions de mon camarade :
Adapter les noms de paramètres
Ajouter des tests spécifiques pour le SMOTE
Vérifier que la target n'est pas modifiée par le scaling
Tester le retour du scaler

- ce que j'ai appris : 
Plus le prompt est détaillé, plus la réponse est utile. Ajouter "avec pandas DataFrame" ou "pour du code médical" améliore considérablement les résultats.
Le premier résultat n'est jamais parfait. Il faut reformuler, préciser, demander des ajustements.
L'IA peut générer du code qui semble correct mais qui ne correspond pas exactement à nos fonctions. Chaque ligne doit être testée et adaptée.
L'IA m'a fourni une base solide que j'ai ensuite personnalisée pour notre projet spécifique, gagnant ainsi un temps précieux.

# Equipe 

HARRAK Douae
GUIMBONE Daouda
HASSAR Nada 
WANDAOGO Zénabou 
DIONE Amina 
  