"""
train_model.py
==============
Entraînement des 4 modèles ML en utilisant le pipeline
de preprocessing (data_processing.py).

Colonne cible : survival_status (1=survie, 0=décès)

Modèles entraînés :
    - RandomForest         (ensemble de décision)
    - XGBoost / GradientBoosting (boosting)
    - SVM                  (séparateur à vaste marge)
    - LightGBM / HistGradientBoosting (boosting rapide)
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

import matplotlib
matplotlib.use("Agg")   # backend non-interactif (pas d'affichage GUI)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Ajout du dossier src/ au path Python ─────────────────────────────────────
# Nécessaire pour que Python trouve data_processing.py qui est dans src/
# sans modifier la variable d'environnement PYTHONPATH manuellement.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_processing import run_preprocessing


# ── Import optionnel de XGBoost ───────────────────────────────────────────────
# XGBoost n'est pas toujours installé dans tous les environnements.
# Si absent → fallback sur GradientBoostingClassifier de sklearn (même logique).
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── Import optionnel de LightGBM ──────────────────────────────────────────────
# Même logique : fallback sur HistGradientBoostingClassifier si absent.
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


#─────────────────────────────────────
# FONCTION : get_models
# ─────────────────────────────────────

def get_models():
    """
    Retourne un dictionnaire des 4 modèles à entraîner.

    Pourquoi ces 4 modèles ?
    - RandomForest   : robuste, peu sensible aux hyperparamètres,
                       bonne baseline pour des données médicales
    - XGBoost        : boosting séquentiel très performant sur données tabulaires
    - SVM            : efficace sur des espaces de grande dimension (après scaling)
    - LightGBM       : boosting rapide, gère bien les features nombreuses

    Corrections apportées :
    - Suppression de `use_label_encoder=False` dans XGBClassifier
      → paramètre déprécié et supprimé dans les versions récentes de XGBoost,
        son utilisation lève une TypeError
    """

    # XGBoost ou fallback GradientBoosting
    xgb = (
        XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            eval_metric="logloss",   # métrique d'évaluation interne pendant fit
            random_state=42,
        )
        if XGBOOST_AVAILABLE
        else GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        )
    )

    # LightGBM ou fallback HistGradientBoosting
    lgbm = (
        LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,   # silence les logs LightGBM
        )
        if LIGHTGBM_AVAILABLE
        else HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            random_state=42,
        )
    )

    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,   # utilise tous les cœurs CPU disponibles
        ),
        "XGBoost":  xgb,
        "SVM":      SVC(
            kernel="rbf",          # noyau radial : efficace sur données non-linéaires
            probability=True,      # active predict_proba() (nécessaire pour ROC-AUC)
            C=1.0,                 # paramètre de régularisation
            random_state=42,
        ),
        "LightGBM": lgbm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FONCTION PRINCIPALE : train_all
# ─────────────────────────────────────────────────────────────────────────────

def train_all(
    path="data/processed/bone_marrow_clean_pregreffe.csv",  
    target_col="survival_status",
    test_size=0.20,
    random_state=42,
):
    """
    Pipeline complet d'entraînement :

        Étape 1 : Preprocessing (Membre 2)
        Étape 2 : Split train / test stratifié
        Étape 3 : Cross-validation + entraînement final sur train
        Étape 4 : Sauvegarde des modèles et du scaler

    Paramètres
    ----------
    path         : chemin vers le CSV nettoyé
    target_col   : nom de la colonne cible (survival_status)
    test_size    : proportion du jeu de test (20 % par défaut)
    random_state : graine aléatoire → résultats reproductibles

    Retourne
    --------
    trained_models : dict {nom: modèle fitté}
    cv_scores      : dict {nom: array des scores CV}
    (X_train, X_test, y_train, y_test) : splits pour evaluate_models.py
    scaler         : StandardScaler fitté → réutilisé dans Streamlit
    """

    # ── Étape 1 : Preprocessing ───────────────────────────────────────────────
    # run_preprocessing() enchaîne :
    #   load → missing values → outliers → encode → optimize → scale → SMOTE
    # Retourne X et y déjà équilibrés et scalés, prêts pour l'entraînement
    X, y, scaler = run_preprocessing(path=path, target_col=target_col)

    # ── S'assurer que y est bien une pandas Series ──────────────
    # Après SMOTE, y peut être retourné comme numpy array selon la version
    # d'imbalanced-learn. train_test_split accepte les deux, mais pd.Series
    # garantit la cohérence avec le reste du pipeline (value_counts, etc.)
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name=target_col)

    # ── Étape 2 : Split train / test ─────────────────────────────────────────
    # stratify=y → conserve les proportions de classes dans train ET test
    # Indispensable pour ne pas biaiser l'évaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    print(f"\n  Split : {len(X_train)} train / {len(X_test)} test")

    # ── Étape 3 : Entraînement avec cross-validation ─────────────────────────
    # StratifiedKFold → conserve les proportions de classes dans chaque fold
    # scoring="roc_auc" → métrique adaptée aux données médicales déséquilibrées
    # (meilleure que l'accuracy car sensible aux faux négatifs)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    trained_models = {}
    cv_scores      = {}

    print("\n" + "=" * 55)
    print("  ENTRAÎNEMENT DES 4 MODÈLES")
    print("=" * 55)

    for name, model in get_models().items():
        # cross_val_score évalue le modèle en CV sans le fitter définitivement
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )

        # fit final sur tout le train set après validation
        model.fit(X_train, y_train)

        trained_models[name] = model
        cv_scores[name]      = scores

        print(f"  {name:<15} CV-AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    print("=" * 55)

    # ── Étape 4 : Sauvegarde ─────────────────────────────────────────────────
    # Les modèles sont sérialisés avec pickle pour être réutilisés dans :
    #   - evaluate_models.py  (évaluation des métriques)
    #   - app.py Streamlit    (prédiction sur nouveau patient)
    os.makedirs("models", exist_ok=True)

    for name, model in trained_models.items():
        model_path = f"models/{name.lower()}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Sauvegardé : {model_path}")

    # Le scaler est sauvegardé séparément → Streamlit doit transformer
    # les données d'un nouveau patient avec exactement les mêmes mean/std
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("  Sauvegardé : models/scaler.pkl")

    print("\n  Tous les modèles sont prêts.")

    return trained_models, cv_scores, (X_train, X_test, y_train, y_test), scaler


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trained_models, cv_scores, splits, scaler = train_all()
    print("\n  Entraînement terminé !")