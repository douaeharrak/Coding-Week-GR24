"""
evaluate_model.py
=================
Évaluation et comparaison des 4 modèles ML.
Métriques : ROC-AUC, Précision, Rappel, F1
Sélectionne automatiquement le meilleur modèle.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")   # backend non-interactif (pas d'affichage GUI)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Import du module d'entraînement ──────────────────────────────────────────
# train_all() retourne les modèles fittés + les splits train/test + le scaler
from train_model import train_all

# ── Palette de couleurs par modèle (cohérente sur tous les graphiques) ───────
PALETTE = {
    "RandomForest": "#4C72B0",
    "XGBoost":      "#DD8452",
    "SVM":          "#55A868",
    "LightGBM":     "#C44E52",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. CALCUL DES MÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_all(models, X_test, y_test, threshold=0.50):
    """
    Calcule les métriques de chaque modèle sur le jeu de test.

    Paramètres
    ----------
    models    : dict {nom: modèle fitté}
    X_test    : features du jeu de test
    y_test    : labels réels
    threshold : seuil de décision pour convertir les probabilités en classes
                (0.5 par défaut, peut être ajusté pour favoriser le rappel)

    Retourne
    --------
    DataFrame trié par ROC-AUC décroissant, index réinitialisé (0 = meilleur).

    .reset_index(drop=True) après sort_values pour que l'index
    reflète bien le classement (0 = meilleur). Sans ça, l'index pandas gardait
    l'ordre d'insertion original et le tag '← MEILLEUR' pointait sur le mauvais modèle.
    """
    rows = []
    for name, model in models.items():
        # predict_proba retourne [[P(classe=0), P(classe=1)], ...]
        # On prend la colonne 1 → probabilité de survie
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        rows.append({
            "Modèle":    name,
            "ROC-AUC":   round(roc_auc_score(y_test, y_prob),                    4),
            "Précision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Rappel":    round(recall_score(y_test, y_pred, zero_division=0),    4),
            "F1":        round(f1_score(y_test, y_pred, zero_division=0),        4),
        })

    # sort + reset_index → index 0 = meilleur modèle garanti
    return (
        pd.DataFrame(rows)
        # Dans evaluate_all(), changer le tri 
        .sort_values(["ROC-AUC", "F1"], ascending=False)  # F1 comme départage
        .reset_index(drop=True)   # sans ça, index != rang
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. AFFICHAGE TERMINAL
# ══════════════════════════════════════════════════════════════════════════════

def afficher_comparaison(results):
    """
    Affiche un tableau comparatif dans le terminal.

    Correction : utilisation de enumerate() pour obtenir le rang réel
    dans le classement. L'ancien code utilisait '_' (index pandas) pour
    détecter le meilleur modèle → le tag '← MEILLEUR' n'apparaissait jamais
    correctement si l'index pandas ne commençait pas à 0.
    Avec reset_index() dans evaluate_all + enumerate ici, rank==0 correspond
    toujours au meilleur modèle.
    """
    icones = {
        "RandomForest": "🌲",
        "XGBoost":      "⚡",
        "SVM":          "🔷",
        "LightGBM":     "💡",
    }

    print("\n" + "=" * 65)
    print("   COMPARAISON DES 4 ALGORITHMES — TEST SET")
    print("=" * 65)
    print(f"  {'Modèle':<15} {'ROC-AUC':>8} {'Précision':>10} {'Rappel':>8} {'F1':>8}")
    print("-" * 65)

    for rank, (_, row) in enumerate(results.iterrows()):
        # rank est le rang réel dans le classement (0 = meilleur)
        # _ est l'index pandas (identique à rank grâce à reset_index)
        nom   = row["Modèle"]
        tag   = " ← MEILLEUR" if rank == 0 else ""   # ← CORRECTION
        icone = icones.get(nom, "•")
        print(
            f"  {icone} {nom:<13} {row['ROC-AUC']:>8.4f} "
            f"{row['Précision']:>10.4f} {row['Rappel']:>8.4f} "
            f"{row['F1']:>8.4f}{tag}"
        )

    print("=" * 65)

    best = results.iloc[0]   # index 0 = meilleur grâce à reset_index
    print(f"\n  🏆 Meilleur modèle  : {best['Modèle']}")
    print(f"  📊 ROC-AUC          : {best['ROC-AUC']:.4f}")
    print(f"  🎯 Précision        : {best['Précision']:.4f}")
    print(f"  🔍 Rappel           : {best['Rappel']:.4f}")
    print(f"  ⚖️  F1-Score         : {best['F1']:.4f}")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 3. GRAPHIQUES
# ══════════════════════════════════════════════════════════════════════════════

def plot_metrics(results, out_dir):
    """
    Génère un graphique en barres groupées comparant les 4 métriques
    pour chaque modèle.
    """
    metrics = ["ROC-AUC", "Précision", "Rappel", "F1"]
    x     = np.arange(len(metrics))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (_, row) in enumerate(results.iterrows()):
        vals   = [row[m] for m in metrics]
        offset = (i - len(results) / 2 + 0.5) * width
        bars   = ax.bar(
            x + offset, vals, width,
            label=row["Modèle"],
            color=PALETTE.get(row["Modèle"], "gray"),
            alpha=0.85,
            edgecolor="black",
        )
        # Afficher la valeur au-dessus de chaque barre
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.005,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Comparaison des 4 algorithmes — Greffe de Moelle Osseuse",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(out_dir, "comparaison_metriques.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✔ Graphique métriques : {path}")


def plot_roc(models, X_test, y_test, out_dir):
    """
    Génère les courbes ROC de tous les modèles sur un même graphique.
    La diagonale pointillée représente un classifieur aléatoire (AUC=0.5).
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, model in models.items():
        y_prob       = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _  = roc_curve(y_test, y_prob)
        auc          = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2,
                color=PALETTE.get(name, "gray"),
                label=f"{name} (AUC={auc:.3f})")

    # Diagonale = classifieur aléatoire
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aléatoire")
    ax.set_xlabel("Taux faux positifs")
    ax.set_ylabel("Taux vrais positifs")
    ax.set_title("Courbes ROC — Test set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✔ Courbes ROC         : {path}")


def plot_confusion(models, X_test, y_test, out_dir, threshold=0.50):
    """
    Génère les matrices de confusion côte à côte pour chaque modèle.

    Correction : les y_pred sont recalculés ici directement depuis les modèles
    au lieu d'être lus depuis le DataFrame results. L'ancienne version stockait
    _y_pred dans le DataFrame, ce qui était fragile : si quelqu'un supprimait
    ces colonnes ou appelait evaluate_all séparément, la fonction crashait.
    Recalculer garantit l'indépendance et la robustesse de la fonction.
    """
    n    = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]   # toujours itérable

    for ax, (name, model) in zip(axes, models.items()):
        # Recalcul propre et indépendant du DataFrame results
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        cm     = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Décès", "Survie"],
            yticklabels=["Décès", "Survie"],
            ax=ax,
        )
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")

    fig.suptitle("Matrices de Confusion", fontsize=13)
    plt.tight_layout()

    path = os.path.join(out_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Matrices confusion  : {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OUT_DIR = "outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Entraînement ──────────────────────────────────────────────────────────
    # train_all() lance le preprocessing complet + entraîne les 4 modèles
    models, cv_scores, (X_train, X_test, y_train, y_test), scaler = train_all()

    # ── Évaluation sur le test set ────────────────────────────────────────────
    results = evaluate_all(models, X_test, y_test)

    # ── Affichage terminal ────────────────────────────────────────────────────
    afficher_comparaison(results)

    # ── Génération des graphiques ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  GÉNÉRATION DES GRAPHIQUES")
    print("=" * 65)

    plot_metrics(results, OUT_DIR)
    plot_roc(models, X_test, y_test, OUT_DIR)
    plot_confusion(models, X_test, y_test, OUT_DIR)   # ← signature corrigée

    print("\n   Évaluation complète ! Voir le dossier outputs/")