"""
evaluate_model.py
=================
Évaluation des modèles entraînés pour la prédiction du taux de réussite
de la greffe de moelle osseuse chez les enfants.

Métriques calculées :
  • ROC-AUC
  • Précision (Precision)
  • Rappel   (Recall)
  • Score F1

Sélectionne automatiquement le meilleur modèle et génère un rapport HTML.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Import local ──────────────────────────────────────────────────────────────
from train_model import generate_synthetic_data, train_all

warnings.filterwarnings("ignore")

PALETTE = {
    "RandomForest": "#4C72B0",
    "XGBoost":      "#DD8452",
    "SVM":          "#55A868",
    "LightGBM":     "#C44E52",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. CALCUL DES MÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_all(trained_pipelines: dict,
                 X_test, y_test,
                 threshold: float = 0.50) -> pd.DataFrame:
    """
    Calcule ROC-AUC, Précision, Rappel et F1 pour chaque modèle.

    Retourne un DataFrame trié par ROC-AUC décroissant.
    """
    rows = []
    for name, pipe in trained_pipelines.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        rows.append({
            "Modèle":     name,
            "ROC-AUC":    round(roc_auc_score(y_test, y_prob),        4),
            "Précision":  round(precision_score(y_test, y_pred,
                                                zero_division=0),      4),
            "Rappel":     round(recall_score(y_test, y_pred,
                                             zero_division=0),         4),
            "F1":         round(f1_score(y_test, y_pred,
                                         zero_division=0),             4),
            "AP":         round(average_precision_score(y_test, y_prob), 4),
            "_y_prob":    y_prob,
            "_y_pred":    y_pred,
        })

    results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 2. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(trained_pipelines: dict, X_test, y_test, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipe in trained_pipelines.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, color=PALETTE.get(name, "gray"),
                label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aléatoire")
    ax.set_xlabel("Taux de faux positifs", fontsize=12)
    ax.set_ylabel("Taux de vrais positifs", fontsize=12)
    ax.set_title("Courbes ROC — Greffe de Moelle Osseuse\n(Test set)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "roc_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✔ Courbes ROC     : {path}")
    return path


def plot_pr_curves(trained_pipelines: dict, X_test, y_test, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipe in trained_pipelines.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, lw=2, color=PALETTE.get(name, "gray"),
                label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Rappel", fontsize=12)
    ax.set_ylabel("Précision", fontsize=12)
    ax.set_title("Courbes Précision-Rappel\n(Test set)", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "pr_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✔ Courbes P-R     : {path}")
    return path


def plot_metrics_radar(results: pd.DataFrame, out_dir: str):
    """Graphique en barres groupées des 4 métriques."""
    metrics = ["ROC-AUC", "Précision", "Rappel", "F1"]
    r = results[["Modèle"] + metrics].copy()

    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (_, row) in enumerate(r.iterrows()):
        vals   = [row[m] for m in metrics]
        offset = (i - len(r) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=row["Modèle"],
                      color=PALETTE.get(row["Modèle"], "gray"),
                      alpha=0.85, edgecolor="black")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Comparaison des métriques par modèle\n(Test set)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(out_dir, "metrics_comparison.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✔ Comparaison     : {path}")
    return path


def plot_confusion_matrices(trained_pipelines: dict,
                             results: pd.DataFrame,
                             X_test, y_test, out_dir: str):
    n = len(trained_pipelines)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1: axes = [axes]

    for ax, (_, row) in zip(axes, results.iterrows()):
        name   = row["Modèle"]
        y_pred = row["_y_pred"]
        cm     = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Échec", "Réussite"],
                    yticklabels=["Échec", "Réussite"], ax=ax,
                    linewidths=0.5, linecolor="gray")
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")

    fig.suptitle("Matrices de Confusion — Test set", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✔ Matrices conf.  : {path}")
    return path


def plot_feature_importance(trained_pipelines: dict,
                             X_test, out_dir: str):
    """Importance des variables pour les modèles arborescents."""
    tree_models = {k: v for k, v in trained_pipelines.items()
                   if k in ("RandomForest", "XGBoost", "LightGBM")}

    for name, pipe in tree_models.items():
        clf = pipe.named_steps["clf"]
        if not hasattr(clf, "feature_importances_"):
            continue
        importances = clf.feature_importances_
        prep = pipe.named_steps["prep"]
        try:
            feature_names = prep.get_feature_names_out()
        except Exception:
            feature_names = [f"feat_{i}" for i in range(len(importances))]

        idx = np.argsort(importances)[-20:][::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(idx)), importances[idx],
                color=PALETTE.get(name, "gray"), alpha=0.85)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([feature_names[i].replace("num__", "").replace("cat__", "")
                            for i in idx], fontsize=9)
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title(f"Importance des variables — {name}", fontsize=12)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        path = os.path.join(out_dir, f"feature_importance_{name.lower()}.png")
        plt.savefig(path, dpi=150); plt.close()
        print(f"  ✔ Importances     : {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. SÉLECTION DU MEILLEUR MODÈLE
# ══════════════════════════════════════════════════════════════════════════════

def select_best_model(results: pd.DataFrame,
                      trained_pipelines: dict,
                      criterion: str = "ROC-AUC") -> tuple:
    """
    Retourne (nom_du_meilleur_modèle, pipeline_correspondant).
    Critère par défaut : ROC-AUC.
    """
    best_row  = results.sort_values(criterion, ascending=False).iloc[0]
    best_name = best_row["Modèle"]
    print(f"\n  🏆 MEILLEUR MODÈLE : {best_name}")
    print(f"     ROC-AUC   = {best_row['ROC-AUC']:.4f}")
    print(f"     Précision = {best_row['Précision']:.4f}")
    print(f"     Rappel    = {best_row['Rappel']:.4f}")
    print(f"     F1        = {best_row['F1']:.4f}")

    # Sauvegarde du meilleur modèle séparément
    path = "models/best_model.pkl"
    os.makedirs("models", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(trained_pipelines[best_name], f)
    print(f"     Sauvegardé : {path}")

    return best_name, trained_pipelines[best_name]


# ══════════════════════════════════════════════════════════════════════════════
# 4. PRÉDICTION SUR UN NOUVEAU PATIENT
# ══════════════════════════════════════════════════════════════════════════════

def predict_new_patient(best_pipeline, patient_data: dict) -> dict:
    """
    Prédit la probabilité de réussite pour un nouveau patient.

    patient_data : dict avec les mêmes clés que les colonnes d'entraînement
                   (sans la colonne cible).
    """
    df_patient = pd.DataFrame([patient_data])
    prob = best_pipeline.predict_proba(df_patient)[0, 1]
    pred = int(prob >= 0.5)
    return {
        "probabilite_reussite_pct": round(prob * 100, 1),
        "prediction": "Réussite probable" if pred == 1 else "Échec probable",
        "classe":     pred,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. RAPPORT HTML
# ══════════════════════════════════════════════════════════════════════════════

def generate_html_report(results: pd.DataFrame,
                          best_name: str, out_dir: str):
    metrics_display = results[["Modèle", "ROC-AUC", "Précision", "Rappel", "F1", "AP"]].copy()
    table_html = metrics_display.to_html(index=False, float_format="{:.4f}".format,
                                          classes="table", border=0)
    best_row = results[results["Modèle"] == best_name].iloc[0]
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<title>Rapport — Greffe Moelle Osseuse</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; background:#f7f9fc; color:#222; }}
  h1   {{ color: #1a3a5c; border-bottom: 3px solid #4C72B0; padding-bottom:8px; }}
  h2   {{ color: #2c5282; margin-top:30px; }}
  .badge {{ display:inline-block; background:#4C72B0; color:white; border-radius:6px;
             padding:4px 14px; font-size:1.1em; margin:6px 0; }}
  .table {{ border-collapse:collapse; width:100%; background:white;
             box-shadow: 0 2px 8px rgba(0,0,0,.08); border-radius:8px; overflow:hidden; }}
  .table th {{ background:#1a3a5c; color:white; padding:10px 14px; text-align:left; }}
  .table td {{ padding:9px 14px; border-bottom:1px solid #e2e8f0; }}
  .table tr:hover td {{ background:#ebf4ff; }}
  img {{ max-width:800px; width:100%; border-radius:8px;
          box-shadow:0 2px 10px rgba(0,0,0,.12); margin:12px 0; }}
  .metric-card {{ display:inline-block; background:white; border-radius:10px;
                   padding:14px 24px; margin:8px; min-width:140px; text-align:center;
                   box-shadow:0 2px 8px rgba(0,0,0,.08); }}
  .metric-val {{ font-size:1.8em; font-weight:bold; color:#2b6cb0; }}
  .metric-lbl {{ font-size:.9em; color:#718096; }}
</style>
</head>
<body>
<h1>🩺 Prédiction de la Réussite de la Greffe de Moelle Osseuse Pédiatrique</h1>
<p>Rapport généré automatiquement • Modèles : RandomForest · XGBoost · SVM · LightGBM</p>

<h2>🏆 Meilleur Modèle</h2>
<p><span class="badge">{best_name}</span></p>
<div>
  <div class="metric-card"><div class="metric-val">{best_row['ROC-AUC']:.3f}</div>
    <div class="metric-lbl">ROC-AUC</div></div>
  <div class="metric-card"><div class="metric-val">{best_row['Précision']:.3f}</div>
    <div class="metric-lbl">Précision</div></div>
  <div class="metric-card"><div class="metric-val">{best_row['Rappel']:.3f}</div>
    <div class="metric-lbl">Rappel</div></div>
  <div class="metric-card"><div class="metric-val">{best_row['F1']:.3f}</div>
    <div class="metric-lbl">F1-Score</div></div>
</div>

<h2>📊 Tableau des Métriques (Test set)</h2>
{table_html}

<h2>📈 Visualisations</h2>
<h3>Courbes ROC</h3>
<img src="roc_curves.png" alt="Courbes ROC"/>
<h3>Courbes Précision-Rappel</h3>
<img src="pr_curves.png" alt="Courbes PR"/>
<h3>Comparaison des Métriques</h3>
<img src="metrics_comparison.png" alt="Métriques"/>
<h3>Matrices de Confusion</h3>
<img src="confusion_matrices.png" alt="Matrices de confusion"/>
<h3>Importance des Variables (RandomForest)</h3>
<img src="feature_importance_randomforest.png" alt="Importance variables"/>

<h2>🔬 Interprétation</h2>
<ul>
  <li><b>ROC-AUC</b> : capacité discriminante globale (1 = parfait, 0.5 = aléatoire).</li>
  <li><b>Précision</b> : parmi les patients prédits "réussite", quel % l'est vraiment.</li>
  <li><b>Rappel</b>   : parmi les vrais succès, quel % est correctement détecté.</li>
  <li><b>F1</b>       : moyenne harmonique Précision/Rappel — équilibre entre les deux.</li>
</ul>
<p><em>⚠️ Cet outil est une aide à la décision clinique, non un substitut au jugement médical.</em></p>
</body></html>"""
    path = os.path.join(out_dir, "rapport_greffe.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✔ Rapport HTML    : {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 6. POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OUT_DIR = "outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Données & entraînement ──────────────────────────────────────────
    df = generate_synthetic_data(n_samples=800)
    TARGET = "reussite_greffe"

    trained_pipelines, cv_scores, (X_train, X_test, y_train, y_test) = \
        train_all(df, TARGET)

    # ── 2. Évaluation ─────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ÉVALUATION SUR LE JEU DE TEST")
    print("═" * 60)

    results = evaluate_all(trained_pipelines, X_test, y_test)

    print("\n  Métriques — Test set :")
    print(results[["Modèle", "ROC-AUC", "Précision", "Rappel", "F1"]].to_string(index=False))

    # ── 3. Visualisations ─────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  GÉNÉRATION DES GRAPHIQUES")
    print("═" * 60)

    plot_roc_curves(trained_pipelines, X_test, y_test, OUT_DIR)
    plot_pr_curves(trained_pipelines, X_test, y_test, OUT_DIR)
    plot_metrics_radar(results, OUT_DIR)
    plot_confusion_matrices(trained_pipelines, results, X_test, y_test, OUT_DIR)
    plot_feature_importance(trained_pipelines, X_test, OUT_DIR)

    # ── 4. Meilleur modèle ────────────────────────────────────────────────
    best_name, best_pipeline = select_best_model(results, trained_pipelines)

    # ── 5. Rapport HTML ───────────────────────────────────────────────────
    generate_html_report(results, best_name, OUT_DIR)

    # ── 6. Exemple de prédiction pour un nouveau patient ──────────────────
    print("\n" + "═" * 60)
    print("  EXEMPLE — NOUVEAU PATIENT")
    print("═" * 60)

    nouveau_patient = {
        "age_years": 8, "poids_kg": 28.5, "sexe": "M",
        "type_maladie": "leucemie_aigue", "stade_maladie": "RC1",
        "compatibilite_hla": "10/10", "type_donneur": "frere_soeur",
        "taux_blastes_pct": 5.0, "hemoglobine_g_dl": 10.2,
        "plaquettes_1e9_l": 85, "creatinine_umol_l": 45,
        "cmv_statut_receveur": "negatif", "cmv_statut_donneur": "negatif",
        "infections_pre_greffe": 0,
        "type_conditionnement": "myeloablatif",
        "source_cellules": "moelle_osseuse",
        "prophylaxie_gvhd": "ciclosporine_mtx",
    }

    pred = predict_new_patient(best_pipeline, nouveau_patient)
    print(f"\n  Patient : enfant 8 ans, leucémie aiguë RC1, donneur fratrie HLA 10/10")
    print(f"  → Probabilité de réussite : {pred['probabilite_reussite_pct']} %")
    print(f"  → Décision                : {pred['prediction']}")
    print("\n  ✅ Évaluation complète. Ouvrez outputs/rapport_greffe.html")