import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ═══════════════════════════════════════════════════════
# CHEMINS
# ═══════════════════════════════════════════════════════
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_path)
sys.path.insert(0, project_root)

# ═══════════════════════════════════════════════════════
# IMPORTATIONS — MEMBRE 2
# ═══════════════════════════════════════════════════════
HAS_SRC = False
import_error = ""
try:
    from src.data_processing import (
        encode_features,
        handle_missing_values,
        handle_outliers,
        optimize_memory,
    )
    HAS_SRC = True
except ImportError as e:
    import_error = str(e)

# MODÈLE — MEMBRE 3
HAS_MODEL = False
model = None
try:
    import joblib
    model_path = os.path.join(project_root, 'src', 'model.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        HAS_MODEL = True
except Exception:
    pass

HAS_SHAP = False
try:
    import shap
    HAS_SHAP = True
except ImportError:
    pass

# ═══════════════════════════════════════════════════════
# CONFIG PAGE
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Aide à la Décision Médicale — Greffe Moelle Osseuse",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════
# CSS GLOBAL
# ═══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Header principal ── */
.main-header {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.8rem 2.5rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.main-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    margin: 0;
    color: #0f172a;
}
.main-header p { margin: 0.3rem 0 0; color: #64748b; font-size: 0.88rem; font-weight: 300; }
.header-accent {
    width: 5px; height: 60px; border-radius: 4px;
    background: linear-gradient(180deg, #60a5fa, #818cf8);
    flex-shrink: 0;
}

/* ── Badges membres ── */
.badge-ok   { background:#dcfce7; color:#15803d; padding:4px 12px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.badge-err  { background:#fee2e2; color:#dc2626; padding:4px 12px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.badge-warn { background:#fef9c3; color:#92400e; padding:4px 12px; border-radius:20px; font-size:0.78rem; font-weight:600; }

/* ── Formulaire : supprime le rectangle natif Streamlit ── */
div[data-testid="stForm"] {
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    background: transparent !important;
}

/* ── Titre formulaire centré ── */
.form-title {
    text-align: center;
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: #1e1b4b;
    margin-bottom: 0.1rem;
    font-weight: 600;
}
.form-subtitle {
    text-align: center;
    font-size: 0.8rem;
    color: #818cf8;
    margin-bottom: 1.2rem;
}

/* ── Bouton calcul ── */
div.stButton > button {
    background: linear-gradient(135deg, #60a5fa, #818cf8);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 2rem;
    font-size: 0.95rem;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    transition: opacity 0.2s;
    width: 100%;
}
div.stButton > button:hover { opacity: 0.88; }

/* ── Carte résultat ── */
.result-card { border-radius: 14px; padding: 1.5rem 2rem; margin-top: 1rem; text-align: center; }
.result-high   { background: #f0fdf4; border: 2px solid #22c55e; }
.result-medium { background: #fffbeb; border: 2px solid #f59e0b; }
.result-low    { background: #fff1f2; border: 2px solid #f43f5e; }
.result-score  { font-family: 'DM Serif Display', serif; font-size: 3.5rem; font-weight: 700; margin: 0; }
.score-high    { color: #16a34a; }
.score-medium  { color: #d97706; }
.score-low     { color: #e11d48; }
.result-label  { font-size: 1rem; font-weight: 500; color: #64748b; margin-bottom: 0.5rem; }
.result-interp { margin-top: 0.8rem; font-size: 0.93rem; color: #475569; line-height: 1.7; }

/* ── Pipeline box ── */
.pipeline-box {
    background: #f0f7ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-top: 1rem;
    font-size: 0.82rem;
    color: #1e40af;
    line-height: 1.8;
}
.pipeline-box strong { color: #1d4ed8; }

/* ── Carte SHAP ── */
.shap-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    margin-top: 1rem;
}
.shap-title    { font-family:'DM Serif Display',serif; font-size:1.2rem; color:#0f172a; margin-bottom:0.2rem; }
.shap-subtitle { font-size:0.82rem; color:#94a3b8; margin-bottom:0.8rem; }

/* ── Légende SHAP ── */
.shap-legend-title { font-size:0.85rem; font-weight:600; color:#334155; margin-bottom:0.8rem; }
.shap-item { display:flex; align-items:flex-start; gap:10px; margin-bottom:0.9rem; }
.shap-dot  { width:11px; height:11px; border-radius:50%; flex-shrink:0; margin-top:3px; }
.shap-dot-pos { background:#6096e0; }
.shap-dot-neg { background:#e07070; }
.shap-item-label { font-size:0.78rem; font-weight:600; color:#334155; }
.shap-item-desc  { font-size:0.74rem; color:#64748b; line-height:1.5; margin-top:2px; }
.shap-note {
    background:#eff6ff; border-left:3px solid #93c5fd; border-radius:4px;
    padding:0.6rem 0.8rem; font-size:0.74rem; color:#1e40af;
    line-height:1.6; margin-top:0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <div class="header-accent"></div>
  <span style="font-size:2.6rem">🩺</span>
  <div>
    <h1>Prédiction de Succès — Greffe de Moelle Osseuse</h1>
    <p>Outil d'aide à la décision clinique · Groupe 24 · Coding Week 2026 · Ecole Centrale Casablanca</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Badges statut membres ──
col_b1, col_b2, col_b3 = st.columns([3, 3, 4])
with col_b1:
    if HAS_SRC:
        st.markdown('<span class="badge-ok">✓ Membre 2 · data_processing connecté</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge-err">✗ Membre 2 introuvable — {import_error}</span>', unsafe_allow_html=True)
with col_b2:
    if HAS_MODEL:
        st.markdown('<span class="badge-ok">✓ Membre 3 · modèle chargé</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warn">⏳ Membre 3 · modèle en attente</span>', unsafe_allow_html=True)

st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# FORMULAIRE
# Titre centré en HTML (hors form pour éviter tout rectangle),
# puis le st.form natif dont on a supprimé la bordure en CSS.
# ═══════════════════════════════════════════════════════
st.markdown('<p class="form-title">👤 Informations sur le Patient</p>', unsafe_allow_html=True)
st.markdown('<p class="form-subtitle">Renseignez les données cliniques pour obtenir la prédiction</p>', unsafe_allow_html=True)

with st.form("patient_data"):
    col1, col2 = st.columns(2, gap="large")
    with col1:
        age    = st.number_input("Âge du patient (années)", min_value=0, max_value=18, value=10, step=1)
        source = st.selectbox("Source des cellules souches",
                              ["Moelle Osseuse", "Sang Périphérique"],
                              help="Origine du greffon utilisé")
    with col2:
        poids   = st.number_input("Poids du patient (kg)", min_value=5.0, max_value=150.0, value=30.0, step=0.5)
        maladie = st.selectbox("Type de maladie",
                               ["ALL", "AML", "Non-malignant"],
                               help="Pathologie traitée par la greffe")
    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
    submit = st.form_submit_button("⚡ Calculer la probabilité de succès")

# ═══════════════════════════════════════════════════════
# TRAITEMENT
# ═══════════════════════════════════════════════════════
if submit:

    if not HAS_SRC:
        st.error("Impossible de calculer : le module 'src' (Membre 2) est introuvable.")
        st.stop()

    try:
        # ── Construction DataFrame patient ──
        input_df = pd.DataFrame({
            'Donorage':       [age],
            'Stemcellsource': [source],
            'Donorpoids':     [poids],
            'Disease':        [maladie],
        })

        # ── Pipeline Membre 2 (ordre impératif) ──
        input_df   = handle_missing_values(input_df)
        input_df   = encode_features(input_df)          # en premier !
        input_df   = handle_outliers(input_df, target_col='__none__')
        data_ready = optimize_memory(input_df)
        if '__none__' in data_ready.columns:
            data_ready = data_ready.drop(columns=['__none__'])

        # ── Prédiction ──
        if HAS_MODEL:
            proba = model.predict_proba(data_ready)[0][1] * 100
        else:
            np.random.seed(int(age * 7 + poids * 3))
            proba = np.random.uniform(55, 97)

        # ── Niveau pronostic ──
        if proba >= 75:
            card_cls, score_cls, niveau, emoji = "result-high",   "score-high",   "Favorable",   "✅"
        elif proba >= 50:
            card_cls, score_cls, niveau, emoji = "result-medium", "score-medium", "Modéré",      "⚠️"
        else:
            card_cls, score_cls, niveau, emoji = "result-low",    "score-low",    "Défavorable", "❌"

        source_txt = "moelle osseuse" if source == "Moelle Osseuse" else "sang périphérique"
        if proba >= 75:
            interp = (f"Le profil de ce patient ({age} ans, {poids} kg, {maladie}) présente des indicateurs "
                      f"globalement favorables pour une greffe à partir de {source_txt}. "
                      f"Probabilité estimée : <strong>{proba:.1f}%</strong>. "
                      f"Les facteurs cliniques sont cohérents avec les cas de succès du dataset d'entraînement.")
        elif proba >= 50:
            interp = (f"Résultats mitigés pour ce profil ({age} ans, {poids} kg, {maladie}) — greffe {source_txt}. "
                      f"Probabilité estimée : <strong>{proba:.1f}%</strong>. "
                      f"Une surveillance rapprochée et des examens complémentaires sont recommandés.")
        else:
            interp = (f"Facteurs de risque significatifs détectés ({age} ans, {poids} kg, {maladie}). "
                      f"Probabilité estimée : <strong>{proba:.1f}%</strong>. "
                      f"Consultation pluridisciplinaire fortement conseillée avant de procéder.")

        # ── Carte résultat ──
        st.markdown(f"""
        <div class="result-card {card_cls}">
            <div class="result-label">Probabilité de succès estimée</div>
            <p class="result-score {score_cls}">{proba:.1f}%</p>
            <div style="font-size:1.05rem;font-weight:600;margin-top:4px;">{emoji} Pronostic {niveau}</div>
            <div class="result-interp">{interp}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Pipeline info ──
        st.markdown(f"""
        <div class="pipeline-box">
            <strong>🔧 Pipeline Membre 2 :</strong>
            handle_missing_values() → encode_features() → handle_outliers() → optimize_memory()
            &nbsp;·&nbsp; <strong>Shape :</strong> {data_ready.shape[1]} features
        </div>
        """, unsafe_allow_html=True)

        # ═══════════════════════════════════════════════
        # SHAP — carte avec 2 colonnes internes (matplotlib | légende HTML)
        # La carte .shap-card est fermée APRÈS les colonnes Streamlit via CSS,
        # donc on utilise st.container() pour grouper proprement.
        # ═══════════════════════════════════════════════
        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="shap-card">', unsafe_allow_html=True)
            st.markdown('<p class="shap-title">📊 Explicabilité SHAP</p>', unsafe_allow_html=True)
            st.markdown('<p class="shap-subtitle">Contribution de chaque paramètre clinique à la prédiction finale</p>', unsafe_allow_html=True)

            # ── Calcul valeurs SHAP ──
            feature_names = list(data_ready.columns)

            if HAS_SHAP and HAS_MODEL:
                explainer   = shap.Explainer(model, data_ready)
                shap_values = explainer(data_ready).values[0]
            else:
                np.random.seed(int(age * 7 + poids * 3))
                raw = np.random.randn(len(feature_names))
                shap_values = raw / (np.abs(raw).sum() + 1e-9) * (proba - 50) / 12

            label_map = {
                'Donorage': 'Âge', 'Donorpoids': 'Poids',
                'Stemcellsource': 'Source', 'Disease': 'Maladie',
            }
            labels = [next((v for k, v in label_map.items() if k in f), f[:10]) for f in feature_names]

            order        = np.argsort(np.abs(shap_values))[::-1]
            shap_sorted  = shap_values[order]
            label_sorted = [labels[i] for i in order]

            COLOR_POS = '#6096e0'  # bleu doux
            COLOR_NEG = '#e07070'  # rouge doux
            colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in shap_sorted]

            # ── 2 colonnes : graphique | interprétation ──
            col_chart, col_legend = st.columns([3, 2], gap="large")

            with col_chart:
                n = len(label_sorted)
                fig, ax = plt.subplots(figsize=(max(3.5, n * 1.4), 3.4))
                fig.patch.set_facecolor('#f8fafc')
                ax.set_facecolor('#f8fafc')

                x = np.arange(n)
                ax.bar(x, shap_sorted, color=colors, width=0.28,
                       edgecolor='none', zorder=3, alpha=0.9)
                ax.axhline(0, color='#94a3b8', linewidth=0.8, zorder=5)

                max_abs = max(abs(shap_sorted)) if len(shap_sorted) > 0 else 1
                for xi, val in zip(x, shap_sorted):
                    offset = max_abs * 0.08
                    va   = 'bottom' if val >= 0 else 'top'
                    ypos = val + (offset if val >= 0 else -offset)
                    ax.text(xi, ypos, f'{val:+.3f}', ha='center', va=va,
                            fontsize=8, fontweight='600', color='#334155')

                ax.set_xticks(x)
                ax.set_xticklabels(label_sorted, fontsize=9.5, color='#475569', fontweight='500')
                ax.set_ylabel("Valeur SHAP", fontsize=8, color='#94a3b8', labelpad=6)
                ax.yaxis.grid(True, linestyle=':', alpha=0.35, color='#e2e8f0', zorder=0)
                ax.set_axisbelow(True)
                for spine in ['top', 'right', 'left']:
                    ax.spines[spine].set_visible(False)
                ax.spines['bottom'].set_color('#f1f5f9')
                ax.tick_params(axis='y', colors='#cbd5e1', labelsize=7)
                ax.tick_params(axis='x', length=0)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin * 1.3, ymax * 1.3)
                plt.tight_layout(pad=1.0)
                st.pyplot(fig)
                plt.close(fig)

            with col_legend:
                desc_map = {
                    'Âge':     ("Âge du patient",
                                "L'âge influence la tolérance au conditionnement et la capacité de récupération médullaire."),
                    'Poids':   ("Poids corporel",
                                "Le poids détermine les doses de chimiothérapie et le ratio cellules souches / kg."),
                    'Source':  ("Source des cellules",
                                "Moelle osseuse vs sang périphérique — impact sur la prise de greffe et le risque GvHD."),
                    'Maladie': ("Type de maladie",
                                "La pathologie (ALL, AML, Non-malignant) détermine le protocole et le pronostic intrinsèque."),
                }

                st.markdown('<p class="shap-legend-title">📌 Interprétation des facteurs</p>', unsafe_allow_html=True)

                for lbl, val in zip(label_sorted, shap_sorted):
                    dot_cls   = "shap-dot-pos" if val >= 0 else "shap-dot-neg"
                    direction = "↑ Impact positif" if val >= 0 else "↓ Impact négatif"
                    color_val = '#6096e0' if val >= 0 else '#e07070'
                    full_name, desc = desc_map.get(lbl, (lbl, "Paramètre clinique du patient."))
                    st.markdown(f"""
                    <div class="shap-item">
                      <div class="shap-dot {dot_cls}"></div>
                      <div>
                        <div class="shap-item-label">{full_name}
                          <span style="font-weight:400;color:{color_val}">
                            ({direction}, {val:+.3f})
                          </span>
                        </div>
                        <div class="shap-item-desc">{desc}</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div class="shap-note">
                  <strong>Comment lire ce graphique ?</strong><br>
                  Une barre <strong style="color:#6096e0">bleue</strong> signifie que ce facteur
                  <em>augmente</em> la probabilité de succès par rapport à la moyenne.
                  Une barre <strong style="color:#e07070">rouge</strong> la <em>diminue</em>.
                  L'amplitude indique l'intensité de l'effet.
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)  # ferme .shap-card

        if not HAS_MODEL:
            st.info("⏳ **Membre 3** — Dès que `src/model.pkl` sera disponible, "
                    "le vrai score et les vraies valeurs SHAP seront utilisés automatiquement.")

    except Exception as e:
        st.error(f"Erreur lors du traitement : {e}")
        st.exception(e)