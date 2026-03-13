import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_path)
sys.path.insert(0, project_root)

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

st.set_page_config(
    page_title="Aide à la Décision Médicale — Greffe Moelle Osseuse",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.8rem 2.5rem;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.main-header h1 { font-family:'DM Serif Display',serif; font-size:1.9rem; margin:0; color:#0f172a; }
.main-header p  { margin:0.3rem 0 0; color:#64748b; font-size:0.88rem; font-weight:300; }
.header-accent  { width:5px; height:60px; border-radius:4px;
                  background:linear-gradient(180deg,#60a5fa,#818cf8); flex-shrink:0; }

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    color: #334155;
    margin: 1.4rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1.5px solid #e2e8f0;
}

div[data-testid="stForm"] {
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    background: transparent !important;
}

.form-title    { text-align:center; font-family:'DM Serif Display',serif;
                 font-size:1.2rem; color:#1e1b4b; margin-bottom:0.1rem; font-weight:600; }
.form-subtitle { text-align:center; font-size:0.8rem; color:#818cf8; margin-bottom:1.2rem; }

div.stButton > button {
    background: linear-gradient(135deg, #60a5fa, #818cf8);
    color: white; border: none; border-radius: 10px;
    padding: 0.65rem 2rem; font-size: 0.95rem; font-weight: 600;
    font-family: 'DM Sans', sans-serif; transition: opacity 0.2s; width: 100%;
}
div.stButton > button:hover { opacity: 0.88; }

.warning-box {
    background: #fff7ed; border: 1px solid #fed7aa; border-radius: 10px;
    padding: 0.8rem 1.2rem; margin-bottom: 0.8rem;
    font-size: 0.83rem; color: #9a3412;
}

.result-card   { border-radius:14px; padding:1.5rem 2rem; margin-top:1rem; text-align:center; }
.result-high   { background:#f0fdf4; border:2px solid #22c55e; }
.result-medium { background:#fffbeb; border:2px solid #f59e0b; }
.result-low    { background:#fff1f2; border:2px solid #f43f5e; }
.result-score  { font-family:'DM Serif Display',serif; font-size:3.5rem; font-weight:700; margin:0; }
.score-high    { color:#16a34a; }
.score-medium  { color:#d97706; }
.score-low     { color:#e11d48; }
.result-label  { font-size:1rem; font-weight:500; color:#64748b; margin-bottom:0.5rem; }
.result-interp { margin-top:0.8rem; font-size:0.93rem; color:#475569; line-height:1.7; }

.pipeline-box {
    background:#f0f7ff; border:1px solid #bfdbfe; border-radius:10px;
    padding:0.9rem 1.2rem; margin-top:1rem; font-size:0.82rem;
    color:#1e40af; line-height:1.8;
}
.pipeline-box strong { color:#1d4ed8; }

.shap-card {
    background:#f8fafc; border:1px solid #e2e8f0;
    border-radius:14px; padding:1.4rem 1.8rem; margin-top:1rem;
}
.shap-title    { font-family:'DM Serif Display',serif; font-size:1.2rem; color:#0f172a; margin-bottom:0.2rem; }
.shap-subtitle { font-size:0.82rem; color:#94a3b8; margin-bottom:0.8rem; }
.shap-legend-title { font-size:0.85rem; font-weight:600; color:#334155; margin-bottom:0.8rem; }
.shap-item     { display:flex; align-items:flex-start; gap:10px; margin-bottom:0.9rem; }
.shap-dot      { width:11px; height:11px; border-radius:50%; flex-shrink:0; margin-top:3px; }
.shap-dot-pos  { background:#6096e0; }
.shap-dot-neg  { background:#e07070; }
.shap-item-label { font-size:0.78rem; font-weight:600; color:#334155; }
.shap-item-desc  { font-size:0.74rem; color:#64748b; line-height:1.5; margin-top:2px; }
.shap-note {
    background:#eff6ff; border-left:3px solid #93c5fd; border-radius:4px;
    padding:0.6rem 0.8rem; font-size:0.74rem; color:#1e40af; line-height:1.6; margin-top:0.8rem;
}
</style>
""", unsafe_allow_html=True)

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

st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
st.markdown('<p class="form-title">📋 Données Cliniques du Dossier</p>', unsafe_allow_html=True)
st.markdown('<p class="form-subtitle">Renseignez toutes les informations disponibles pour obtenir la prédiction</p>', unsafe_allow_html=True)

with st.form("patient_data"):

    st.markdown('<p class="section-title">👤 Receveur (Patient)</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        recipient_age = st.number_input(
            "Âge du receveur (années)", min_value=0, max_value=20, value=8, step=1,
            help="Entre 0 et 20 ans")
    with c2:
        recipient_gender = st.selectbox("Sexe du receveur", ["Féminin", "Masculin"])
    with c3:
        recipient_body_mass_str = st.text_input(
            "Poids du receveur (kg)", value="30",
            help="Entrez un nombre entre 3 et 150 kg")

    c4, c5, c6 = st.columns(3, gap="medium")
    with c4:
        recipient_ABO = st.selectbox("Groupe sanguin (receveur)", ["0", "A", "B", "AB"])
    with c5:
        recipient_rh  = st.selectbox("Rhésus (receveur)", ["Positif (+)", "Négatif (−)"])
    with c6:
        recipient_CMV = st.selectbox("CMV receveur", ["Absent", "Présent"],
                                     help="Statut sérologique Cytomégalovirus")

    st.markdown('<p class="section-title">🧬 Donneur</p>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3, gap="medium")
    with d1:
        donor_age = st.number_input(
            "Âge du donneur (années)", min_value=18, max_value=70, value=30, step=1,
            help="Un donneur < 35 ans est un facteur favorable")
    with d2:
        donor_ABO = st.selectbox("Groupe sanguin (donneur)", ["0", "A", "B", "AB"])
    with d3:
        donor_CMV = st.selectbox("CMV donneur", ["Absent", "Présent"])

    st.markdown('<p class="section-title">🔬 Compatibilité Immunologique</p>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3, gap="medium")
    with i1:
        HLA_match = st.selectbox("Compatibilité HLA", ["10/10", "9/10", "8/10", "7/10"],
                                 help="10/10 = compatibilité parfaite → pronostic favorable")
    with i2:
        HLA_mismatch = st.number_input("Nb antigènes HLA différents", min_value=0, max_value=3, value=0, step=1)
    with i3:
        ABO_match = st.selectbox("Compatibilité ABO", ["Compatible", "Incompatible"])

    st.markdown('<p class="section-title">🏥 Maladie & Protocole</p>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3, gap="medium")
    with p1:
        disease = st.selectbox("Diagnostic", ["ALL", "AML", "CML", "MDS", "SAA", "Fanconi", "Autre"],
                               help="Pathologie traitée par la greffe")
    with p2:
        risk_group = st.selectbox("Groupe de risque", ["Faible", "Élevé"])
    with p3:
        stem_cell_source = st.selectbox("Source des cellules souches",
                                        ["Moelle Osseuse", "Sang Périphérique"])

    p4, p5 = st.columns([1, 2], gap="medium")
    with p4:
        tx_post_relapse = st.selectbox("Greffe post-rechute ?", ["Non", "Oui"])
    with p5:
        CD34_str = st.text_input(
            "Dose CD34+ (×10⁶ cellules/kg)", value="3.0",
            help="Dose de cellules souches CD34+. Ex : 3.5 — variable #1 selon la littérature")

    st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
    submit = st.form_submit_button("⚡ Calculer la probabilité de succès")

if submit:

    # ── Validation des champs texte libres ──
    errors = []

    try:
        recipient_body_mass = float(recipient_body_mass_str.replace(",", ".").strip())
        if not (3.0 <= recipient_body_mass <= 150.0):
            errors.append("Le poids du receveur doit être compris entre 3 et 150 kg.")
    except ValueError:
        errors.append(f"Poids du receveur invalide : « {recipient_body_mass_str} » — veuillez entrer un nombre (ex : 28.5).")
        recipient_body_mass = None

    try:
        CD34 = float(CD34_str.replace(",", ".").strip())
        if not (0.1 <= CD34 <= 30.0):
            errors.append("La dose CD34+ doit être comprise entre 0.1 et 30 ×10⁶/kg.")
    except ValueError:
        errors.append(f"Dose CD34+ invalide : « {CD34_str} » — veuillez entrer un nombre (ex : 3.5).")
        CD34 = None

    if errors:
        for err in errors:
            st.markdown(f'<div class="warning-box">⚠️ {err}</div>', unsafe_allow_html=True)
        st.stop()

    # ── Alertes cliniques (non bloquantes) ──
    warnings = []
    if recipient_age < 10:
        warnings.append("🔴 Receveur pédiatrique critique (< 10 ans) — protocole de conditionnement adapté recommandé.")
    if donor_age >= 35:
        warnings.append("🟡 Âge du donneur ≥ 35 ans — facteur moins favorable selon la littérature.")
    if donor_CMV == "Présent" and recipient_CMV == "Absent":
        warnings.append("🟠 Combinaison CMV donneur+ / receveur− — risque de réactivation CMV post-greffe élevé.")
    if HLA_match in ["8/10", "7/10"]:
        warnings.append("🟡 Compatibilité HLA incomplète — risque de GvHD accru.")
    if tx_post_relapse == "Oui":
        warnings.append("🟠 Greffe post-rechute — pronostic plus réservé.")

    if warnings:
        warn_html = "".join(f"<div style='margin-bottom:4px'>{w}</div>" for w in warnings)
        st.markdown(f'<div class="warning-box">{warn_html}</div>', unsafe_allow_html=True)

    if not HAS_SRC:
        st.error(f"Module de traitement introuvable — {import_error}")
        st.stop()

    try:
        input_df = pd.DataFrame({
            'recipient_age':         [recipient_age],
            'recipient_gender':      [recipient_gender],
            'recipient_body_mass':   [recipient_body_mass],
            'recipient_ABO':         [recipient_ABO],
            'recipient_rh':          [recipient_rh],
            'recipient_CMV':         [recipient_CMV],
            'donor_age':             [donor_age],
            'donor_ABO':             [donor_ABO],
            'donor_CMV':             [donor_CMV],
            'HLA_match':             [HLA_match],
            'HLA_mismatch':          [HLA_mismatch],
            'ABO_match':             [ABO_match],
            'disease':               [disease],
            'risk_group':            [risk_group],
            'stem_cell_source':      [stem_cell_source],
            'tx_post_relapse':       [tx_post_relapse],
            'CD34_x1e6_per_kg':      [CD34],
        })

        input_df   = handle_missing_values(input_df)
        input_df   = encode_features(input_df)
        input_df   = handle_outliers(input_df, target_col='__none__')
        data_ready = optimize_memory(input_df)
        if '__none__' in data_ready.columns:
            data_ready = data_ready.drop(columns=['__none__'])

        if HAS_MODEL:
            proba = model.predict_proba(data_ready)[0][1] * 100
        else:
            np.random.seed(int(recipient_age * 7 + (recipient_body_mass or 30) * 3 + donor_age))
            proba = np.random.uniform(45, 97)

        if proba >= 75:
            card_cls, score_cls, niveau, emoji_res = "result-high",   "score-high",   "Favorable",   "✅"
        elif proba >= 50:
            card_cls, score_cls, niveau, emoji_res = "result-medium", "score-medium", "Modéré",      "⚠️"
        else:
            card_cls, score_cls, niveau, emoji_res = "result-low",    "score-low",    "Défavorable", "❌"

        source_txt = "moelle osseuse" if stem_cell_source == "Moelle Osseuse" else "sang périphérique"
        hla_txt    = f"compatibilité HLA {HLA_match}"
        age_ctx    = "pédiatrique critique" if recipient_age < 10 else f"{recipient_age} ans"

        if proba >= 75:
            interp = (f"Le profil de ce receveur ({age_ctx}, {recipient_body_mass} kg, {disease}) avec {hla_txt} "
                      f"et une greffe à partir de {source_txt} présente des indicateurs globalement favorables. "
                      f"Probabilité estimée : <strong>{proba:.1f}%</strong>.")
        elif proba >= 50:
            interp = (f"Résultats mitigés pour ce profil ({age_ctx}, {disease}, {hla_txt}). "
                      f"Probabilité estimée : <strong>{proba:.1f}%</strong>. "
                      f"Une surveillance rapprochée est recommandée.")
        else:
            interp = (f"Facteurs de risque significatifs détectés ({age_ctx}, {disease}, {hla_txt}). "
                      f"Probabilité estimée : <strong>{proba:.1f}%</strong>. "
                      f"Consultation pluridisciplinaire fortement conseillée.")

        st.markdown(f"""
        <div class="result-card {card_cls}">
            <div class="result-label">Probabilité de succès estimée</div>
            <p class="result-score {score_cls}">{proba:.1f}%</p>
            <div style="font-size:1.05rem;font-weight:600;margin-top:4px;">{emoji_res} Pronostic {niveau}</div>
            <div class="result-interp">{interp}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="pipeline-box">
            <strong>🔧 Pipeline de traitement :</strong>
            handle_missing_values() → encode_features() → handle_outliers() → optimize_memory()
            &nbsp;·&nbsp; <strong>Shape :</strong> {data_ready.shape[1]} features
        </div>
        """, unsafe_allow_html=True)

        # ── SHAP ──
        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="shap-card">', unsafe_allow_html=True)
            st.markdown('<p class="shap-title">📊 Explicabilité SHAP</p>', unsafe_allow_html=True)
            st.markdown('<p class="shap-subtitle">Contribution de chaque paramètre clinique à la prédiction finale</p>', unsafe_allow_html=True)

            feature_names = list(data_ready.columns)

            if HAS_SHAP and HAS_MODEL:
                explainer   = shap.Explainer(model, data_ready)
                shap_values = explainer(data_ready).values[0]
            else:
                np.random.seed(int(recipient_age * 7 + (recipient_body_mass or 30) * 3))
                raw = np.random.randn(len(feature_names))
                shap_values = raw / (np.abs(raw).sum() + 1e-9) * (proba - 50) / 12

            label_map = {
                'recipient_age':       'Âge receveur',
                'recipient_body_mass': 'Poids',
                'recipient_gender':    'Sexe',
                'recipient_ABO':       'Groupe ABO',
                'recipient_rh':        'Rhésus',
                'recipient_CMV':       'CMV receveur',
                'donor_age':           'Âge donneur',
                'donor_ABO':           'ABO donneur',
                'donor_CMV':           'CMV donneur',
                'HLA_match':           'HLA match',
                'HLA_mismatch':        'HLA diff.',
                'ABO_match':           'Compat. ABO',
                'disease':             'Maladie',
                'risk_group':          'Risque',
                'stem_cell_source':    'Source cellules',
                'tx_post_relapse':     'Post-rechute',
                'CD34_x1e6_per_kg':    'CD34+/kg',
            }
            labels = [next((v for k, v in label_map.items() if k in f), f[:12]) for f in feature_names]

            order        = np.argsort(np.abs(shap_values))[::-1][:8]  # top 8 features
            shap_sorted  = shap_values[order]
            label_sorted = [labels[i] for i in order]

            COLOR_POS = '#6096e0'
            COLOR_NEG = '#e07070'
            colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in shap_sorted]

            col_chart, col_legend = st.columns([3, 2], gap="large")

            with col_chart:
                n = len(label_sorted)
                fig, ax = plt.subplots(figsize=(max(4, n * 0.9), 3.6))
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
                            fontsize=7.5, fontweight='600', color='#334155')

                ax.set_xticks(x)
                ax.set_xticklabels(label_sorted, fontsize=8, color='#475569',
                                   fontweight='500', rotation=20, ha='right')
                ax.set_ylabel("Valeur SHAP", fontsize=8, color='#94a3b8', labelpad=6)
                ax.yaxis.grid(True, linestyle=':', alpha=0.35, color='#e2e8f0', zorder=0)
                ax.set_axisbelow(True)
                for spine in ['top', 'right', 'left']:
                    ax.spines[spine].set_visible(False)
                ax.spines['bottom'].set_color('#f1f5f9')
                ax.tick_params(axis='y', colors='#cbd5e1', labelsize=7)
                ax.tick_params(axis='x', length=0)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin * 1.35, ymax * 1.35)
                plt.tight_layout(pad=1.0)
                st.pyplot(fig)
                plt.close(fig)

            with col_legend:
                desc_map = {
                    'Âge receveur':    "Influence la tolérance au conditionnement et la capacité de récupération médullaire.",
                    'Poids':           "Détermine les doses de chimiothérapie et le ratio cellules souches / kg.",
                    'Sexe':            "Le sexe peut influencer les réponses immunologiques post-greffe.",
                    'Groupe ABO':      "Compatibilité ABO receveur — impact sur le risque de rejet.",
                    'Rhésus':          "Compatibilité rhésus entre donneur et receveur.",
                    'CMV receveur':    "Statut CMV — détermine le risque de réactivation post-greffe.",
                    'Âge donneur':     "Un donneur < 35 ans est associé à de meilleurs résultats.",
                    'ABO donneur':     "Groupe sanguin du donneur — compatibilité ABO.",
                    'CMV donneur':     "CMV donneur+ / receveur− = combinaison à risque élevé.",
                    'HLA match':       "10/10 = prédicteur fort de succès. Chaque mismatch augmente le risque de GvHD.",
                    'HLA diff.':       "Nombre d'antigènes HLA incompatibles — corrélé au risque de rejet.",
                    'Compat. ABO':     "Incompatibilité ABO → risque de réaction hémolytique.",
                    'Maladie':         "La pathologie détermine le protocole et le pronostic intrinsèque.",
                    'Risque':          "Classification du risque de la maladie au moment de la greffe.",
                    'Source cellules': "Moelle vs sang périphérique — impact sur la prise de greffe et le risque GvHD.",
                    'Post-rechute':    "Greffe après rechute — pronostic plus réservé.",
                    'CD34+/kg':        "Dose de cellules souches — variable #1 selon la littérature : doses élevées prolongent la survie.",
                }

                st.markdown('<p class="shap-legend-title">📌 Top facteurs influents</p>', unsafe_allow_html=True)

                for lbl, val in zip(label_sorted, shap_sorted):
                    dot_cls   = "shap-dot-pos" if val >= 0 else "shap-dot-neg"
                    direction = "↑ favorable" if val >= 0 else "↓ défavorable"
                    color_val = '#6096e0' if val >= 0 else '#e07070'
                    desc = desc_map.get(lbl, "Paramètre clinique.")
                    st.markdown(f"""
                    <div class="shap-item">
                      <div class="shap-dot {dot_cls}"></div>
                      <div>
                        <div class="shap-item-label">{lbl}
                          <span style="font-weight:400;color:{color_val}">({direction}, {val:+.3f})</span>
                        </div>
                        <div class="shap-item-desc">{desc}</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div class="shap-note">
                  <strong>Comment lire ?</strong><br>
                  Barre <strong style="color:#6096e0">bleue</strong> = facteur qui <em>augmente</em>
                  la probabilité de succès. Barre <strong style="color:#e07070">rouge</strong> = facteur
                  qui la <em>diminue</em>. Seuls les 8 facteurs les plus influents sont affichés.
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        if not HAS_MODEL:
            st.info("⏳ Dès que le fichier `src/model.pkl` sera disponible, "
                    "le vrai score et les vraies valeurs SHAP seront utilisés automatiquement.")

    except Exception as e:
        st.error(f"Erreur lors du traitement : {e}")
        st.exception(e)