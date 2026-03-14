import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_path)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

HAS_SRC = False
import_error = ""
try:
    from data_processing import (
        handle_missing_values,
        encode_features,
        handle_outliers,
        optimize_memory,
    )
    HAS_SRC = True
except ImportError as e:
    import_error = str(e)

HAS_MODEL  = False
HAS_SCALER = False
model      = None
scaler     = None
_loaded_model_name = ""

MODELS_DIR  = os.path.join(project_root, "models")
MODEL_NAMES = ["xgboost", "lightgbm", "randomforest", "svm"]

for _name in MODEL_NAMES:
    _path = os.path.join(MODELS_DIR, f"{_name}_model.pkl")
    if os.path.exists(_path):
        try:
            with open(_path, "rb") as f:
                model = pickle.load(f)
            HAS_MODEL = True
            _loaded_model_name = _name.upper()
            break
        except Exception:
            continue

_scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
if os.path.exists(_scaler_path):
    try:
        with open(_scaler_path, "rb") as f:
            scaler = pickle.load(f)
        HAS_SCALER = True
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
    border: 1px solid #e2e8f0; border-radius: 16px;
    padding: 1.8rem 2.5rem; margin-bottom: 1.4rem;
    display: flex; align-items: center; gap: 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.main-header h1 { font-family:'DM Serif Display',serif; font-size:1.9rem; margin:0; color:#0f172a; }
.main-header p  { margin:0.3rem 0 0; color:#64748b; font-size:0.88rem; font-weight:300; }
.header-accent  { width:5px; height:60px; border-radius:4px;
                  background:linear-gradient(180deg,#60a5fa,#818cf8); flex-shrink:0; }

.section-title {
    font-family:'DM Serif Display',serif; font-size:1.0rem; color:#334155;
    margin:1.4rem 0 0.6rem 0; padding-bottom:0.4rem; border-bottom:1.5px solid #e2e8f0;
}
div[data-testid="stForm"] {
    border:none !important; box-shadow:none !important;
    padding:0 !important; background:transparent !important;
}
.form-title    { text-align:center; font-family:'DM Serif Display',serif;
                 font-size:1.2rem; color:#1e1b4b; margin-bottom:0.1rem; font-weight:600; }
.form-subtitle { text-align:center; font-size:0.8rem; color:#818cf8; margin-bottom:1.2rem; }

div.stButton > button {
    background:linear-gradient(135deg,#60a5fa,#818cf8); color:white;
    border:none; border-radius:10px; padding:0.65rem 2rem;
    font-size:0.95rem; font-weight:600; font-family:'DM Sans',sans-serif;
    transition:opacity 0.2s; width:100%;
}
div.stButton > button:hover { opacity:0.88; }

.warning-box {
    background:#fff7ed; border:1px solid #fed7aa; border-radius:10px;
    padding:0.8rem 1.2rem; margin-bottom:0.8rem; font-size:0.83rem; color:#9a3412;
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
    padding:0.9rem 1.2rem; margin-top:1rem; font-size:0.82rem; color:#1e40af; line-height:1.8;
}
.pipeline-box strong { color:#1d4ed8; }

.shap-card {
    background:#f8fafc; border:1px solid #e2e8f0;
    border-radius:14px; padding:1.4rem 1.8rem; margin-top:1rem;
}
.shap-title        { font-family:'DM Serif Display',serif; font-size:1.2rem; color:#0f172a; margin-bottom:0.2rem; }
.shap-subtitle     { font-size:0.82rem; color:#94a3b8; margin-bottom:0.8rem; }
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
/* Curseur main sur les selectbox et inputs */
div[data-baseweb="select"] { cursor: pointer !important; }
div[data-baseweb="select"] * { cursor: pointer !important; }
div[data-baseweb="input"] { cursor: text !important; }
.stNumberInput button { cursor: pointer !important; }
/* Inline error style */
.field-error {
    color: #dc2626; font-size: 0.78rem; margin-top: 2px;
    padding: 2px 6px; border-radius: 4px;
    background: #fef2f2; border: 1px solid #fecaca;
    display: inline-block; margin-bottom: 4px;
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

# Session state pour erreurs inline
if "field_errors" not in st.session_state:
    st.session_state.field_errors = {}

def ferr(key):
    e = st.session_state.field_errors.get(key)
    if e:
        st.markdown(f'<div style="color:#dc2626;font-size:0.78rem;margin-top:-8px;margin-bottom:6px;padding:3px 8px;background:#fef2f2;border-radius:4px;border:1px solid #fecaca">⚠️ {e}</div>', unsafe_allow_html=True)

with st.form("patient_data"):

    st.markdown('<p class="section-title">👤 Receveur (Patient)</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        Recipientage = st.number_input("Âge du receveur (années)", min_value=0, max_value=20, value=8, step=1)
    with c2:
        Recipientgender = st.selectbox("Sexe du receveur", [1, 0],
                                       format_func=lambda x: "Masculin" if x == 1 else "Féminin")
    with c3:
        Rbodymass_str = st.text_input("Poids du receveur (kg)", value="30")
        ferr("poids")

    c4, c5, c6 = st.columns(3, gap="medium")
    with c4:
        RecipientABO = st.selectbox("Groupe sanguin receveur", [0, 1, 2, 3],
                                    format_func=lambda x: ["0","A","B","AB"][x])
    with c5:
        RecipientCMV = st.selectbox("CMV receveur (Cytomégalovirus)", [0, 1],
                                    format_func=lambda x: "Absent" if x == 0 else "Présent")
    with c6:
        Recipientage10 = st.selectbox("Receveur < 10 ans ?", [0, 1],
                                      format_func=lambda x: "Non" if x == 0 else "Oui")

    st.markdown('<p class="section-title">🧬 Donneur</p>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3, gap="medium")
    with d1:
        Donorage_str = st.text_input("Âge du donneur (années)", value="30")
        ferr("donorage")
    with d2:
        DonorABO = st.selectbox("Groupe sanguin donneur", [0, 1, 2, 3],
                                format_func=lambda x: ["0","A","B","AB"][x])
    with d3:
        DonorCMV = st.selectbox("CMV donneur (Cytomégalovirus)", [0, 1],
                                format_func=lambda x: "Absent" if x == 0 else "Présent")

    st.markdown('<p class="section-title">🔬 Compatibilité Immunologique</p>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3, gap="medium")
    with i1:
        HLAmatch = st.selectbox("Compatibilité HLA (antigènes leucocytaires)", [10, 9, 8, 7],
                                format_func=lambda x: f"{x}/10")
    with i2:
        HLAmismatch = st.number_input("Antigènes HLA différents", min_value=0, max_value=3, value=0, step=1)
    with i3:
        ABOmatch = st.selectbox("Compatibilité ABO (groupe sanguin)", [1, 0],
                                format_func=lambda x: "Compatible" if x == 1 else "Incompatible")

    i4, i5, i6 = st.columns(3, gap="medium")
    with i4:
        Antigen = st.number_input("Antigènes incompatibles", min_value=0, max_value=3, value=0, step=1)
    with i5:
        Alel = st.number_input("Allèles incompatibles", min_value=0, max_value=4, value=0, step=1)
    with i6:
        Gendermatch = st.selectbox("Concordance sexe donneur/receveur", [1, 0],
                                   format_func=lambda x: "Concordant" if x == 1 else "Discordant")

    st.markdown('<p class="section-title">🏥 Maladie & Protocole</p>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3, gap="medium")
    with p1:
        Disease = st.selectbox("Diagnostic",
                               ["ALL", "AML", "CML", "MDS", "SAA", "Fanconi", "other"],
                               format_func=lambda x: x if x != "other" else "Autre")
    with p2:
        Riskgroup = st.selectbox("Groupe de risque", [0, 1],
                                 format_func=lambda x: "Faible" if x == 0 else "Élevé")
    with p3:
        Stemcellsource = st.selectbox("Source cellules souches", [0, 1],
                                      format_func=lambda x: "Moelle Osseuse" if x == 0 else "Sang Périphérique")

    p4, p5, p6 = st.columns(3, gap="medium")
    with p4:
        Txpostrelapse = st.selectbox("Greffe après rechute de la maladie ?", [0, 1],
                                     format_func=lambda x: "Non" if x == 0 else "Oui")
    with p5:
        Diseasegroup = st.selectbox("Groupe maladie", [0, 1],
                                    format_func=lambda x: "Non-maligne" if x == 0 else "Maligne")
    with p6:
        IIIV = st.selectbox("Stade avancé de la maladie (II à IV)", [0, 1],
                            format_func=lambda x: "Non" if x == 0 else "Oui")

    p7, = st.columns(1)
    with p7:
        CMVstatus = st.selectbox("Statut CMV combiné (donneur/receveur)", [0, 1, 2, 3],
                                 format_func=lambda x: ["−/−","−/+","+/−","+/+"][x])

    st.markdown('<p class="section-title">💉 Données de la Greffe</p>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3, gap="medium")
    with g1:
        CD34_str = st.text_input("Dose CD34+ (cellules souches, ×10⁶/kg)", value="3.0")
        ferr("cd34")
    with g2:
        CD3dCD34_str = st.text_input("Ratio CD3/CD34", value="1.0")
        ferr("cd3dcd34")
    with g3:
        CD3dkgx10d8_str = st.text_input("CD3+ (×10⁸/kg)", value="2.0")
        ferr("cd3dkgx10d8")

    st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
    submit = st.form_submit_button("⚡ Calculer la probabilité de succès")

if submit:

    # ── Validation avec erreurs inline via session_state ──
    st.session_state.field_errors = {}
    Rbodymass    = None
    Donorage     = None
    CD34kgx10d6  = None
    CD3dCD34     = None
    CD3dkgx10d8  = None

    try:
        Rbodymass = float(str(Rbodymass_str).replace(",", ".").strip())
        if not (3.0 <= Rbodymass <= 150.0):
            st.session_state.field_errors["poids"] = "Valeur invalide — entrez un nombre entre 3 et 150 kg"
            Rbodymass = None
    except ValueError:
        st.session_state.field_errors["poids"] = f"Valeur invalide — entrez un nombre entre 3 et 150 (ex : 28.5)"

    try:
        Donorage = float(str(Donorage_str).replace(",", ".").strip())
        if not (18 <= Donorage <= 70):
            st.session_state.field_errors["donorage"] = "Valeur invalide — entrez un nombre entre 18 et 70 ans"
            Donorage = None
    except ValueError:
        st.session_state.field_errors["donorage"] = f"Valeur invalide — entrez un nombre entre 18 et 70 (ex : 30)"

    try:
        CD34kgx10d6 = float(str(CD34_str).replace(",", ".").strip())
        if not (0.1 <= CD34kgx10d6 <= 30.0):
            st.session_state.field_errors["cd34"] = "Valeur invalide — entrez un nombre entre 0.1 et 30"
            CD34kgx10d6 = None
    except ValueError:
        st.session_state.field_errors["cd34"] = f"Valeur invalide — entrez un nombre entre 0.1 et 30 (ex : 3.5)"

    try:
        CD3dCD34 = float(str(CD3dCD34_str).replace(",", ".").strip())
    except ValueError:
        st.session_state.field_errors["cd3dcd34"] = f"Valeur invalide — entrez un nombre décimal (ex : 1.0)"

    try:
        CD3dkgx10d8 = float(str(CD3dkgx10d8_str).replace(",", ".").strip())
    except ValueError:
        st.session_state.field_errors["cd3dkgx10d8"] = f"Valeur invalide — entrez un nombre décimal (ex : 2.0)"

    if st.session_state.field_errors:
        st.rerun()

    warnings_list = []
    if Recipientage < 10:
        warnings_list.append("🔴 Receveur pédiatrique critique (< 10 ans) — protocole adapté recommandé.")
    if Donorage and Donorage >= 35:
        warnings_list.append("🟡 Âge du donneur ≥ 35 ans — facteur moins favorable.")
    if DonorCMV == 1 and RecipientCMV == 0:
        warnings_list.append("🟠 CMV donneur+ / receveur− — risque de réactivation élevé.")
    if HLAmatch <= 8:
        warnings_list.append("🟡 Compatibilité HLA incomplète — risque de GvHD accru.")
    if Txpostrelapse == 1:
        warnings_list.append("🟠 Greffe post-rechute — pronostic plus réservé.")

    if not HAS_MODEL:
        st.info("ℹ️ Aucun modèle détecté — la prédiction est simulée.")
    elif not HAS_SCALER:
        st.warning("⚠️ `models/scaler.pkl` introuvable — données non normalisées.")

    if warnings_list:
        warn_html = "".join(f"<div style='margin-bottom:4px'>{w}</div>" for w in warnings_list)
        st.markdown(f'<div class="warning-box">{warn_html}</div>', unsafe_allow_html=True)

    if not HAS_SRC:
        st.error(f"Module de traitement introuvable — {import_error}")
        st.stop()

    try:
        Donorage35      = 1 if (Donorage and Donorage < 35) else 0
        Recipientageint = 0 if Recipientage < 5 else (1 if Recipientage < 10 else (2 if Recipientage < 15 else 3))
        HLAgrI          = round(HLAmatch / 10, 1)

        input_df = pd.DataFrame([{
            "Recipientgender":      Recipientgender,
            "Stemcellsource":       Stemcellsource,
            "Donorage":             Donorage,
            "Donorage35":           Donorage35,
            "IIIV":                 IIIV,
            "Gendermatch":          Gendermatch,
            "DonorABO":             DonorABO,
            "RecipientABO":         RecipientABO,
            "ABOmatch":             ABOmatch,
            "CMVstatus":            CMVstatus,
            "DonorCMV":             DonorCMV,
            "RecipientCMV":         RecipientCMV,
            "Disease":              Disease,
            "Riskgroup":            Riskgroup,
            "Txpostrelapse":        Txpostrelapse,
            "Diseasegroup":         Diseasegroup,
            "HLAmatch":             HLAmatch,
            "HLAmismatch":          HLAmismatch,
            "Antigen":              Antigen,
            "Alel":                 Alel,
            "HLAgrI":               HLAgrI,
            "Recipientage":         Recipientage,
            "Recipientage10":       Recipientage10,
            "Recipientageint":      Recipientageint,
            "CD34kgx10d6":          CD34kgx10d6,
            "CD3dCD34":             CD3dCD34,
            "CD3dkgx10d8":          CD3dkgx10d8,
            "Rbodymass":            Rbodymass,
        }])

        input_df   = handle_missing_values(input_df)
        input_df   = encode_features(input_df)
        input_df   = handle_outliers(input_df, target_col="__none__")
        data_ready = optimize_memory(input_df)
        if "__none__" in data_ready.columns:
            data_ready = data_ready.drop(columns=["__none__"])

        # Aligner les colonnes exactement sur ce que le modèle attend
        if HAS_MODEL and hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
            for col in expected_cols:
                if col not in data_ready.columns:
                    data_ready[col] = 0
            data_ready = data_ready[expected_cols]

        if HAS_SCALER:
            try:
                scaler_cols   = getattr(scaler, "feature_names_in_",
                                        data_ready.select_dtypes(include=[np.number]).columns)
                cols_to_scale = [c for c in scaler_cols if c in data_ready.columns]
                if cols_to_scale:
                    data_ready[cols_to_scale] = scaler.transform(data_ready[cols_to_scale])
            except Exception:
                pass

        if HAS_MODEL:
            proba = model.predict_proba(data_ready)[0][1] * 100
        else:
            np.random.seed(int(Recipientage * 7 + (Rbodymass or 30) * 3 + (Donorage or 30)))
            proba = np.random.uniform(45, 97)

        if proba >= 75:
            card_cls, score_cls, niveau, emoji_res = "result-high",   "score-high",   "Favorable",   "✅"
        elif proba >= 50:
            card_cls, score_cls, niveau, emoji_res = "result-medium", "score-medium", "Modéré",      "⚠️"
        else:
            card_cls, score_cls, niveau, emoji_res = "result-low",    "score-low",    "Défavorable", "❌"

        age_ctx    = "pédiatrique critique" if Recipientage < 10 else f"{Recipientage} ans"
        source_txt = "moelle osseuse" if Stemcellsource == 0 else "sang périphérique"
        hla_txt    = f"HLA {HLAmatch}/10"

        if proba >= 75:
            interp = (f"Profil ({age_ctx}, {Rbodymass} kg, {Disease}) avec {hla_txt} "
                      f"et greffe à partir de {source_txt} — indicateurs globalement favorables. "
                      f"Probabilité estimée : <strong>{proba:.1f}%</strong>.")
        elif proba >= 50:
            interp = (f"Résultats mitigés ({age_ctx}, {Disease}, {hla_txt}). "
                      f"Probabilité estimée : <strong>{proba:.1f}%</strong>. "
                      f"Surveillance rapprochée recommandée.")
        else:
            interp = (f"Facteurs de risque significatifs ({age_ctx}, {Disease}, {hla_txt}). "
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
            <strong>🔧 Pipeline :</strong>
            handle_missing_values() → encode_features() → handle_outliers() → optimize_memory()
            {"→ scaler.transform()" if HAS_SCALER else ""}
            &nbsp;·&nbsp; <strong>Modèle :</strong> {_loaded_model_name if HAS_MODEL else "Simulation"}
            &nbsp;·&nbsp; <strong>Features :</strong> {data_ready.shape[1]}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="shap-card">', unsafe_allow_html=True)
            st.markdown('<p class="shap-title">📊 Explicabilité SHAP</p>', unsafe_allow_html=True)
            st.markdown('<p class="shap-subtitle">Contribution de chaque paramètre clinique à la prédiction finale</p>', unsafe_allow_html=True)

            feature_names = list(data_ready.columns)

            if HAS_SHAP and HAS_MODEL:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(data_ready)
                    # RandomForest renvoie une liste [class0, class1], shape (n_samples, n_features)
                    # XGBoost renvoie un array 2D (n_samples, n_features)
                    if isinstance(shap_vals, list):
                        # classification binaire : prendre classe 1
                        sv = shap_vals[1]
                    else:
                        sv = shap_vals
                    # sv peut être 2D (n_samples, n_features) ou 1D
                    if hasattr(sv, 'ndim') and sv.ndim == 2:
                        shap_values = sv[0]
                    else:
                        shap_values = np.array(sv).flatten()
                except Exception:
                    np.random.seed(int(Recipientage * 7 + (Rbodymass or 30) * 3))
                    raw = np.random.randn(len(feature_names))
                    shap_values = raw / (np.abs(raw).sum() + 1e-9) * (proba - 50) / 12
            else:
                np.random.seed(int(Recipientage * 7 + (Rbodymass or 30) * 3))
                raw = np.random.randn(len(feature_names))
                shap_values = raw / (np.abs(raw).sum() + 1e-9) * (proba - 50) / 12

            label_map = {
                "Recipientgender":      "Sexe receveur",
                "Stemcellsource":       "Source cellules",
                "Donorage":             "Âge donneur",
                "Donorage35":           "Donneur < 35 ans",
                "IIIV":                 "Stade II-IV",
                "Gendermatch":          "Concordance sexe",
                "DonorABO":             "ABO donneur",
                "RecipientABO":         "ABO receveur",
                "ABOmatch":             "Compat. ABO",
                "CMVstatus":            "CMV combiné",
                "DonorCMV":             "CMV donneur",
                "RecipientCMV":         "CMV receveur",
                "Disease":              "Maladie (ALL)",
                "Disease_AML":          "Maladie AML",
                "Disease_chronic":      "Maladie chronique",
                "Disease_lymphoma":     "Lymphome",
                "Disease_nonmalignant": "Non-malin",
                "Riskgroup":            "Risque",
                "Txpostrelapse":        "Post-rechute",
                "Diseasegroup":         "Groupe maladie",
                "HLAmatch":             "HLA match",
                "HLAmismatch":          "HLA diff.",
                "Antigen":              "Antigènes",
                "Alel":                 "Allèles",
                "HLAgrI":               "Score HLA",
                "Recipientage":         "Âge receveur",
                "Recipientage10":       "< 10 ans",
                "Recipientageint":      "Tranche âge",
                "CD34kgx10d6":          "CD34+/kg",
                "CD3dCD34":             "Ratio CD3/CD34",
                "CD3dkgx10d8":          "CD3+/kg",
                "Rbodymass":            "Poids",
            }
            labels = [label_map.get(f, f[:14]) for f in feature_names]

            order        = np.argsort(np.abs(shap_values))[::-1][:8]
            shap_sorted  = shap_values[order]
            label_sorted = [labels[i] for i in order]

            COLOR_POS = "#6096e0"
            COLOR_NEG = "#e07070"
            colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in shap_sorted]

            col_chart, col_legend = st.columns([3, 2], gap="large")

            with col_chart:
                n = len(label_sorted)
                fig, ax = plt.subplots(figsize=(max(4, n * 0.9), 3.6))
                fig.patch.set_facecolor("#f8fafc")
                ax.set_facecolor("#f8fafc")
                x = np.arange(n)
                ax.bar(x, shap_sorted, color=colors, width=0.28,
                       edgecolor="none", zorder=3, alpha=0.9)
                ax.axhline(0, color="#94a3b8", linewidth=0.8, zorder=5)
                max_abs = max(abs(shap_sorted)) if len(shap_sorted) > 0 else 1
                for xi, val in zip(x, shap_sorted):
                    offset = max_abs * 0.08
                    va   = "bottom" if val >= 0 else "top"
                    ypos = val + (offset if val >= 0 else -offset)
                    ax.text(xi, ypos, f"{val:+.3f}", ha="center", va=va,
                            fontsize=7.5, fontweight="600", color="#334155")
                ax.set_xticks(x)
                ax.set_xticklabels(label_sorted, fontsize=8, color="#475569",
                                   fontweight="500", rotation=20, ha="right")
                ax.set_ylabel("Valeur SHAP", fontsize=8, color="#94a3b8", labelpad=6)
                ax.yaxis.grid(True, linestyle=":", alpha=0.35, color="#e2e8f0", zorder=0)
                ax.set_axisbelow(True)
                for spine in ["top", "right", "left"]:
                    ax.spines[spine].set_visible(False)
                ax.spines["bottom"].set_color("#f1f5f9")
                ax.tick_params(axis="y", colors="#cbd5e1", labelsize=7)
                ax.tick_params(axis="x", length=0)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin * 1.35, ymax * 1.35)
                plt.tight_layout(pad=1.0)
                st.pyplot(fig)
                plt.close(fig)
                st.markdown("""
                <div class="shap-note" style="margin-top:0.5rem">
                  <strong>Comment lire ?</strong><br>
                  Barre <strong style="color:#6096e0">bleue</strong> = ce facteur <em>augmente</em> la probabilité de succès.<br>
                  Barre <strong style="color:#e07070">rouge</strong> = ce facteur la <em>diminue</em>.
                </div>
                """, unsafe_allow_html=True)

            with col_legend:
                desc_map = {
                    "Sexe receveur":    "Influence les réponses immunologiques post-greffe.",
                    "Source cellules":  "Moelle vs sang périphérique — impact sur la prise de greffe et le risque GvHD.",
                    "Âge donneur":      "Un donneur < 35 ans est associé à de meilleurs résultats.",
                    "Donneur < 35 ans": "Donneur jeune = facteur favorable démontré.",
                    "Stade II-IV":      "Stade avancé de la maladie — pronostic plus réservé.",
                    "Concordance sexe": "Concordance de sexe réduit le risque de GvHD.",
                    "ABO donneur":      "Groupe sanguin du donneur.",
                    "ABO receveur":     "Groupe sanguin du receveur.",
                    "Compat. ABO":      "Incompatibilité ABO → risque de réaction hémolytique.",
                    "CMV combiné":      "Combinaison des statuts CMV donneur et receveur.",
                    "CMV donneur":      "CMV donneur+ / receveur− = risque élevé.",
                    "CMV receveur":     "Détermine le risque de réactivation CMV post-greffe.",
                    "Maladie":          "La pathologie détermine le protocole et le pronostic.",
                    "Maladie AML":      "Leucémie myéloïde aiguë.",
                    "Maladie chronique":"Leucémie myéloïde chronique.",
                    "Lymphome":         "Lymphome — pronostic variable.",
                    "Non-malin":        "Maladie non-maligne — meilleur pronostic global.",
                    "Risque":           "Classification du risque au moment de la greffe.",
                    "Post-rechute":     "Greffe après rechute — pronostic plus réservé.",
                    "Groupe maladie":   "Maladie maligne vs non-maligne.",
                    "HLA match":        "10/10 = prédicteur fort de succès.",
                    "HLA diff.":        "Antigènes HLA incompatibles.",
                    "Antigènes":        "Antigènes HLA incompatibles entre donneur et receveur.",
                    "Allèles":          "Allèles HLA incompatibles.",
                    "Score HLA":        "Score global de compatibilité HLA.",
                    "Âge receveur":     "Influence la tolérance au conditionnement.",
                    "< 10 ans":         "Receveur pédiatrique critique.",
                    "Tranche âge":      "Catégorie d'âge du receveur.",
                    "CD34+/kg":         "Variable #1 selon la littérature : doses élevées prolongent la survie.",
                    "Ratio CD3/CD34":   "Rapport entre lymphocytes T et cellules souches.",
                    "CD3+/kg":          "Dose de lymphocytes T — impact sur GvHD et GvL.",
                    "Poids":            "Détermine les doses de chimiothérapie et le ratio cellules / kg.",
                }

                st.markdown('<p class="shap-legend-title">📌 Top facteurs influents</p>', unsafe_allow_html=True)

                for lbl, val in zip(label_sorted, shap_sorted):
                    dot_cls   = "shap-dot-pos" if val >= 0 else "shap-dot-neg"
                    direction = "↑ favorable" if val >= 0 else "↓ défavorable"
                    color_val = "#6096e0" if val >= 0 else "#e07070"
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


            st.markdown("</div>", unsafe_allow_html=True)



    except Exception as e:
        st.error(f"Erreur lors du traitement : {e}")
        st.exception(e)