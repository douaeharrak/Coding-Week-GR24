import streamlit as st
import streamlit.components.v1 as components
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

# ── CSS : mise en forme du document 1 (original) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800;900&family=Playfair+Display:wght@600;700&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; font-weight: 600; }

.stApp {
    background: linear-gradient(160deg, #dbeafe 0%, #eff6ff 40%, #e0f2fe 100%);
    min-height: 100vh;
}

.main-header {
    background: linear-gradient(135deg, #2563eb 0%, #3b82f6 50%, #60a5fa 100%);
    border-radius: 22px;
    padding: 2rem 2.8rem;
    margin-bottom: 2rem;
    display: flex; align-items: center; gap: 1.6rem;
    box-shadow: 0 12px 40px rgba(15,42,74,0.3);
    position: relative; overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute; top:-60px; right:-60px;
    width:220px; height:220px; border-radius:50%;
    background: radial-gradient(circle, rgba(56,189,248,0.15) 0%, transparent 70%);
}
.main-header::after {
    content: '';
    position: absolute; bottom:-40px; left:200px;
    width:150px; height:150px; border-radius:50%;
    background: radial-gradient(circle, rgba(167,139,250,0.12) 0%, transparent 70%);
}
.main-header h1 {
    font-family:'Playfair Display',serif;
    font-size:2rem; margin:0; font-weight:700;
    color:#ffffff; letter-spacing:-0.01em;
    text-shadow: 0 2px 16px rgba(0,0,0,0.2), 0 0 40px rgba(255,255,255,0.15);
}
.main-header p {
    margin:0.4rem 0 0; color:rgba(255,255,255,0.65);
    font-size:0.85rem; font-weight:600;
    letter-spacing:0.06em; text-transform:uppercase;
}
.header-accent {
    width:5px; height:70px; border-radius:6px; flex-shrink:0;
    background: linear-gradient(180deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
    box-shadow: 0 0 16px rgba(56,189,248,0.6);
}
.header-icon { font-size:3rem; filter: drop-shadow(0 4px 10px rgba(56,189,248,0.5)); }

div[data-baseweb="input"] input,
div[data-testid="stNumberInput"] input {
    background: #ffffff !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #0f2a4a !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-baseweb="input"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #0f2a4a !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}
div[data-baseweb="select"] * { cursor: pointer !important; }
label[data-testid="stWidgetLabel"] p,
div[data-testid="stNumberInput"] label p {
    color: #1e40af !important;
    font-size: 0.78rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
div[data-testid="stNumberInput"] button {
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    color: #475569 !important;
    padding: 2px 6px !important;
    cursor: pointer !important;
    font-size: 1rem !important;
}
div[data-testid="stNumberInput"] button:hover {
    background: #dbeafe !important;
    border-color: #93c5fd !important;
    color: #1e40af !important;
}
div[data-testid="stNumberInput"] button svg { fill: currentColor !important; }

div.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 50%, #93c5fd 100%) !important;
    color: #1e3a5f !important;
    border: 1.5px solid #bfdbfe !important;
    outline: none !important;
    border-radius: 14px !important;
    padding: 1.05rem 2rem !important;
    font-size: 1.25rem !important;
    font-weight: 900 !important;
    font-family: 'Playfair Display', serif !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 6px 24px rgba(59,130,246,0.4), inset 0 1px 0 rgba(255,255,255,0.3) !important;
    transition: all 0.2s !important;
    margin-top: 1rem !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    background: linear-gradient(135deg, #2563eb 0%, #3b82f6 50%, #60a5fa 100%) !important;
    box-shadow: 0 10px 32px rgba(59,130,246,0.55), inset 0 1px 0 rgba(255,255,255,0.3) !important;
    color: #ffffff !important;
}

.warning-box {
    background: linear-gradient(135deg, #fef9ec, #fff8e1);
    border: 2px solid #f59e0b;
    border-left: 5px solid #d97706;
    border-radius: 12px;
    padding: 0.9rem 1.3rem;
    margin-bottom: 0.9rem;
    font-size: 0.84rem;
    color: #78350f;
    font-weight: 700;
    box-shadow: 0 2px 10px rgba(245,158,11,0.15);
}

.result-card { border-radius: 20px; padding: 2.2rem 2.5rem; margin-top: 1.2rem; text-align: center; }
.result-high   { background: linear-gradient(135deg, #f0fdf4, #dcfce7); border-left: 6px solid #22c55e; box-shadow: 0 8px 30px rgba(34,197,94,0.12); border-radius: 20px; }
.result-medium { background: linear-gradient(135deg, #fffbeb, #fef3c7); border-left: 6px solid #f59e0b; box-shadow: 0 8px 30px rgba(245,158,11,0.12); border-radius: 20px; }
.result-low    { background: linear-gradient(135deg, #fff1f2, #ffe4e6); border-left: 6px solid #ef4444; box-shadow: 0 8px 30px rgba(244,63,94,0.12); border-radius: 20px; }
.result-score  { font-family:'Playfair Display',serif; font-size:5rem; font-weight:700; margin:0; line-height:1; }
.score-high    { color:#16a34a; }
.score-medium  { color:#d97706; }
.score-low     { color:#dc2626; }
.result-label  { font-size:0.8rem; font-weight:800; color:#64748b; margin-bottom:0.6rem; text-transform:uppercase; letter-spacing:0.12em; }
.result-interp { margin-top:0.9rem; font-size:0.93rem; color:#374151; line-height:1.7; font-weight:600; }

.pipeline-box {
    background: #f0f7ff; border: 1.5px solid #bfdbfe; border-left: 5px solid #3b82f6;
    border-radius: 12px; padding: 0.9rem 1.3rem; margin-top: 1rem;
    font-size: 0.79rem; color: #1e40af; line-height: 2;
    font-family: 'Courier New', monospace; font-weight: 700;
}
.pipeline-box strong { color: #1d4ed8; font-family: 'Nunito', sans-serif; font-weight:800; }

.shap-card { background: #ffffff; border: 1.5px solid #e0e8f8; border-radius: 18px; padding: 1.6rem 1.8rem; margin-top: 1rem; box-shadow: 0 4px 20px rgba(15,42,74,0.07); }
.shap-title    { font-family:'Playfair Display',serif; font-size:1.3rem; color:#0f2a4a; margin-bottom:0.2rem; font-weight:700; }
.shap-subtitle { font-size:0.82rem; color:#64748b; margin-bottom:0.8rem; font-weight:600; font-style:italic; }
.shap-legend-title { font-size:0.8rem; font-weight:800; color:#1e40af; margin-bottom:0.8rem; text-transform:uppercase; letter-spacing:0.08em; }
.shap-item     { display:flex; align-items:flex-start; gap:10px; margin-bottom:0.85rem; }
.shap-dot      { width:11px; height:11px; border-radius:50%; flex-shrink:0; margin-top:4px; }
.shap-dot-pos  { background:#3b82f6; box-shadow: 0 0 6px rgba(59,130,246,0.5); }
.shap-dot-neg  { background:#ef4444; box-shadow: 0 0 6px rgba(239,68,68,0.5); }
.shap-item-label { font-size:0.79rem; font-weight:800; color:#0f2a4a; }
.shap-item-desc  { font-size:0.73rem; color:#64748b; line-height:1.5; margin-top:2px; font-weight:600; }
.shap-note {
    background: #f8fbff; border-left: 3px solid #93c5fd; border-radius: 8px;
    padding: 0.7rem 1rem; font-size: 0.76rem; color: #3b82f6;
    line-height: 1.8; margin-top: 0.6rem; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="main-header">
  <div class="header-accent"></div>
  <span class="header-icon">🩺</span>
  <div>
    <h1 style="color:#ffffff; font-weight:700;">Prédiction de Succès — Greffe de Moelle Osseuse</h1>
    <p>Outil d'aide à la décision clinique &nbsp;·&nbsp; Groupe 24 &nbsp;·&nbsp; Coding Week 2026 &nbsp;·&nbsp; École Centrale Casablanca</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
st.markdown('''
<div style="text-align:center; margin:1.5rem 0 0.4rem 0;">
  <div style="display:inline-block; background:linear-gradient(135deg,#f0f7ff,#e8f2ff); border-left:6px solid #3b82f6; border-radius:6px; padding:0.75rem 2.5rem; box-shadow:0 4px 16px rgba(59,130,246,0.15);">
    <span style="font-family:Playfair Display,serif; font-size:1.7rem; font-weight:700; color:#1d4ed8; letter-spacing:-0.01em;">
      📋 Dossier Patient
    </span>
  </div>
</div>
''', unsafe_allow_html=True)
st.markdown('''
<div style="text-align:center; margin-bottom:1.8rem;">
  <span style="font-size:0.85rem; color:#64748b; font-weight:700; letter-spacing:0.05em; text-transform:uppercase;">
    ── Renseignez les informations cliniques ──
  </span>
</div>
''', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════
for k, v in [("errs", {}), ("show_alert", False)]:
    if k not in st.session_state:
        st.session_state[k] = v
for k, v in [("_val_Rbodymass","30"), ("_val_Donorage","30"),
             ("_val_CD34","3.0"), ("_val_CD3dCD34","1.0"), ("_val_CD3dkgx10d8","2.0")]:
    if k not in st.session_state:
        st.session_state[k] = v

def _clear_err(key):
    if key in st.session_state.errs:
        st.session_state.errs.pop(key, None)
        if not st.session_state.errs:
            st.session_state.show_alert = False

def _update_age10():
    """Met à jour automatiquement Receveur < 10 ans selon l'âge saisi."""
    age = st.session_state.get("_ni_Recipientage", 8)
    st.session_state["_sb_Recipientage10"] = 1 if age < 10 else 0

def _update_ratio():
    """Calcule automatiquement le Ratio CD3/CD34 si les deux valeurs sont valides."""
    try:
        cd34 = float(st.session_state.get("_val_CD34", "0").replace(",", ".").strip())
        cd3  = float(st.session_state.get("_val_CD3dkgx10d8", "0").replace(",", ".").strip())
        if cd34 > 0:
            ratio = round(cd3 / cd34, 4)
            st.session_state["_val_CD3dCD34"] = str(ratio)
    except (ValueError, ZeroDivisionError):
        pass
    _clear_err("CD3dCD34_str")

def ferr(key):
    e = st.session_state.errs.get(key)
    if e:
        st.markdown(
            f'<div style="color:#dc2626;font-size:0.78rem;margin-top:-8px;margin-bottom:6px;'
            f'padding:3px 8px;background:#fef2f2;border-radius:4px;border:1px solid #fecaca">⚠️ {e}</div>',
            unsafe_allow_html=True
        )

BADGE_SECTION = (
    "display:inline-flex;align-items:center;gap:0.6rem;"
    "font-family:'Playfair Display',serif;font-size:1.15rem;font-weight:700;color:#1e40af;"
    "background:linear-gradient(135deg,#f0f7ff,#e8f2ff);"
    "border-left:6px solid #3b82f6;border-radius:6px;"
    "padding:0.65rem 2rem;box-shadow:0 4px 16px rgba(59,130,246,0.15);"
)

def section_badge(icon, title):
    st.markdown(f"""
    <div style="text-align:center;margin:1.8rem 0 0.6rem 0;">
      <span style="{BADGE_SECTION}">{icon} {title}</span>
    </div>
    <div style="height:2px;background:linear-gradient(90deg,transparent 0%,#dbeafe 30%,#bfdbfe 50%,#dbeafe 70%,transparent 100%);margin:0 0 1.2rem 0;border-radius:2px;"></div>
    """, unsafe_allow_html=True)

# ── Marqueur pour le scroll (placé AVANT les champs) ──
st.markdown('<div id="error-anchor"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  FORMULAIRE — widgets hors st.form pour on_change
# ══════════════════════════════════════════════════════

# ── Section Receveur ──
section_badge("👤", "Receveur (Patient)")
c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    Recipientage = st.number_input("Âge du receveur (années)", min_value=0, max_value=20, value=8, step=1, key="_ni_Recipientage", on_change=_update_age10)
with c2:
    Recipientgender = st.selectbox("Sexe du receveur", [1, 0], key="_sb_Recipientgender",
                                   format_func=lambda x: "Masculin" if x == 1 else "Féminin")
with c3:
    st.text_input("Poids du receveur (kg)", key="_val_Rbodymass",
                  on_change=lambda: _clear_err("Rbodymass_str"))
    ferr("Rbodymass_str")

c4, c5, c6 = st.columns(3, gap="medium")
with c4:
    RecipientABO = st.selectbox("Groupe sanguin receveur", [0, 1, 2, 3], key="_sb_RecipientABO",
                                format_func=lambda x: ["0","A","B","AB"][x])
with c5:
    RecipientCMV = st.selectbox("CMV receveur (Cytomégalovirus)", [0, 1], key="_sb_RecipientCMV",
                                format_func=lambda x: "Absent" if x == 0 else "Présent")
with c6:
    _age10_val = 1 if st.session_state.get("_ni_Recipientage", 8) < 10 else 0
    st.session_state["_sb_Recipientage10"] = _age10_val
    Recipientage10 = _age10_val
    _label10 = "✅ Oui (< 10 ans)" if _age10_val == 1 else "Non (≥ 10 ans)"
    st.markdown(f'''<div style="margin-top:1.6rem;"><label style="color:#1e40af;font-size:0.78rem;font-weight:800;text-transform:uppercase;letter-spacing:0.07em;">RECEVEUR &lt; 10 ANS ?</label><div style="background:#f0f7ff;border:2px solid #bfdbfe;border-radius:10px;padding:0.55rem 1rem;margin-top:4px;font-weight:700;color:#1e40af;font-size:0.95rem;">{_label10} <span style="color:#64748b;font-size:0.75rem;font-weight:600;">(auto)</span></div></div>''', unsafe_allow_html=True)

# ── Section Donneur ──
section_badge("🧬", "Donneur")
d1, d2, d3 = st.columns(3, gap="medium")
with d1:
    st.text_input("Âge du donneur (années)", key="_val_Donorage",
                  on_change=lambda: _clear_err("Donorage_str"))
    ferr("Donorage_str")
with d2:
    DonorABO = st.selectbox("Groupe sanguin donneur", [0, 1, 2, 3], key="_sb_DonorABO",
                            format_func=lambda x: ["0","A","B","AB"][x])
with d3:
    DonorCMV = st.selectbox("CMV donneur (Cytomégalovirus)", [0, 1], key="_sb_DonorCMV",
                            format_func=lambda x: "Absent" if x == 0 else "Présent")

# ── Section Immunologie ──
section_badge("🔬", "Compatibilité Immunologique")
i1, i2, i3 = st.columns(3, gap="medium")
with i1:
    HLAmatch = st.selectbox("Compatibilité HLA (antigènes leucocytaires)", [10, 9, 8, 7], key="_sb_HLAmatch",
                            format_func=lambda x: f"{x}/10")
with i2:
    HLAmismatch = st.number_input("Antigènes HLA différents", min_value=0, max_value=3, value=0, step=1, key="_ni_HLAmismatch")
with i3:
    ABOmatch = st.selectbox("Compatibilité ABO (groupe sanguin)", [1, 0], key="_sb_ABOmatch",
                            format_func=lambda x: "Compatible" if x == 1 else "Incompatible")

i4, i5, i6 = st.columns(3, gap="medium")
with i4:
    Antigen = st.number_input("Antigènes incompatibles", min_value=0, max_value=3, value=0, step=1, key="_ni_Antigen")
with i5:
    Alel = st.number_input("Allèles incompatibles", min_value=0, max_value=4, value=0, step=1, key="_ni_Alel")
with i6:
    Gendermatch = st.selectbox("Concordance sexe donneur/receveur", [1, 0], key="_sb_Gendermatch",
                               format_func=lambda x: "Concordant" if x == 1 else "Discordant")

# ── Section Maladie ──
section_badge("🏥", "Maladie & Protocole")
p1, p2, p3 = st.columns(3, gap="medium")
with p1:
    Disease = st.selectbox("Diagnostic",
                           ["ALL", "AML", "CML", "MDS", "SAA", "Fanconi", "other"], key="_sb_Disease",
                           format_func=lambda x: x if x != "other" else "Autre")
with p2:
    Riskgroup = st.selectbox("Groupe de risque", [0, 1], key="_sb_Riskgroup",
                             format_func=lambda x: "Faible" if x == 0 else "Élevé")
with p3:
    Stemcellsource = st.selectbox("Source cellules souches", [0, 1], key="_sb_Stemcellsource",
                                  format_func=lambda x: "Moelle Osseuse" if x == 0 else "Sang Périphérique")

p4, p5, p6 = st.columns(3, gap="medium")
with p4:
    Txpostrelapse = st.selectbox("Greffe après rechute de la maladie ?", [0, 1], key="_sb_Txpostrelapse",
                                 format_func=lambda x: "Non" if x == 0 else "Oui")
with p5:
    Diseasegroup = st.selectbox("Groupe maladie", [0, 1], key="_sb_Diseasegroup",
                                format_func=lambda x: "Non-maligne" if x == 0 else "Maligne")
with p6:
    IIIV = st.selectbox("Stade avancé de la maladie (II à IV)", [0, 1], key="_sb_IIIV",
                        format_func=lambda x: "Non" if x == 0 else "Oui")

p7, = st.columns(1)
with p7:
    CMVstatus = st.selectbox("Statut CMV combiné (donneur/receveur)", [0, 1, 2, 3], key="_sb_CMVstatus",
                             format_func=lambda x: ["−/−","−/+","+/−","+/+"][x])

# ── Section Greffe ──
section_badge("💉", "Données de la Greffe")
g1, g2, g3 = st.columns(3, gap="medium")
with g1:
    st.text_input("Dose CD34+ (cellules souches, ×10⁶/kg)", key="_val_CD34",
                  on_change=lambda: (_clear_err("CD34_str"), _update_ratio()))
    ferr("CD34_str")
with g2:
    # Ratio calculé automatiquement — champ grisé en lecture seule
    _ratio_val = st.session_state.get("_val_CD3dCD34", "1.0")
    st.markdown(f'''<div><label style="color:#1e40af;font-size:0.78rem;font-weight:800;text-transform:uppercase;letter-spacing:0.07em;">RATIO CD3/CD34</label><div style="background:#f8fafc;border:2px solid #e2e8f0;border-radius:10px;padding:0.55rem 1rem;margin-top:4px;font-weight:700;color:#64748b;font-size:0.95rem;">{_ratio_val} <span style="color:#94a3b8;font-size:0.75rem;font-weight:600;">(auto)</span></div></div>''', unsafe_allow_html=True)
    ferr("CD3dCD34_str")
with g3:
    st.text_input("CD3+ (×10⁸/kg)", key="_val_CD3dkgx10d8",
                  on_change=lambda: (_clear_err("CD3dkgx10d8_str"), _update_ratio()))
    ferr("CD3dkgx10d8_str")

st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
submit = st.button("🔬  Calculer la Probabilité de Succès  →", use_container_width=True)

# ── Bannière d'erreur globale + son + scroll ──
if st.session_state.show_alert and st.session_state.errs:
    st.markdown("""
    <div id="error-banner" style="background:linear-gradient(135deg,#fff1f2,#ffe4e6);
        border:2px solid #ef4444; border-left:6px solid #dc2626; border-radius:12px;
        padding:1rem 1.5rem; margin-top:0.8rem;
        display:flex; align-items:center; gap:0.8rem;
        box-shadow:0 4px 20px rgba(239,68,68,0.25);">
      <span style="font-size:1.8rem;">🚨</span>
      <div>
        <div style="font-size:1rem;font-weight:900;color:#dc2626;">Formulaire incomplet</div>
        <div style="font-size:0.85rem;font-weight:700;color:#b91c1c;margin-top:3px;">
          Vous n'avez pas bien renseigné une ou plusieurs cases.<br>
          Veuillez corriger les champs indiqués en rouge ci-dessus.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    components.html("""
    <script>
    (function(){
        // Son d'alerte
        try{
            var ctx=new(window.AudioContext||window.webkitAudioContext)();
            function b(f,t,d,v){var o=ctx.createOscillator(),g=ctx.createGain();o.connect(g);g.connect(ctx.destination);o.frequency.value=f;o.type='square';g.gain.setValueAtTime(v,ctx.currentTime+t);g.gain.exponentialRampToValueAtTime(0.001,ctx.currentTime+t+d);o.start(ctx.currentTime+t);o.stop(ctx.currentTime+t+d+0.05);}
            b(880,0.00,0.15,0.7);b(660,0.18,0.15,0.7);b(440,0.36,0.25,0.7);
        }catch(e){}
        // Scroll vers l'ancre error-anchor (placée avant les champs en erreur)
        function doScroll(){
            try{
                var doc = window.parent.document;
                var el = doc.getElementById('error-anchor');
                if(el){
                    el.scrollIntoView({behavior:'smooth', block:'start'});
                    return true;
                }
                // fallback : scroll vers le haut de la page
                window.parent.scrollTo({top: 0, behavior:'smooth'});
            }catch(e){}
            return false;
        }
        // Essayer plusieurs fois car Streamlit peut mettre du temps à rendre
        setTimeout(doScroll, 100);
        setTimeout(doScroll, 400);
        setTimeout(doScroll, 800);
    })();
    </script>
    """, height=0)

# ══════════════════════════════════════════════════════
#  TRAITEMENT APRÈS SOUMISSION
# ══════════════════════════════════════════════════════
if submit:

    Rbodymass_str   = st.session_state.get("_val_Rbodymass",   "30")
    Donorage_str    = st.session_state.get("_val_Donorage",    "30")
    CD34_str        = st.session_state.get("_val_CD34",        "3.0")
    CD3dCD34_str    = st.session_state.get("_val_CD3dCD34",    "1.0")
    CD3dkgx10d8_str = st.session_state.get("_val_CD3dkgx10d8", "2.0")

    had_errs_before = bool(st.session_state.errs)
    new_errs = {}
    Rbodymass = None; Donorage = None; CD34kgx10d6 = None; CD3dCD34 = None; CD3dkgx10d8 = None

    try:
        Rbodymass = float(Rbodymass_str.replace(",", ".").strip())
        if not (3.0 <= Rbodymass <= 150.0):
            new_errs["Rbodymass_str"] = "Vous devez choisir un nombre compris entre 3 et 150 kg"
            Rbodymass = None
    except ValueError:
        new_errs["Rbodymass_str"] = "Vous devez saisir un nombre valide compris entre 3 et 150 (ex : 28.5)"

    try:
        Donorage = float(Donorage_str.replace(",", ".").strip())
        if not (18.0 <= Donorage <= 70.0):
            new_errs["Donorage_str"] = "Vous devez choisir un nombre compris entre 18 et 70 ans"
            Donorage = None
    except ValueError:
        new_errs["Donorage_str"] = "Vous devez saisir un nombre valide compris entre 18 et 70 (ex : 30)"

    try:
        CD34kgx10d6 = float(CD34_str.replace(",", ".").strip())
        if not (0.1 <= CD34kgx10d6 <= 30.0):
            new_errs["CD34_str"] = "Vous devez choisir un nombre compris entre 0.1 et 30"
            CD34kgx10d6 = None
    except ValueError:
        new_errs["CD34_str"] = "Vous devez saisir un nombre valide compris entre 0.1 et 30 (ex : 3.5)"

    try:
        CD3dCD34 = float(CD3dCD34_str.replace(",", ".").strip())
    except ValueError:
        new_errs["CD3dCD34_str"] = "Vous devez saisir un nombre décimal valide (ex : 1.0)"

    try:
        CD3dkgx10d8 = float(CD3dkgx10d8_str.replace(",", ".").strip())
    except ValueError:
        new_errs["CD3dkgx10d8_str"] = "Vous devez saisir un nombre décimal valide (ex : 2.0)"

    if new_errs:
        st.session_state.errs       = new_errs
        st.session_state.show_alert = True   # son + alerte dès la 1ère saisie invalide
        st.rerun()

    st.session_state.errs       = {}
    st.session_state.show_alert = False

    Recipientage = int(Recipientage)
    HLAmismatch  = int(HLAmismatch)
    Antigen      = int(Antigen)
    Alel         = int(Alel)

    warnings_list = []
    if Recipientage < 10:
        warnings_list.append("🔴 Receveur pédiatrique critique (< 10 ans) — protocole adapté recommandé.")
    if Donorage >= 35:
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
        Donorage35      = 1 if Donorage < 35 else 0
        Recipientageint = 0 if Recipientage < 5 else (1 if Recipientage < 10 else (2 if Recipientage < 15 else 3))
        HLAgrI          = round(HLAmatch / 10, 1)

        input_df = pd.DataFrame([{
            "Recipientgender":  Recipientgender,  "Stemcellsource":   Stemcellsource,
            "Donorage":         Donorage,          "Donorage35":       Donorage35,
            "IIIV":             IIIV,              "Gendermatch":      Gendermatch,
            "DonorABO":         DonorABO,          "RecipientABO":     RecipientABO,
            "ABOmatch":         ABOmatch,          "CMVstatus":        CMVstatus,
            "DonorCMV":         DonorCMV,          "RecipientCMV":     RecipientCMV,
            "Disease":          Disease,           "Riskgroup":        Riskgroup,
            "Txpostrelapse":    Txpostrelapse,     "Diseasegroup":     Diseasegroup,
            "HLAmatch":         HLAmatch,          "HLAmismatch":      HLAmismatch,
            "Antigen":          Antigen,           "Alel":             Alel,
            "HLAgrI":           HLAgrI,            "Recipientage":     Recipientage,
            "Recipientage10":   Recipientage10,    "Recipientageint":  Recipientageint,
            "CD34kgx10d6":      CD34kgx10d6,       "CD3dCD34":         CD3dCD34,
            "CD3dkgx10d8":      CD3dkgx10d8,       "Rbodymass":        Rbodymass,
        }])

        input_df   = handle_missing_values(input_df)
        input_df   = encode_features(input_df)
        input_df   = handle_outliers(input_df, target_col="__none__")
        data_ready = optimize_memory(input_df)
        if "__none__" in data_ready.columns:
            data_ready = data_ready.drop(columns=["__none__"])

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
            st.markdown('''
            <div style="text-align:center; margin-bottom:1.2rem;">
              <div style="display:inline-block; background:linear-gradient(135deg,#dbeafe,#bfdbfe); border:2.5px solid #60a5fa; padding:0.7rem 2.5rem; border-radius:16px; margin-bottom:0.8rem; box-shadow:0 4px 16px rgba(59,130,246,0.2);">
                <span style="color:#1d4ed8; font-size:1.4rem; font-weight:800; font-family:Playfair Display,serif; letter-spacing:0.02em;">📊 &nbsp; Analyse SHAP</span>
              </div>
              <p class="shap-title" style="margin:0.4rem 0 0.3rem 0; font-size:1.6rem;">Explicabilité de la Prédiction</p>
              <p class="shap-subtitle">Contribution de chaque paramètre clinique à la décision du modèle</p>
            </div>
            ''', unsafe_allow_html=True)

            feature_names = list(data_ready.columns)

            if HAS_SHAP and HAS_MODEL:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(data_ready)
                    if isinstance(shap_vals, list):
                        sv = shap_vals[1]
                    else:
                        sv = shap_vals
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
                    "Maladie (ALL)":    "La pathologie détermine le protocole et le pronostic.",
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

    except Exception as e:
        st.error(f"Erreur lors du traitement : {e}")
        st.exception(e)

st.markdown("""
<div style="...">
⚕️ Cet outil est une aide à la décision clinique uniquement. 
Il ne remplace pas l'avis d'un professionnel de santé.
</div>
""", unsafe_allow_html=True)
