import streamlit as st
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Aide à la Décision Médicale", layout="wide")

# Titre principal
st.title("🩺 Prédiction de Succès : Greffe de Moelle Osseuse")
st.markdown("---")

# Sidebar
st.sidebar.header("Options")
st.sidebar.info("Interface de test - Groupe 24")

# Formulaire de saisie
with st.form("patient_data"):
    st.subheader("Informations sur le Patient")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Âge du patient (années)", 0, 18, 10)
        source = st.selectbox("Source des cellules", ["Moelle Osseuse", "Sang Périphérique"])
    
    with col2:
        poids = st.number_input("Poids du patient (kg)", 5.0, 100.0, 30.0)
        maladie = st.selectbox("Type de Maladie", ["ALL", "AML", "Non-malignant"])
    
    submit = st.form_submit_button("Calculer la probabilité")

# Résultats simulés (votre futur modèle viendra ici)
if submit:
    st.divider()
    score = np.random.uniform(75, 98)
    st.metric("Probabilité de succès", f"{score:.2f}%")
    
    st.subheader("Analyse de l'Explicabilité (SHAP)")
    st.info("Cette section affichera l'impact de chaque paramètre sur le score final.")
    # On met un graphique vide pour la démo
    st.bar_chart(np.random.randn(5, 1))