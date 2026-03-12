import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# FONCTION 1 — load_data

def load_data(path='bone_marrow_clean.csv'):
    """
    Charge le dataset CSV nettoyé produit par l'EDA.
    Le fichier ARFF a déjà été converti et les NaN supprimés.
    """
    df = pd.read_csv(path)
    print(f' Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes')
    return df


# FONCTION 2 — handle_missing_values

def handle_missing_values(df):
    """
    Contexte médical : on supprime les lignes incomplètes.
    Imputer des valeurs fictives (médiane, moyenne) sur des données
    cliniques risque de fausser un pronostic réel → suppression choisie.
    Cette étape est dèja effectuée dans l'EDA.
    Cette fonction est une vérification défensive uniquement.
    """
    before = len(df)
    df = df.dropna()
    after = len(df)

    if before != after:
        print(f'⚠️  {before - after} lignes supprimées (NaN résiduels détectés).')
    else:
        print(' Aucune valeur manquante — dataset déjà propre.')

    return df



# FONCTION 3 — handle_outliers

def handle_outliers(df, target_col='survival_status'):
    """
    Détecte et traite les valeurs aberrantes sur les colonnes numériques
    en utilisant la méthode IQR (Interquartile Range).
    
    On exclut :
    - La colonne cible (survival_status)
    - Les colonnes binaires (0/1) car IQR n'a pas de sens sur elles
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclure la target ET les colonnes binaires
    num_cols = [col for col in num_cols 
                if col != target_col 
                and df[col].nunique() > 2]  

    outliers_count = {}

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            outliers_count[col] = n_outliers

        df[col] = df[col].clip(lower, upper)

    print(f' Outliers traités sur {len(num_cols)} colonnes numériques.')
    if outliers_count:
        print('   Colonnes affectées :')
        for col, count in sorted(outliers_count.items(), key=lambda x: -x[1]):
            print(f'   - {col} : {count} valeur(s) clippée(s)')
    else:
        print('   Aucun outlier détecté.')

    return df


# FONCTION 4 — encode_features

def encode_features(df):
    """
    Convertit les variables catégorielles (texte) en valeurs numériques.
    Les algorithmes ML ne peuvent pas traiter des chaînes de caractères.

    Deux stratégies selon le nombre de modalités :

    1. LabelEncoder → colonnes binaires (exactement 2 valeurs)
       Exemple : 'Gender' → {'Male': 0, 'Female': 1}
       Pourquoi : simple, efficace, ne crée pas de colonne supplémentaire

    2. One-Hot Encoding → colonnes avec 3+ valeurs (pd.get_dummies)
       Exemple : 'Disease' → colonnes Disease_ALL, Disease_AML, Disease_CML
       Pourquoi : évite de créer une fausse relation ordinale
       (ex : ALL=0, AML=1, CML=2 laisserait croire que AML > ALL)
       drop_first=True supprime une colonne redondante (évite multicolinéarité)
    """
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    if not cat_cols:
        print(' Aucune colonne catégorielle détectée.')
        return df

    # Séparer colonnes binaires et multi-catégorielles
    binary_cols = [col for col in cat_cols if df[col].nunique() == 2]
    multi_cols  = [col for col in cat_cols if df[col].nunique() > 2]

    # ── LabelEncoding pour les colonnes binaires ──
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f'   LabelEncoded  : {col}')

    # ── One-Hot Encoding pour les colonnes multi-catégorielles ──
    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        print(f'   One-Hot encoded : {multi_cols}')

    print(f' Encodage terminé. Shape : {df.shape}')
    return df


# FONCTION 5 — scale_features

def scale_features(df, target_col='survival_status'):
    """
    Normalise les variables numériques avec StandardScaler.
    Transforme chaque colonne : X_scaled = (X - moyenne) / écart-type
    Résultat : moyenne = 0, écart-type = 1 pour chaque feature.

    Pourquoi normaliser ?
    - SVM est très sensible à l'échelle : une variable 'âge' (0–100)
      ne doit pas dominer 'poids' (0–120) ni 'CD34' (0–20)
    - Sans normalisation, les features à grande échelle biaisent
      les distances et les coefficients des modèles linéaires

    Pourquoi exclure la target ?
    - survival_status est ce qu'on prédit : la modifier serait une erreur
    - Le scaler est retourné pour être réutilisé dans l'app Streamlit
     afin de transformer les données d'un nouveau patient
      avec exactement les mêmes paramètres mean/std
    """
    feature_cols = [col for col in df.columns if col != target_col]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    print(f' Normalisation appliquée sur {len(feature_cols)} features.')
    print(f'   Colonne cible exclue : {target_col}')

    # On retourne aussi le scaler → sera sauvegardé et réutilisé dans streamlit
    return df, scaler



# FONCTION 6 — balance_classes


def balance_classes(X, y):
    """
    Équilibre les classes avec SMOTE (Synthetic Minority Oversampling TEchnique).

    Pourquoi équilibrer ?
    - Dans les datasets médicaux, la classe minoritaire est souvent
      la plus importante cliniquement (ex : patients décédés)
    - Sans équilibrage, un modèle peut prédire toujours 'survie'
      et obtenir 80% d'accuracy sans rien apprendre d'utile
    - Les métriques comme F1-score et ROC-AUC sont faussées

    Pourquoi SMOTE et pas un simple oversampling ?
    - L'oversampling classique duplique des exemples existants
      → risque de surapprentissage (overfitting)
    - SMOTE génère des exemples synthétiques en interpolant
      entre les k plus proches voisins de la classe minoritaire
    - Résultat : le modèle apprend à mieux généraliser

    random_state=42 → résultats reproductibles entre les membres
    """
    print('Distribution avant SMOTE :')
    print(f'   {pd.Series(y).value_counts().to_dict()}')

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print('Distribution après SMOTE :')
    print(f'   {pd.Series(y_resampled).value_counts().to_dict()}')
    print(f' SMOTE appliqué : {X.shape[0]} → {X_resampled.shape[0]} exemples.')

    return X_resampled, y_resampled



# FONCTION 7 — optimize_memory


def optimize_memory(df):
    """
    Réduit l'empreinte mémoire du DataFrame en abaissant la précision
    des types numériques, sans perte significative d'information.

    Conversions effectuées :
    - float64 (8 octets) → float32 (4 octets)
    - int64   (8 octets) → int32   (4 octets)

    Pourquoi c'est sûr ?
    - float32 offre ~7 chiffres significatifs : largement suffisant
      pour des données médicales (âge, doses, taux biologiques)
    - int32 supporte jusqu'à ~2 milliards : bien au-delà des valeurs
      présentes dans ce dataset

    Impact attendu : réduction de ~40–50% de la RAM utilisée.
    Utile lors de l'entraînement de plusieurs modèles en parallèle.
    """
    mem_before = df.memory_usage(deep=True).sum() / 1024**2  # en MB

    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')

    mem_after = df.memory_usage(deep=True).sum() / 1024**2

    print(f' Optimisation mémoire :')
    print(f'   Avant  : {mem_before:.2f} MB')
    print(f'   Après  : {mem_after:.2f} MB')
    print(f'   Gain   : {(1 - mem_after / mem_before) * 100:.1f}%')

    return df


# PIPELINE COMPLET — run_preprocessing

def run_preprocessing(path='data/processed/bone_marrow_clean.csv', target_col='survival_status'):
    """
    Pipeline complet appelé dans train_model.py.
    Enchaîne toutes les étapes dans le bon ordre.

    Retourne :
    - X_balanced  : features prêtes pour l'entraînement
    - y_balanced  : labels équilibrés
    - scaler      : StandardScaler fitted → réutilisé par Membre 4 (Streamlit)
    """
    print('═' * 55)
    print('        PIPELINE PREPROCESSING — DÉBUT')
    print('═' * 55)

    df = load_data(path)                          # Étape 1
    df = handle_missing_values(df)                # Étape 2 (vérification)
    df = handle_outliers(df, target_col)          # Étape 3
    df = encode_features(df)                      # Étape 4
    df = optimize_memory(df)                      # Étape 5

    # Séparation features / target avant scaling et SMOTE
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_df, scaler = scale_features(              # Étape 6
        pd.concat([X, y], axis=1), target_col
    )
    X = X_df.drop(columns=[target_col])

    X_balanced, y_balanced = balance_classes(X, y)  # Étape 7

    print('═' * 55)
    print('        PIPELINE PREPROCESSING — TERMINÉ ')
    print(f'        Shape final : {X_balanced.shape}')
    print('═' * 55)

    return X_balanced, y_balanced, scaler

# TEST DU PIPELINE

if __name__ == '__main__':
    X, y, scaler = run_preprocessing(
        path='data/processed/bone_marrow_clean.csv',
        target_col='survival_status'
    )
    print(f'Shape X : {X.shape}')
    print(f'Shape y : {y.shape}')
