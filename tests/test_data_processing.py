# tests pour le traitements de données 

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ajouter le chemin src pour pouvoir importer les modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importer les fonctions depuis le module de votre camarade
from src.data_processing import (
    load_data,
    handle_missing_values,
    handle_outliers,
    encode_features,
    scale_features,
    balance_classes,
    optimize_memory,
    run_preprocessing
)


class TestLoadData:
    """Tests pour la fonction load_data."""
    
    def test_load_data_file_exists(self, tmp_path):
        """Vérifie que load_data fonctionne avec un fichier existant."""
        # Créer un fichier CSV temporaire
        test_file = tmp_path / "test_data.csv"
        df_test = pd.DataFrame({
            'age': [25, 30, 35],
            'survival_status': [0, 1, 0]
        })
        df_test.to_csv(test_file, index=False)
        
        # Tester le chargement
        df_result = load_data(str(test_file))
        
        assert isinstance(df_result, pd.DataFrame)
        assert len(df_result) == 3
        assert 'age' in df_result.columns
        assert 'survival_status' in df_result.columns
    
    def test_load_data_file_not_found(self):
        """Vérifie que load_data lève une erreur si fichier inexistant."""
        with pytest.raises(FileNotFoundError):
            load_data("fichier_inexistant.csv")


class TestHandleMissingValues:
    """Tests pour la fonction handle_missing_values."""
    
    def test_drop_missing_values(self):
        """Vérifie que les lignes avec NaN sont supprimées."""
        # Créer DataFrame avec valeurs manquantes
        df_test = pd.DataFrame({
            'age': [25, 30, None, 35, None],
            'blood_pressure': [120, None, 130, 125, 140],
            'survival_status': [0, 1, 0, 1, 0]
        })
        
        df_result = handle_missing_values(df_test)
        
        # Vérifier qu'il n'y a plus de NaN
        assert df_result.isnull().sum().sum() == 0
        # Vérifier que des lignes ont été supprimées
        assert len(df_result) < len(df_test)
        # Vérifier le message de suppression
        assert len(df_result) == 2  # Seules les lignes sans NaN
    
    def test_no_missing_values(self):
        """Vérifie le comportement quand il n'y a pas de NaN."""
        df_test = pd.DataFrame({
            'age': [25, 30, 35],
            'blood_pressure': [120, 130, 125],
            'survival_status': [0, 1, 0]
        })
        
        df_result = handle_missing_values(df_test)
        
        # Le DataFrame ne devrait pas changer
        assert len(df_result) == len(df_test)
        assert df_result.isnull().sum().sum() == 0
        pd.testing.assert_frame_equal(df_result, df_test)


class TestHandleOutliers:
    """Tests pour la fonction handle_outliers."""
    
    def test_outliers_clipped(self):
        """Vérifie que les outliers sont bien clippés avec IQR."""
        # Créer données avec outliers évidents
        df_test = pd.DataFrame({
            'age': [25, 30, 35, 40, 200, 45, 28, 32],  # 200 est outlier
            'blood_pressure': [120, 125, 130, 118, 122, 1000, 124, 126],  # 1000 outlier
            'survival_status': [0, 1, 0, 1, 0, 1, 0, 1],  # target - doit être ignorée
            'binary_feature': [0, 1, 0, 1, 0, 1, 0, 1]  # binaire - doit être ignorée
        })
        
        df_result = handle_outliers(df_test, target_col='survival_status')
        
        # Vérifier que les outliers ont été clippés
        assert df_result['age'].max() < 100
        assert df_result['blood_pressure'].max() < 200
        
        # Vérifier que la target et les features binaires n'ont pas changé
        assert df_result['survival_status'].tolist() == df_test['survival_status'].tolist()
        assert df_result['binary_feature'].tolist() == df_test['binary_feature'].tolist()
        
        # Vérifier que les valeurs normales n'ont pas changé
        normal_ages = df_test[df_test['age'] < 100]['age']
        assert all(age in df_result['age'].values for age in normal_ages)
    
    def test_no_outliers(self):
        """Vérifie le comportement quand il n'y a pas d'outliers."""
        df_test = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'blood_pressure': [120, 125, 130, 118, 122],
            'survival_status': [0, 1, 0, 1, 0]
        })
        
        df_result = handle_outliers(df_test, target_col='survival_status')
        
        # Les données ne devraient pas changer
        pd.testing.assert_frame_equal(df_result, df_test)
    
    def test_outliers_count_report(self, capsys):
        """Vérifie que le rapport du nombre d'outliers est correct."""
        df_test = pd.DataFrame({
            'age': [25, 30, 35, 200, 40, 45],
            'survival_status': [0, 1, 0, 1, 0, 1]
        })
        
        handle_outliers(df_test, target_col='survival_status')
        captured = capsys.readouterr()
        
        assert 'Colonnes affectées' in captured.out
        assert 'age' in captured.out


class TestEncodeFeatures:
    """Tests pour la fonction encode_features."""
    
    def test_encode_binary_columns(self):
        """Vérifie que les colonnes binaires sont encodées avec LabelEncoder."""
        df_test = pd.DataFrame({
            'gender': ['Male', 'Female', 'Male', 'Female'],
            'donor_type': ['Related', 'Unrelated', 'Related', 'Unrelated'],
            'survival_status': [0, 1, 0, 1]
        })
        
        df_result = encode_features(df_test)
        
        # Vérifier que les colonnes binaires sont devenues numériques
        assert df_result['gender'].dtype in ['int32', 'int64']
        assert df_result['donor_type'].dtype in ['int32', 'int64']
        
        # Vérifier les valeurs encodées (ordre peut varier)
        assert set(df_result['gender'].unique()) == {0, 1}
        assert set(df_result['donor_type'].unique()) == {0, 1}
    
    def test_encode_multi_category(self):
        """Vérifie que les colonnes multi-catégories sont one-hot encodées."""
        df_test = pd.DataFrame({
            'disease': ['ALL', 'AML', 'CML', 'ALL', 'AML'],
            'blood_type': ['A', 'B', 'O', 'AB', 'A'],
            'survival_status': [0, 1, 0, 1, 0]
        })
        
        df_result = encode_features(df_test)
        
        # Les colonnes originales devraient avoir disparu
        assert 'disease' not in df_result.columns
        assert 'blood_type' not in df_result.columns
        
        # De nouvelles colonnes devraient apparaître
        assert any(col.startswith('disease_') for col in df_result.columns)
        assert any(col.startswith('blood_type_') for col in df_result.columns)
        
        # Vérifier le nombre de colonnes (n_categories - 1 à cause de drop_first=True)
        n_disease_cols = len([c for c in df_result.columns if c.startswith('disease_')])
        n_blood_cols = len([c for c in df_result.columns if c.startswith('blood_type_')])
        
        assert n_disease_cols == 3  # 4 catégories - 1
        assert n_blood_cols == 3    # 4 catégories - 1
    
    def test_mixed_encoding(self):
        """Vérifie le mélange de LabelEncoder et One-Hot."""
        df_test = pd.DataFrame({
            'gender': ['Male', 'Female', 'Male'],  # binaire
            'disease': ['ALL', 'AML', 'CML'],       # multi
            'survival_status': [0, 1, 0]
        })
        
        df_result = encode_features(df_test)
        
        # gender devrait être LabelEncoded (binaire)
        assert 'gender' in df_result.columns
        assert df_result['gender'].dtype in ['int32', 'int64']
        
        # disease devrait être one-hot
        assert 'disease' not in df_result.columns
        assert any(col.startswith('disease_') for col in df_result.columns)
    
    def test_no_categorical_columns(self):
        """Vérifie le comportement quand il n'y a pas de colonnes catégorielles."""
        df_test = pd.DataFrame({
            'age': [25, 30, 35],
            'survival_status': [0, 1, 0]
        })
        
        df_result = encode_features(df_test)
        
        # Le DataFrame ne devrait pas changer
        pd.testing.assert_frame_equal(df_result, df_test)


class TestScaleFeatures:
    """Tests pour la fonction scale_features."""
    
    def test_scaling_applied(self):
        """Vérifie que le scaling est bien appliqué."""
        df_test = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'blood_pressure': [120, 125, 130, 135, 140],
            'survival_status': [0, 1, 0, 1, 0]
        })
        
        df_scaled, scaler = scale_features(df_test, target_col='survival_status')
        
        # Vérifier que le scaler est retourné
        assert scaler is not None
        assert hasattr(scaler, 'mean_')
        
        # Vérifier que les features sont centrées (moyenne ~0)
        assert abs(df_scaled['age'].mean()) < 1e-10
        assert abs(df_scaled['blood_pressure'].mean()) < 1e-10
        
        # Vérifier que l'écart-type est ~1
        assert abs(df_scaled['age'].std() - 1) < 1e-10
        assert abs(df_scaled['blood_pressure'].std() - 1) < 1e-10
        
        # Vérifier que la target n'a pas été scalée
        assert df_scaled['survival_status'].tolist() == df_test['survival_status'].tolist()
    
    def test_scaler_consistency(self):
        """Vérifie que le scaler peut être réutilisé."""
        df_train = pd.DataFrame({
            'age': [25, 30, 35],
            'survival_status': [0, 1, 0]
        })
        
        df_train_scaled, scaler = scale_features(df_train, target_col='survival_status')
        
        # Nouvelles données
        df_new = pd.DataFrame({
            'age': [40, 45],
            'survival_status': [1, 0]
        })
        
        # Appliquer le même scaler
        feature_cols = [c for c in df_new.columns if c != 'survival_status']
        df_new[feature_cols] = scaler.transform(df_new[feature_cols])
        
        # Vérifier la transformation
        expected_age_40 = (40 - scaler.mean_[0]) / scaler.scale_[0]
        assert abs(df_new.loc[0, 'age'] - expected_age_40) < 1e-10


class TestBalanceClasses:
    """Tests pour la fonction balance_classes."""
    
    def test_smote_balancing(self):
        """Vérifie que SMOTE équilibre les classes."""
        # Créer données déséquilibrées (80-20)
        np.random.seed(42)
        X_test = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y_test = pd.Series([0] * 80 + [1] * 20)  # 80% classe 0, 20% classe 1
        
        X_balanced, y_balanced = balance_classes(X_test, y_test)
        
        # Vérifier l'équilibre
        unique, counts = np.unique(y_balanced, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        # Les classes devraient être équilibrées
        assert class_counts[0] == class_counts[1]
        assert class_counts[0] == 80  # La classe majoritaire détermine la taille
        
        # Vérifier que le nombre d'échantillons a augmenté
        assert len(X_balanced) > len(X_test)
        assert len(X_balanced) == 160  # 80 + 80
    
    def test_smote_output_type(self):
        """Vérifie que SMOTE retourne bien des DataFrames/Series."""
        X_test = pd.DataFrame({
            'feature1': range(20),
            'feature2': range(20, 40)
        })
        y_test = pd.Series([0] * 15 + [1] * 5)
        
        X_balanced, y_balanced = balance_classes(X_test, y_test)
        
        # Vérifier les types
        assert isinstance(X_balanced, pd.DataFrame)
        assert isinstance(y_balanced, pd.Series)
        
        # Vérifier les colonnes
        assert list(X_balanced.columns) == ['feature1', 'feature2']
    
    def test_smote_report(self, capsys):
        """Vérifie que SMOTE affiche les distributions."""
        X_test = pd.DataFrame({'feature': range(20)})
        y_test = pd.Series([0] * 15 + [1] * 5)
        
        balance_classes(X_test, y_test)
        captured = capsys.readouterr()
        
        assert "Distribution avant SMOTE" in captured.out
        assert "Distribution après SMOTE" in captured.out
        assert "0: 15" in captured.out
        assert "1: 15" in captured.out


class TestOptimizeMemory:
    """Tests pour la fonction optimize_memory."""
    
    def test_memory_reduction(self):
        """Vérifie que optimize_memory réduit bien la mémoire."""
        # Créer DataFrame avec types 64 bits
        df_test = pd.DataFrame({
            'int64_col': np.random.randint(0, 100, 1000),
            'float64_col': np.random.random(1000),
            'survival_status': np.random.randint(0, 2, 1000)
        })
        
        # Forcer les types à 64 bits
        df_test['int64_col'] = df_test['int64_col'].astype('int64')
        df_test['float64_col'] = df_test['float64_col'].astype('float64')
        
        mem_before = df_test.memory_usage(deep=True).sum()
        
        df_optimized = optimize_memory(df_test)
        
        mem_after = df_optimized.memory_usage(deep=True).sum()
        
        # Vérifier la réduction mémoire
        assert mem_after < mem_before
        assert df_optimized['int64_col'].dtype == 'int32'
        assert df_optimized['float64_col'].dtype == 'float32'
        
        # Vérifier que les données n'ont pas changé (aux arrondis près pour float)
        assert df_optimized['int64_col'].tolist() == df_test['int64_col'].tolist()
        np.testing.assert_array_almost_equal(
            df_optimized['float64_col'].values, 
            df_test['float64_col'].values, 
            decimal=5
        )
    
    def test_memory_report(self, capsys):
        """Vérifie que optimize_memory affiche un rapport."""
        df_test = pd.DataFrame({
            'int64_col': np.random.randint(0, 100, 100).astype('int64'),
            'float64_col': np.random.random(100).astype('float64')
        })
        
        optimize_memory(df_test)
        captured = capsys.readouterr()
        
        assert "Optimisation mémoire" in captured.out
        assert "Avant" in captured.out
        assert "Après" in captured.out
        assert "Gain" in captured.out


class TestFullPipeline:
    """Tests pour le pipeline complet."""
    
    def test_pipeline_integration(self, tmp_path):
        """Test l'intégration de tout le pipeline."""
        # Créer un dataset synthétique complet
        df_synthetic = pd.DataFrame({
            'age': [25, 30, None, 200, 40, 45, 28, 32],  # missing + outlier
            'gender': ['M', 'F', 'M', None, 'F', 'M', 'F', 'M'],  # missing
            'disease': ['ALL', 'AML', 'CML', 'ALL', 'AML', 'CML', 'ALL', 'AML'],
            'survival_status': [0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Sauvegarder dans un fichier temporaire
        test_file = tmp_path / "test_bone_marrow.csv"
        df_synthetic.to_csv(test_file, index=False)
        
        # Exécuter le pipeline complet
        X, y, scaler = run_preprocessing(
            path=str(test_file),
            target_col='survival_status'
        )
        
        # Vérifications
        assert X is not None
        assert y is not None
        assert scaler is not None
        
        # Vérifier qu'il n'y a pas de valeurs manquantes
        assert X.isnull().sum().sum() == 0
        
        # Vérifier que les classes sont équilibrées
        unique, counts = np.unique(y, return_counts=True)
        assert abs(counts[0] - counts[1]) <= 1
        
        # Vérifier que le scaler est fitted
        assert hasattr(scaler, 'mean_')
        assert len(scaler.mean_) == X.shape[1]
    
    def test_pipeline_reproducibility(self, tmp_path):
        """Vérifie que le pipeline est reproductible."""
        # Mêmes données
        df_synthetic = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'gender': ['M', 'F', 'M', 'F', 'M'],
            'survival_status': [0, 1, 0, 1, 0]
        })
        
        test_file = tmp_path / "test_data.csv"
        df_synthetic.to_csv(test_file, index=False)
        
        # Exécuter deux fois
        X1, y1, scaler1 = run_preprocessing(str(test_file), target_col='survival_status')
        X2, y2, scaler2 = run_preprocessing(str(test_file), target_col='survival_status')
        
        # Les résultats devraient être identiques
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)
        np.testing.assert_array_equal(scaler1.mean_, scaler2.mean_)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])