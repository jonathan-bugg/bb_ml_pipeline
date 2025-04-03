"""Unit tests for LGBMModel class."""

import os
import tempfile
import unittest
import pandas as pd
import numpy as np
from bb_ml_pipeline.models.lgbm_model import LGBMModel


class TestLGBMModel(unittest.TestCase):
    """Test cases for LGBMModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create simple test data for classification
        np.random.seed(42)
        n_samples = 100
        
        # Create 5 features with some correlation to target
        X = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.normal(0, 1, n_samples),
            'feature_4': np.random.normal(0, 1, n_samples),
            'feature_5': np.random.normal(0, 1, n_samples)
        })
        
        # Create target with some dependency on features
        y_proba = 1 / (1 + np.exp(-(0.5 * X['feature_1'] - 0.3 * X['feature_2'] + 0.7 * X['feature_3'])))
        y = (y_proba > 0.5).astype(int)
        
        self.X = X
        self.y = pd.Series(y)
        
        # Create a test model configuration
        self.model_config = {
            'target_variable': 'target',
            'problem_type': 'classification',
            'evaluation_metric': 'rocauc',
            'better_performance': 'gt',
            'models': ['lgbm'],
            'use_custom_splits': False,
            'retrain_with_whole_dataset': True,
            'importance_extraction_method': 'shap',
            'sample_for_contribution': 0.5,  # Use small sample for speed
            'num_hp_searches': 2,  # Reduce for faster tests
            'hyperparameter_space': {
                'lgbm': {
                    'boosting_type': "trial.suggest_categorical('boosting_type', ['gbdt'])",
                    'learning_rate': "trial.suggest_uniform('learning_rate', 0.05, 0.1)",
                    'num_leaves': "int(trial.suggest_discrete_uniform('num_leaves', 30, 50, 10))",
                    'feature_fraction': "trial.suggest_uniform('feature_fraction', 0.5, 0.8)",
                    'n_estimators': "int(trial.suggest_discrete_uniform('n_estimators', 50, 100, 50))"
                }
            }
        }
    
    def test_initialization(self):
        """Test LGBMModel initialization with configuration."""
        model = LGBMModel(self.model_config)
        
        # Check config parameters were set correctly
        self.assertEqual(model.problem_type, 'classification')
        self.assertEqual(model.evaluation_metric, 'rocauc')
        self.assertEqual(model.target_variable, 'target')
        self.assertEqual(model.importance_extraction_method, 'shap')
        self.assertEqual(model.num_hp_searches, 2)
    
    def test_train_model(self):
        """Test training a classification model."""
        model = LGBMModel(self.model_config)
        
        # Train the model
        training_results = model.train(self.X, self.y)
        
        # Check training completed successfully
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.best_params)
        self.assertIsNotNone(model.feature_importance)
        
        # Check performance metrics are available
        self.assertIn('cv_mean_train_score', training_results)
        self.assertIn('cv_mean_val_score', training_results)
        self.assertIn('cv_std_val_score', training_results)
        
        # Check mean validation score is reasonable (above 0.5)
        self.assertGreater(training_results['cv_mean_val_score'], 0.5)
    
    def test_predict(self):
        """Test making predictions with a trained model."""
        model = LGBMModel(self.model_config)
        model.train(self.X, self.y)
        
        # Make predictions
        predictions = model.predict(self.X.head(10))
        
        # Check predictions format
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(0 <= prob <= 1 for prob in predictions))
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        model = LGBMModel(self.model_config)
        model.train(self.X, self.y)
        
        # Check native importance is available
        self.assertIn('native', model.feature_importance)
        native_importance = model.feature_importance['native']
        
        # Check all features have importance values
        for feature in self.X.columns:
            self.assertIn(feature, native_importance)
        
        # Check SHAP importance is available
        self.assertIn('shap', model.feature_importance)
        shap_importance = model.feature_importance['shap']
        
        # Check SHAP importance has expected structure
        self.assertIn('importance', shap_importance)
        self.assertIn('sorted_importance', shap_importance)
        self.assertIn('values', shap_importance)
        self.assertIn('sample_indices', shap_importance)
    
    def test_treeinterpreter_importance(self):
        """Test TreeInterpreter feature importance extraction."""
        # Create a copy of the config with treeinterpreter
        config = self.model_config.copy()
        config['importance_extraction_method'] = 'treeinterpreter'
        
        model = LGBMModel(config)
        model.train(self.X, self.y)
        
        # Check treeinterpreter importance is available
        self.assertIn('treeinterpreter', model.feature_importance)
        ti_importance = model.feature_importance['treeinterpreter']
        
        # Check importance has expected structure
        self.assertIn('importance', ti_importance)
        self.assertIn('sorted_importance', ti_importance)
        self.assertIn('contributions', ti_importance)
        self.assertIn('bias', ti_importance)
    
    def test_save_load(self):
        """Test saving and loading a model."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_file = f.name
        
        try:
            # Create and train a model
            model = LGBMModel(self.model_config)
            model.train(self.X, self.y)
            
            # Make predictions with original model
            original_preds = model.predict(self.X.head(5))
            
            # Save the model
            model.save(temp_file)
            
            # Load the model
            loaded_model = LGBMModel.load(temp_file)
            
            # Check the loaded model has the same attributes
            self.assertEqual(model.problem_type, loaded_model.problem_type)
            self.assertEqual(model.target_variable, loaded_model.target_variable)
            
            # Check predictions are the same
            loaded_preds = loaded_model.predict(self.X.head(5))
            np.testing.assert_array_almost_equal(original_preds, loaded_preds)
        
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main() 