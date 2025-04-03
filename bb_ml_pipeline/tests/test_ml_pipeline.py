"""Integration tests for ML_Pipeline class."""

import os
import json
import tempfile
import unittest
import pandas as pd
import numpy as np
from bb_ml_pipeline.ml_pipeline import ML_Pipeline


class TestMLPipeline(unittest.TestCase):
    """Integration test cases for ML_Pipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create simple test data for classification
        np.random.seed(42)
        n_samples = 100
        
        # Create features
        self.data = pd.DataFrame({
            'id_col': range(1, n_samples + 1),
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
            'feature_4': np.random.choice([True, False], n_samples)
        })
        
        # Create target with some dependency on features
        y_proba = 1 / (1 + np.exp(-(0.5 * self.data['feature_1'] - 0.3 * self.data['feature_2'])))
        self.data['target'] = (y_proba > 0.5).astype(int)
        
        # Create train/test split
        train_indices = np.random.choice(range(n_samples), int(n_samples * 0.8), replace=False)
        self.train_data = self.data.iloc[train_indices].reset_index(drop=True)
        self.test_data = self.data.drop(train_indices).reset_index(drop=True)
        
        # Save training data
        self.train_data_path = os.path.join(self.temp_dir.name, 'train_data.csv')
        self.train_data.to_csv(self.train_data_path, index=False)
        
        # Save test data
        self.test_data_path = os.path.join(self.temp_dir.name, 'test_data.csv')
        self.test_data.to_csv(self.test_data_path, index=False)
        
        # Create feature metadata
        self.metadata = pd.DataFrame({
            'id_col': ['category', 0, 'unknown', 'NaN', 'id'],
            'target': ['bool', 1, 'unknown', 'NaN', 'target'],
            'feature_1': ['float', 1, 'mean', 'NaN', 'features'],
            'feature_2': ['float', 1, 'mean', 'NaN', 'features'],
            'feature_3': ['category', 1, 'unknown', 'one-hot', 'features'],
            'feature_4': ['bool', 1, 'unknown', 'NaN', 'features']
        })
        
        # Save metadata
        self.metadata_path = os.path.join(self.temp_dir.name, 'feature_metadata.xlsx')
        self.metadata.to_excel(self.metadata_path, index=False)
        
        # Create model configuration
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
        
        # Save model configuration
        self.model_config_path = os.path.join(self.temp_dir.name, 'model_config.json')
        with open(self.model_config_path, 'w') as f:
            json.dump(self.model_config, f)
        
        # Create training data configuration
        self.train_config = {
            'training': True,
            'metadata_file_path': self.metadata_path,
            'input_data_source': 'CSV',
            'file_path': self.train_data_path
        }
        
        # Save training data configuration
        self.train_config_path = os.path.join(self.temp_dir.name, 'train_config.json')
        with open(self.train_config_path, 'w') as f:
            json.dump(self.train_config, f)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_training_pipeline(self):
        """Test the full training pipeline."""
        # Initialize and run the pipeline
        pipeline = ML_Pipeline(
            modeling_config=self.model_config_path,
            data_config=self.train_config_path
        )
        
        results = pipeline.run()
        
        # Check pipeline run successfully
        self.assertTrue(results.get('success', False))
        self.assertIsNotNone(results.get('output_dir'))
        
        # Check output directory exists and contains expected files
        output_dir = results['output_dir']
        self.assertTrue(os.path.exists(output_dir))
        
        # Check for model file
        model_files = [f for f in os.listdir(output_dir) if f.endswith('.pkl')]
        self.assertGreater(len(model_files), 0)
        
        # Check for predictions
        predictions_dir = os.path.join(output_dir, 'predictions')
        self.assertTrue(os.path.exists(predictions_dir))
        self.assertTrue(os.path.exists(os.path.join(predictions_dir, 'original_data_with_predictions.csv')))
        
        # Check for feature importance
        insights_dir = os.path.join(output_dir, 'deep_insights_input')
        self.assertTrue(os.path.exists(insights_dir))
        self.assertTrue(os.path.exists(os.path.join(insights_dir, 'native_feature_importance.csv')))
        
        # Store model path for prediction test
        self.model_path = os.path.join(output_dir, 'target_classification_lgbm.pkl')
        self.processor_path = os.path.join(
            output_dir, 'deep_insights_input', 'target_classification_lgbm_data_processor.pkl')
        
        return output_dir
    
    def test_full_pipeline_train_and_predict(self):
        """Test training and then prediction with the pipeline."""
        # First train the model
        output_dir = self.test_training_pipeline()
        
        # Create prediction data configuration
        predictions_dir = os.path.join(self.temp_dir.name, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        predictions_path = os.path.join(predictions_dir, 'predictions.csv')
        
        test_config = {
            'training': False,
            'metadata_file_path': self.metadata_path,
            'input_data_source': 'CSV',
            'file_path': self.test_data_path,
            'data_processor_file_path': self.processor_path,
            'model_file_path': self.model_path,
            'uid_column_name': 'id_col',
            'predictions_output_path': predictions_path,
            'calculate_importance': True
        }
        
        # Save test data configuration
        test_config_path = os.path.join(self.temp_dir.name, 'test_config.json')
        with open(test_config_path, 'w') as f:
            json.dump(test_config, f)
        
        # Initialize and run the prediction pipeline
        pipeline = ML_Pipeline(
            modeling_config=self.model_config_path,
            data_config=test_config_path
        )
        
        results = pipeline.run()
        
        # Check pipeline run successfully
        self.assertTrue(results.get('success', False))
        
        # Check predictions file exists
        self.assertTrue(os.path.exists(predictions_path))
        
        # Load and check predictions
        predictions_df = pd.read_csv(predictions_path)
        
        # Check format
        self.assertIn('id_col', predictions_df.columns)
        self.assertIn('prediction_proba', predictions_df.columns)
        self.assertIn('prediction_class', predictions_df.columns)
        
        # Check predictions are reasonable
        self.assertEqual(len(predictions_df), len(self.test_data))
        self.assertTrue(all(0 <= prob <= 1 for prob in predictions_df['prediction_proba']))
        self.assertTrue(all(pred in [0, 1] for pred in predictions_df['prediction_class']))


if __name__ == '__main__':
    unittest.main() 