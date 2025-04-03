"""Unit tests for DataLoader class."""

import os
import json
import tempfile
import unittest
import pandas as pd
import numpy as np
from bb_ml_pipeline.data.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test CSV data
        self.data = pd.DataFrame({
            'id_col': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0],
            'feature_1': [1.5, 2.5, 3.5, 4.5, 5.5],
            'feature_2': ['A', 'B', 'A', 'C', 'B'],
            'feature_3': [True, False, True, True, False]
        })
        
        # Save test data
        self.data_path = os.path.join(self.temp_dir.name, 'test_data.csv')
        self.data.to_csv(self.data_path, index=False)
        
        # Create test metadata
        self.metadata = pd.DataFrame({
            'id_col': ['category', 0, 'unknown', 'NaN', '1_id'],
            'target': ['bool', 1, 'unknown', 'NaN', '2_DEPENDENT'],
            'feature_1': ['float', 1, 'mean', 'NaN', '3_features'],
            'feature_2': ['category', 1, 'unknown', 'one-hot', '3_features'],
            'feature_3': ['bool', 1, 'unknown', 'NaN', '3_features']
        })
        
        # Save test metadata
        self.metadata_path = os.path.join(self.temp_dir.name, 'test_metadata.xlsx')
        self.metadata.to_excel(self.metadata_path, index=False)
        
        # Create data config
        self.config = {
            'training': True,
            'metadata_file_path': self.metadata_path,
            'input_data_source': 'CSV',
            'file_path': self.data_path
        }
        
        # Save data config
        self.config_path = os.path.join(self.temp_dir.name, 'test_config.json')
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader(self.config_path)
        self.assertEqual(loader.data_config, self.config)
        self.assertTrue(loader.is_training)
    
    def test_load_data(self):
        """Test loading data from CSV."""
        loader = DataLoader(self.config_path)
        loaded_data = loader.load_data()
        
        # Check data was loaded correctly
        pd.testing.assert_frame_equal(loaded_data, self.data)
    
    def test_load_metadata(self):
        """Test loading metadata from Excel."""
        loader = DataLoader(self.config_path)
        loaded_metadata = loader.load_metadata()
        
        # Check first column values to confirm metadata was loaded
        self.assertEqual(loaded_metadata.iloc[0]['id_col'], 'category')
        self.assertEqual(loaded_metadata.iloc[1]['target'], 1)
    
    def test_get_target_and_features(self):
        """Test extracting target and features."""
        loader = DataLoader(self.config_path)
        loader.load_data()
        
        # Create a modeling config for the test
        modeling_config = {'target_variable': 'target'}
        
        y, X = loader.get_target_and_features(modeling_config)
        
        # Check target was extracted correctly
        pd.testing.assert_series_equal(y, self.data['target'])
        
        # Check features were extracted correctly (excluding target)
        expected_features = self.data.drop(columns=['target'])
        pd.testing.assert_frame_equal(X, expected_features)
    
    def test_validate_data_success(self):
        """Test successful data validation."""
        loader = DataLoader(self.config_path)
        loader.load_data()
        loader.load_metadata()
        
        # Create a modeling config for the test
        modeling_config = {'target_variable': 'target'}
        
        validation_results = loader.validate_data(modeling_config)
        
        # Check validation passed
        self.assertTrue(validation_results['passed'])
        self.assertEqual(len(validation_results['errors']), 0)
    
    def test_validate_data_missing_target(self):
        """Test validation with missing target variable."""
        # Create data without target variable
        data_no_target = self.data.drop(columns=['target'])
        data_path = os.path.join(self.temp_dir.name, 'test_data_no_target.csv')
        data_no_target.to_csv(data_path, index=False)
        
        # Update config to point to the new data
        config = self.config.copy()
        config['file_path'] = data_path
        config_path = os.path.join(self.temp_dir.name, 'test_config_no_target.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Create loader and validate
        loader = DataLoader(config_path)
        loader.load_data()
        loader.load_metadata()
        
        # Create a modeling config for the test
        modeling_config = {'target_variable': 'target'}
        
        validation_results = loader.validate_data(modeling_config)
        
        # Check validation failed
        self.assertFalse(validation_results['passed'])
        self.assertTrue(any('target' in error for error in validation_results['errors']))


if __name__ == '__main__':
    unittest.main() 