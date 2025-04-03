"""Unit tests for DataProcessor class."""

import os
import tempfile
import unittest
import pandas as pd
import numpy as np
from bb_ml_pipeline.data.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        self.data = pd.DataFrame({
            'id_col': [1, 2, 3, 4, 5],
            'float_feature': [1.5, 2.5, None, 4.5, 5.5],
            'cat_feature': ['A', 'B', 'A', 'C', None],
            'bool_feature': [True, False, True, None, False]
        })
        
        # Create test metadata
        self.metadata = pd.DataFrame({
            'id_col': ['category', 0, 'unknown', 'one-hot', '1_id'],
            'float_feature': ['float', 1, 'mean', 'NaN', '3_features'],
            'cat_feature': ['category', 1, 'unknown', 'one-hot', '3_features'],
            'bool_feature': ['bool', 1, 'unknown', 'NaN', '3_features']
        })
    
    def test_initialization(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor(self.metadata)
        
        # Check metadata was parsed correctly
        self.assertEqual(processor.feature_types['float_feature'], 'float')
        self.assertEqual(processor.feature_usage['float_feature'], 1)
        self.assertEqual(processor.feature_usage['id_col'], 0)
        self.assertEqual(processor.imputation_methods['float_feature'], 'mean')
        self.assertEqual(processor.category_transformations['cat_feature'], 'one-hot')
    
    def test_get_columns_by_type(self):
        """Test grouping columns by type."""
        processor = DataProcessor(self.metadata)
        feature_names = ['float_feature', 'cat_feature', 'bool_feature']
        
        numeric, categorical, boolean = processor._get_columns_by_type(feature_names)
        
        # Check columns were grouped correctly
        self.assertEqual(numeric, ['float_feature'])
        self.assertEqual(categorical, ['cat_feature'])
        self.assertEqual(boolean, ['bool_feature'])
    
    def test_get_feature_names(self):
        """Test getting feature names based on metadata usage."""
        processor = DataProcessor(self.metadata)
        feature_names = processor._get_feature_names(self.data)
        
        # Check only columns with usage=1 are included
        self.assertEqual(set(feature_names), {'float_feature', 'cat_feature', 'bool_feature'})
        self.assertNotIn('id_col', feature_names)
    
    def test_fit_transform(self):
        """Test fitting and transforming data."""
        processor = DataProcessor(self.metadata)
        X_transformed = processor.fit_transform(self.data)
        
        # Check transformed data has the expected structure
        self.assertIsNotNone(X_transformed)
        self.assertIsInstance(X_transformed, pd.DataFrame)
        
        # Check missing values were handled
        self.assertFalse(X_transformed.isna().any().any())
        
        # Check one-hot encoding was applied to categorical features
        self.assertIn('cat_feature_A', X_transformed.columns)
        self.assertIn('cat_feature_B', X_transformed.columns)
        self.assertIn('cat_feature_C', X_transformed.columns)
        
        # Check missing indicators were created
        self.assertIn('float_feature_missing', X_transformed.columns)
        self.assertIn('cat_feature_missing', X_transformed.columns)
        self.assertIn('bool_feature_missing', X_transformed.columns)
    
    def test_save_load(self):
        """Test saving and loading a DataProcessor."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_file = f.name
        
        try:
            # Create and fit a processor
            processor = DataProcessor(self.metadata)
            processor.fit(self.data)
            
            # Save the processor
            processor.save(temp_file)
            
            # Load the processor
            loaded_processor = DataProcessor.load(temp_file)
            
            # Check the loaded processor has the same attributes
            self.assertEqual(processor.feature_types, loaded_processor.feature_types)
            self.assertEqual(processor.feature_usage, loaded_processor.feature_usage)
            self.assertEqual(processor.feature_names, loaded_processor.feature_names)
            
            # Check the loaded processor can transform data
            X_transformed_original = processor.transform(self.data)
            X_transformed_loaded = loaded_processor.transform(self.data)
            
            pd.testing.assert_frame_equal(X_transformed_original, X_transformed_loaded)
        
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main() 