"""Data loader module for the ML Pipeline."""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional


class DataLoader:
    """
    Data loader class for the ML Pipeline.
    
    This class handles loading data from different sources (currently CSV supported),
    and loading metadata from Excel or JSON.
    """
    
    def __init__(self, data_config_path: str):
        """
        Initialize the DataLoader with a data configuration file.
        
        Args:
            data_config_path: Path to the data configuration JSON file.
        """
        self.data_config = self._load_json_config(data_config_path)
        self.is_training = self.data_config.get('training', False)
        self.metadata = None
        self.data = None
        self.target_variable = None
        
    def _load_json_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load JSON configuration file.
        
        Args:
            config_path: Path to the JSON configuration file.
            
        Returns:
            Dict containing the configuration.
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            json.JSONDecodeError: If the configuration file is not valid JSON.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as file:
            return json.load(file)
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load feature metadata from file specified in the data configuration.
        
        Returns:
            DataFrame containing the metadata.
        """
        metadata_path = self.data_config.get('metadata_file_path')
        if not metadata_path:
            raise ValueError("Metadata file path not specified in data configuration")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        if metadata_path.endswith('.xlsx'):
            self.metadata = pd.read_excel(metadata_path)
        elif metadata_path.endswith('.json'):
            with open(metadata_path, 'r') as file:
                metadata_json = json.load(file)
                self.metadata = pd.DataFrame(metadata_json)
        else:
            raise ValueError(f"Unsupported metadata file format: {metadata_path}")
        
        return self.metadata
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from the source specified in the data configuration.
        
        Returns:
            DataFrame containing the loaded data.
            
        Raises:
            ValueError: If the data source is not supported.
        """
        data_source = self.data_config.get('input_data_source')
        file_path = self.data_config.get('file_path')
        
        if not file_path:
            raise ValueError("Data file path not specified in data configuration")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if data_source == 'CSV':
            self.data = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
        
        return self.data
    
    def get_target_and_features(self, modeling_config: Dict[str, Any]) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Extract target variable and feature variables from the loaded data.
        
        Args:
            modeling_config: Modeling configuration containing the target variable.
            
        Returns:
            Tuple of (target series, features dataframe)
            
        Raises:
            ValueError: If the target variable is not found in the data.
        """
        if self.data is None:
            self.load_data()
            
        if not self.is_training:
            # For prediction mode, we may not have the target variable
            target_variable = modeling_config.get('target_variable')
            if target_variable in self.data.columns:
                y = self.data[target_variable]
                X = self.data.drop(columns=[target_variable])
            else:
                y = None
                X = self.data
        else:
            # For training mode, we must have the target variable
            target_variable = modeling_config.get('target_variable')
            if target_variable not in self.data.columns:
                raise ValueError(f"Target variable '{target_variable}' not found in the data")
            
            y = self.data[target_variable]
            X = self.data.drop(columns=[target_variable])
        
        return y, X
    
    def validate_data(self, modeling_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that the data meets the requirements for training or prediction.
        
        Args:
            modeling_config: Modeling configuration.
            
        Returns:
            Dictionary with validation results.
            
        Raises:
            ValueError: If validation fails.
        """
        if self.data is None:
            self.load_data()
            
        if self.metadata is None:
            self.load_metadata()
        
        validation_results = {
            'passed': True,
            'errors': [],
            'warnings': []
        }
        
        # Extract feature info from metadata
        # Assuming first row contains the feature types
        feature_types = self.metadata.iloc[0].to_dict()
        
        # Check if target exists in training mode
        if self.is_training:
            target_variable = modeling_config.get('target_variable')
            if target_variable not in self.data.columns:
                validation_results['passed'] = False
                validation_results['errors'].append(f"Target variable '{target_variable}' not found in the data")
            else:
                # Check for nulls in target variable
                null_count = self.data[target_variable].isna().sum()
                if null_count > 0:
                    validation_results['passed'] = False
                    validation_results['errors'].append(f"Target variable '{target_variable}' contains {null_count} null values")
        
        # Check feature types and null values
        for col, col_type in feature_types.items():
            if col in self.data.columns:
                # Check data type consistency
                if col_type == 'float':
                    if not pd.api.types.is_numeric_dtype(self.data[col]):
                        validation_results['warnings'].append(f"Column '{col}' should be float but has non-numeric values")
                elif col_type == 'bool':
                    if not pd.api.types.is_bool_dtype(self.data[col]) and not set(self.data[col].dropna().unique()).issubset({0, 1, True, False}):
                        validation_results['warnings'].append(f"Column '{col}' should be boolean but has non-boolean values")
                elif col_type == 'category':
                    # Categories don't need validation for type
                    pass
            else:
                # Column in metadata but not in data
                # Check if this is a feature that should be used (use=1)
                use_row = self.metadata.iloc[1]
                if col in use_row and use_row[col] == 1:
                    validation_results['errors'].append(f"Required feature '{col}' is missing from the data")
                    validation_results['passed'] = False
        
        return validation_results 