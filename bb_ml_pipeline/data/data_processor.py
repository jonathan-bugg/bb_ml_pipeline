"""Data processor module for the ML Pipeline."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os


class DataProcessor:
    """
    Data processor class for the ML Pipeline.
    
    This class handles data transformations according to the metadata,
    including imputation, encoding, and scaling.
    """
    
    def __init__(self, metadata: pd.DataFrame):
        """
        Initialize the DataProcessor with metadata.
        
        Args:
            metadata: DataFrame containing feature metadata.
        """
        self.metadata = metadata
        self.feature_types = self.metadata.iloc[0].to_dict()
        self.feature_usage = self.metadata.iloc[1].to_dict()
        self.imputation_methods = self.metadata.iloc[2].to_dict()
        self.category_transformations = self.metadata.iloc[3].to_dict()
        self.super_categories = self.metadata.iloc[4].to_dict()
        
        # Will be initialized during fit
        self.preprocessor = None
        self.feature_names = None
        self.feature_names_after_transform = None
        self.categorical_columns = None
        self.numeric_columns = None
        self.bool_columns = None
        self.missing_indicator_columns = []
        
    def _get_columns_by_type(self, feature_names: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Group features by their type (numeric, categorical, boolean).
        
        Args:
            feature_names: List of feature names to be grouped.
            
        Returns:
            Tuple of (numeric columns, categorical columns, boolean columns)
        """
        numeric_cols = []
        categorical_cols = []
        bool_cols = []
        
        for col in feature_names:
            if col in self.feature_types:
                col_type = self.feature_types[col]
                if col_type == 'float':
                    numeric_cols.append(col)
                elif col_type == 'category':
                    categorical_cols.append(col)
                elif col_type == 'bool':
                    bool_cols.append(col)
        
        return numeric_cols, categorical_cols, bool_cols
    
    def _get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """
        Get the list of features to use in the model based on metadata.
        
        Args:
            X: Input feature DataFrame.
            
        Returns:
            List of feature names to use.
        """
        feature_names = []
        
        for col in X.columns:
            if col in self.feature_usage and self.feature_usage[col] == 1:
                feature_names.append(col)
        
        return feature_names
    
    def fit(self, X: pd.DataFrame, create_missing_indicators: bool = True) -> 'DataProcessor':
        """
        Fit the data processor on the training data.
        
        Args:
            X: Input feature DataFrame.
            create_missing_indicators: Whether to create dummy variables for missing values.
            
        Returns:
            Fitted DataProcessor instance.
        """
        self.feature_names = self._get_feature_names(X)
        self.numeric_columns, self.categorical_columns, self.bool_columns = self._get_columns_by_type(self.feature_names)
        
        # Create transformers for each column type
        transformers = []
        
        # Numeric features pipeline
        if self.numeric_columns:
            numeric_transformer_steps = []
            
            # Add imputers for numeric features
            imputers = {}
            for col in self.numeric_columns:
                imputation_method = self.imputation_methods.get(col, 'mean')
                if imputation_method == 'mean':
                    imputers[col] = SimpleImputer(strategy='mean')
                elif imputation_method == 'median':
                    imputers[col] = SimpleImputer(strategy='median')
                elif imputation_method == '0':
                    imputers[col] = SimpleImputer(strategy='constant', fill_value=0)
                else:
                    # Default to mean
                    imputers[col] = SimpleImputer(strategy='mean')
            
            if imputers:
                for col, imputer in imputers.items():
                    # For each numeric column with potential missing values
                    if create_missing_indicators:
                        # Create a missing indicator transformer
                        missing_indicator = Pipeline([
                            ('missing_indicator', SimpleImputer(strategy='constant', fill_value=1)),
                        ])
                        
                        # Create a new column name for the missing indicator
                        missing_col_name = f"{col}_missing"
                        self.missing_indicator_columns.append(missing_col_name)
                        
                        # Add a transformer for the missing indicator
                        transformers.append((
                            f"missing_{col}",
                            missing_indicator,
                            [col]
                        ))
                    
                    # Add the regular imputer for the column
                    numeric_transformer = Pipeline([
                        ('imputer', imputer),
                        ('scaler', StandardScaler())
                    ])
                    
                    transformers.append((
                        f"num_{col}",
                        numeric_transformer,
                        [col]
                    ))
        
        # Categorical features pipeline
        if self.categorical_columns:
            for col in self.categorical_columns:
                transform_method = self.category_transformations.get(col, 'one-hot')
                
                if transform_method == 'one-hot':
                    # For one-hot encoding
                    imputation_method = self.imputation_methods.get(col, 'unknown')
                    
                    if imputation_method == 'unknown':
                        fill_value = 'unknown'
                    elif imputation_method == 'most_frequent':
                        fill_value = 'most_frequent'
                    else:
                        fill_value = 'unknown'
                    
                    categorical_transformer = Pipeline([
                        ('imputer', SimpleImputer(strategy='constant' if fill_value != 'most_frequent' else 'most_frequent', 
                                                 fill_value=fill_value if fill_value != 'most_frequent' else None)),
                        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
                    ])
                    
                    transformers.append((
                        f"cat_{col}",
                        categorical_transformer,
                        [col]
                    ))
                    
                    if create_missing_indicators:
                        # Create a missing indicator for categorical features
                        missing_indicator = Pipeline([
                            ('missing_indicator', SimpleImputer(strategy='constant', fill_value=1)),
                        ])
                        
                        missing_col_name = f"{col}_missing"
                        self.missing_indicator_columns.append(missing_col_name)
                        
                        transformers.append((
                            f"missing_{col}",
                            missing_indicator,
                            [col]
                        ))
        
        # Boolean features pipeline
        if self.bool_columns:
            for col in self.bool_columns:
                # For boolean features, we typically impute with False (0)
                bool_transformer = Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ])
                
                transformers.append((
                    f"bool_{col}",
                    bool_transformer,
                    [col]
                ))
                
                if create_missing_indicators:
                    # Create a missing indicator for boolean features
                    missing_indicator = Pipeline([
                        ('missing_indicator', SimpleImputer(strategy='constant', fill_value=1)),
                    ])
                    
                    missing_col_name = f"{col}_missing"
                    self.missing_indicator_columns.append(missing_col_name)
                    
                    transformers.append((
                        f"missing_{col}",
                        missing_indicator,
                        [col]
                    ))
        
        # Create the column transformer
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        # Fit the preprocessor
        self.preprocessor.fit(X)
        
        # Generate feature names after transformation
        self.feature_names_after_transform = self._get_feature_names_after_transform(X)
        
        return self
    
    def _get_feature_names_after_transform(self, X: pd.DataFrame) -> List[str]:
        """
        Generate the feature names after transformation.
        
        Args:
            X: Input feature DataFrame.
            
        Returns:
            List of feature names after transformation.
        """
        feature_names = []
        
        # Add transformed feature names
        for name, transformer, columns in self.preprocessor.transformers_:
            if name.startswith('num_'):
                # Numeric features keep their name after scaling
                feature_names.extend(columns)
            elif name.startswith('cat_'):
                # Get the one-hot encoded column names
                encoder = transformer.named_steps['encoder']
                categories = encoder.categories_[0]
                col = columns[0]
                
                for category in categories:
                    feature_names.append(f"{col}_{category}")
            elif name.startswith('bool_'):
                # Boolean features keep their name
                feature_names.extend(columns)
            elif name.startswith('missing_'):
                # Missing indicators
                col = columns[0]
                feature_names.append(f"{col}_missing")
        
        return feature_names
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data according to the fitted preprocessor.
        
        Args:
            X: Input feature DataFrame.
            
        Returns:
            Transformed DataFrame.
        """
        if self.preprocessor is None:
            raise ValueError("DataProcessor has not been fitted. Call fit() before transform().")
        
        # Apply the transformation
        X_transformed = self.preprocessor.transform(X)
        
        # Convert to DataFrame with proper column names
        X_transformed_df = pd.DataFrame(
            X_transformed, 
            columns=self.feature_names_after_transform,
            index=X.index
        )
        
        return X_transformed_df
    
    def fit_transform(self, X: pd.DataFrame, create_missing_indicators: bool = True) -> pd.DataFrame:
        """
        Fit the processor and transform the input data.
        
        Args:
            X: Input feature DataFrame.
            create_missing_indicators: Whether to create dummy variables for missing values.
            
        Returns:
            Transformed DataFrame.
        """
        self.fit(X, create_missing_indicators)
        return self.transform(X)
    
    def save(self, file_path: str) -> None:
        """
        Save the data processor to a file.
        
        Args:
            file_path: Path to save the processor.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
    
    @classmethod
    def load(cls, file_path: str) -> 'DataProcessor':
        """
        Load a saved data processor from a file.
        
        Args:
            file_path: Path to the saved processor.
            
        Returns:
            Loaded DataProcessor instance.
        """
        with open(file_path, 'rb') as file:
            processor = pickle.load(file)
        
        return processor 