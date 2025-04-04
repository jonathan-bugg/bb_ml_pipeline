"""LGBM model module for the ML Pipeline."""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, accuracy_score, f1_score, precision_score, recall_score
import optuna
import shap
import treeinterpreter as ti


class LGBMModel:
    """
    LGBM model class for the ML Pipeline.
    
    This class handles training and prediction for LightGBM models.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the LGBM model with configuration.
        
        Args:
            model_config: Dictionary containing model configuration.
        """
        self.model_config = model_config
        self.problem_type = model_config.get('problem_type', 'classification')
        self.evaluation_metric = model_config.get('evaluation_metric', 'rocauc')
        self.target_variable = model_config.get('target_variable', 'target')
        self.better_performance = model_config.get('better_performance', 'gt')  # 'gt' for greater is better, 'lt' for less is better
        self.num_hp_searches = model_config.get('num_hp_searches', 10)
        self.retrain_with_whole_dataset = model_config.get('retrain_with_whole_dataset', True)
        self.importance_extraction_method = model_config.get('importance_extraction_method', 'shap')
        self.sample_for_contribution = model_config.get('sample_for_contribution', 1.0)
        
        # Will be set during training
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.cv_results = None
        self.feature_names = None
    
    def _get_objective_function(self, X_train: pd.DataFrame, y_train: pd.Series, 
                               X_val: pd.DataFrame, y_val: pd.Series) -> callable:
        """
        Create an objective function for Optuna hyperparameter optimization.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            
        Returns:
            Objective function for Optuna.
        """
        def objective(trial):
            # Get model hyperparameters based on the model_config
            hp_space = self.model_config.get('hyperparameter_space', {}).get('lgbm', {})
            params = {}
            
            # Parse the hyperparameter space dynamically
            for param_name, param_value_string in hp_space.items():
                # Evaluate the string as Python code with the trial object
                # This allows for dynamic hyperparameter definition
                try:
                    param_value = eval(param_value_string)
                    params[param_name] = param_value
                except Exception as e:
                    print(f"Error parsing hyperparameter {param_name}: {e}")
            
            # Set fixed parameters based on problem type
            if self.problem_type == 'classification':
                params['objective'] = 'binary'
                params['metric'] = 'auc'
            else:
                params['objective'] = 'regression'
                params['metric'] = 'rmse'
            
            # Create dataset
            dtrain = lgb.Dataset(X_train, label=y_train)
            dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            
            # Train the model
            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(10)]
            )
            
            # Get the best score based on the evaluation metric
            if self.evaluation_metric == 'rocauc':
                y_pred_proba = model.predict(X_val)
                score = roc_auc_score(y_val, y_pred_proba)
            elif self.evaluation_metric == 'mse':
                y_pred = model.predict(X_val)
                score = mean_squared_error(y_val, y_pred)
                # Lower is better for MSE
                score = -score
            elif self.evaluation_metric == 'r2':
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
            elif self.evaluation_metric == 'accuracy':
                y_pred_proba = model.predict(X_val)
                y_pred = (y_pred_proba > 0.5).astype(int)
                score = accuracy_score(y_val, y_pred)
            elif self.evaluation_metric == 'f1':
                y_pred_proba = model.predict(X_val)
                y_pred = (y_pred_proba > 0.5).astype(int)
                score = f1_score(y_val, y_pred)
            else:
                raise ValueError(f"Unsupported evaluation metric: {self.evaluation_metric}")
            
            # If 'lt' (less is better) is specified, negate the score
            if self.better_performance == 'lt':
                score = -score
            
            return score
        
        return objective
    
    def train(self, X: pd.DataFrame, y: pd.Series, X_test: Optional[pd.DataFrame] = None, 
              y_test: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the LGBM model with hyperparameter optimization.
        
        Args:
            X: Training features.
            y: Training target.
            X_test: Optional test features for final evaluation.
            y_test: Optional test target for final evaluation.
            
        Returns:
            Dictionary with training results.
        """
        self.feature_names = X.columns.tolist()
        
        # Set up cross-validation strategy
        if self.problem_type == 'classification':
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Store CV results
        self.cv_results = {
            'train_scores': [],
            'val_scores': [],
            'feature_importance': [],
            'fold_models': []
        }
        
        # Perform cross-validation with hyperparameter optimization
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"Training fold {fold+1}/5...")
            
            # Split data for this fold
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create and optimize the objective function
            objective = self._get_objective_function(X_train, y_train, X_val, y_val)
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.num_hp_searches)
            
            # Get best parameters and train final model for this fold
            best_params = study.best_params
            
            # Set fixed parameters based on problem type
            if self.problem_type == 'classification':
                best_params['objective'] = 'binary'
                best_params['metric'] = 'auc'
            else:
                best_params['objective'] = 'regression'
                best_params['metric'] = 'rmse'
            
            # Train final model for this fold
            dtrain = lgb.Dataset(X_train, label=y_train)
            dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            
            model = lgb.train(
                best_params,
                dtrain,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(10)]
            )
            
            # Evaluate the model
            if self.problem_type == 'classification':
                train_preds = model.predict(X_train)
                val_preds = model.predict(X_val)
                
                train_score = roc_auc_score(y_train, train_preds)
                val_score = roc_auc_score(y_val, val_preds)
            else:
                train_preds = model.predict(X_train)
                val_preds = model.predict(X_val)
                
                train_score = mean_squared_error(y_train, train_preds, squared=False)
                val_score = mean_squared_error(y_val, val_preds, squared=False)
            
            # Store results for this fold
            self.cv_results['train_scores'].append(train_score)
            self.cv_results['val_scores'].append(val_score)
            self.cv_results['feature_importance'].append(model.feature_importance())
            self.cv_results['fold_models'].append(model)
            
            print(f"Fold {fold+1} - Train score: {train_score:.4f}, Validation score: {val_score:.4f}")
            
            # Save the best parameters from the first fold
            if fold == 0:
                self.best_params = best_params
        
        # Train the final model on the entire dataset if specified
        if self.retrain_with_whole_dataset:
            print("Training final model on entire dataset...")
            
            dtrain = lgb.Dataset(X, label=y)
            self.model = lgb.train(
                self.best_params,
                dtrain,
                valid_sets=[dtrain],
                callbacks=[lgb.early_stopping(10)]
            )
        else:
            # Use the best model from cross-validation
            best_fold_idx = np.argmax(self.cv_results['val_scores'])
            self.model = self.cv_results['fold_models'][best_fold_idx]
        
        # Calculate and store feature importance
        self.feature_importance = {
            'native': self._get_native_feature_importance()
        }
        
        # Calculate SHAP or TreeInterpreter importances if requested
        if X_test is not None:
            if self.importance_extraction_method == 'shap':
                self.feature_importance['shap'] = self._get_shap_importance(X_test)
            elif self.importance_extraction_method == 'treeinterpreter':
                self.feature_importance['treeinterpreter'] = self._get_treeinterpreter_importance(X_test)
        
        # Calculate final performance metrics
        training_results = {
            'cv_mean_train_score': np.mean(self.cv_results['train_scores']),
            'cv_mean_val_score': np.mean(self.cv_results['val_scores']),
            'cv_std_val_score': np.std(self.cv_results['val_scores']),
            'best_params': self.best_params
        }
        
        # Add test score if test data is provided
        if X_test is not None and y_test is not None:
            test_preds = self.predict(X_test)
            
            if self.problem_type == 'classification':
                test_score = roc_auc_score(y_test, test_preds)
                training_results['test_roc_auc'] = test_score
                
                # Calculate additional classification metrics
                binary_preds = (test_preds > 0.5).astype(int)
                training_results['test_accuracy'] = accuracy_score(y_test, binary_preds)
                training_results['test_f1'] = f1_score(y_test, binary_preds)
                training_results['test_precision'] = precision_score(y_test, binary_preds)
                training_results['test_recall'] = recall_score(y_test, binary_preds)
            else:
                test_rmse = mean_squared_error(y_test, test_preds, squared=False)
                test_r2 = r2_score(y_test, test_preds)
                
                training_results['test_rmse'] = test_rmse
                training_results['test_r2'] = test_r2
        
        return training_results
    
    def _get_native_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the LightGBM model.
        
        Returns:
            Dictionary mapping feature names to importance values.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        importance = self.model.feature_importance()
        return dict(zip(self.feature_names, importance))
    
    def _get_shap_importance(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate SHAP values for feature importance.
        
        Args:
            X: DataFrame to calculate SHAP values on.
            
        Returns:
            Dictionary with SHAP values and importance.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Sample data if needed
        if self.sample_for_contribution < 1.0:
            sample_size = max(int(len(X) * self.sample_for_contribution), 1)
            X_sample = X.sample(sample_size, random_state=42)
        else:
            X_sample = X
        
        # Create an explainer and calculate SHAP values
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        # For classification, shap_values is a list with one element per class
        if self.problem_type == 'classification':
            shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values
        
        # Calculate overall feature importance (mean absolute SHAP value for each feature)
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Determine signs by correlating features with their contribution
        signs = []
        for i, col in enumerate(self.feature_names):
            if X_sample[col].nunique() > 1:  # Only calculate correlation if there is variation
                correlation = np.corrcoef(X_sample[col].values, shap_values[:, i])[0, 1]
                signs.append(np.sign(correlation))
            else:
                signs.append(1.0)  # Default to positive if no variation
        
        # Apply signs to feature importance
        signed_importance = feature_importance * signs
        
        # Create dictionary mapping feature names to importance
        importance_dict = dict(zip(self.feature_names, signed_importance))
        
        # Sort features by absolute importance
        sorted_importance = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'values': shap_values,
            'importance': importance_dict,
            'sorted_importance': sorted_importance,
            'sample_indices': X_sample.index.tolist()
        }
    
    def _get_treeinterpreter_importance(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate TreeInterpreter values for feature importance.
        
        Args:
            X: DataFrame to calculate TreeInterpreter values on.
            
        Returns:
            Dictionary with TreeInterpreter values and importance.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        try:
            import treeinterpreter as ti
        except ImportError:
            print("TreeInterpreter is not installed. Please install it using pip install treeinterpreter")
            return {}
        
        # Sample data if needed
        if self.sample_for_contribution < 1.0:
            sample_size = max(int(len(X) * self.sample_for_contribution), 1)
            X_sample = X.sample(sample_size, random_state=42)
        else:
            X_sample = X
        
        # Calculate TreeInterpreter values
        pred, bias, contributions = ti.predict(self.model, X_sample.values)
        
        # For classification, contributions has shape (n_samples, n_classes, n_features)
        # We take the positive class (index 1) for binary classification
        if self.problem_type == 'classification':
            if len(contributions.shape) == 3:
                contributions = contributions[:, 1, :]
        
        # Calculate overall feature importance (mean absolute contribution for each feature)
        feature_importance = np.abs(contributions).mean(axis=0)
        
        # Determine signs by correlating features with their contribution
        signs = []
        for i, col in enumerate(self.feature_names):
            if X_sample[col].nunique() > 1:  # Only calculate correlation if there is variation
                correlation = np.corrcoef(X_sample[col].values, contributions[:, i])[0, 1]
                signs.append(np.sign(correlation))
            else:
                signs.append(1.0)  # Default to positive if no variation
        
        # Apply signs to feature importance
        signed_importance = feature_importance * signs
        
        # Create dictionary mapping feature names to importance
        importance_dict = dict(zip(self.feature_names, signed_importance))
        
        # Sort features by absolute importance
        sorted_importance = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'contributions': contributions,
            'bias': bias,
            'importance': importance_dict,
            'sorted_importance': sorted_importance,
            'sample_indices': X_sample.index.tolist()
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on.
            
        Returns:
            Array of predictions. For classification, these are probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        return self.model.predict(X)
    
    def save(self, file_path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            file_path: Path to save the model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
    
    @classmethod
    def load(cls, file_path: str) -> 'LGBMModel':
        """
        Load a saved model from a file.
        
        Args:
            file_path: Path to the saved model.
            
        Returns:
            Loaded LGBMModel instance.
        """
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        
        return model 