"""Main ML Pipeline module."""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from bb_ml_pipeline.data.data_loader import DataLoader
from bb_ml_pipeline.data.data_processor import DataProcessor
from bb_ml_pipeline.models.lgbm_model import LGBMModel


class ML_Pipeline:
    """
    Main ML Pipeline class.
    
    This class orchestrates the entire ML pipeline process, from data loading
    to model training/prediction and result exporting.
    """
    
    def __init__(self, modeling_config: str, data_config: str):
        """
        Initialize the ML Pipeline.
        
        Args:
            modeling_config: Path to the modeling configuration JSON file.
            data_config: Path to the data configuration JSON file.
        """
        self.modeling_config_path = modeling_config
        self.data_config_path = data_config
        
        # Load configurations
        self.modeling_config = self._load_json_config(modeling_config)
        
        # Initialize components
        self.data_loader = DataLoader(data_config)
        self.data_processor = None
        self.model = None
        
        # Set up output directory
        self.timestamp = datetime.now().strftime("%m%d%Y%H%M%S")
        self.is_training = self.data_loader.is_training
        
        if self.is_training:
            self.output_dir = self._create_output_dir()
        else:
            # For prediction mode, get output dir from data config
            self.output_dir = os.path.dirname(self.data_loader.data_config.get('predictions_output_path', ''))
            if not self.output_dir:
                # If not specified, create a new output directory
                self.output_dir = self._create_output_dir()
    
    def _load_json_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load JSON configuration file.
        
        Args:
            config_path: Path to the JSON configuration file.
            
        Returns:
            Dict containing the configuration.
        """
        with open(config_path, 'r') as file:
            return json.load(file)
    
    def _create_output_dir(self) -> str:
        """
        Create output directory for results.
        
        Returns:
            Path to the created directory.
        """
        # Extract base directory from data config if available
        base_dir = os.path.dirname(self.data_loader.data_config.get('file_path', ''))
        
        if not base_dir:
            base_dir = os.getcwd()
        
        # Create timestamped directory
        output_dir = os.path.join(base_dir, self.timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'deep_insights_input'), exist_ok=True)
        
        return output_dir
    
    def run(self) -> Dict[str, Any]:
        """
        Run the ML Pipeline.
        
        This method orchestrates the entire process:
        1. Load and validate data
        2. Process data
        3. Train model or make predictions
        4. Export results
        
        Returns:
            Dictionary with the results of the pipeline run.
        """
        print("Starting ML Pipeline...")
        
        # Load and validate data
        print("Loading data...")
        self.data_loader.load_data()
        self.data_loader.load_metadata()
        
        validation_results = self.data_loader.validate_data(self.modeling_config)
        if not validation_results['passed']:
            print("Data validation failed:")
            for error in validation_results['errors']:
                print(f"  - {error}")
            raise ValueError("Data validation failed. Please fix the errors and try again.")
        
        # Process data differently based on training or prediction mode
        if self.is_training:
            # Training mode
            return self._run_training()
        else:
            # Prediction mode
            return self._run_prediction()
    
    def _run_training(self) -> Dict[str, Any]:
        """
        Run the ML Pipeline in training mode.
        
        Returns:
            Dictionary with training results.
        """
        print("Running in training mode...")
        
        # Get target and features
        y, X = self.data_loader.get_target_and_features(self.modeling_config)
        
        # Process data
        print("Processing data...")
        self.data_processor = DataProcessor(self.data_loader.metadata)
        X_processed = self.data_processor.fit_transform(X)
        
        # Train model
        print("Training model...")
        self.model = LGBMModel(self.modeling_config)
        training_results = self.model.train(X_processed, y)
        
        # Extract feature importance
        print("Extracting feature importance...")
        feature_importance = self.model.feature_importance
        
        # Save results
        print("Saving results...")
        self._save_training_results(X, y, X_processed, training_results, feature_importance)
        
        return {
            'success': True,
            'output_dir': self.output_dir,
            'training_results': training_results
        }
    
    def _run_prediction(self) -> Dict[str, Any]:
        """
        Run the ML Pipeline in prediction mode.
        
        Returns:
            Dictionary with prediction results.
        """
        print("Running in prediction mode...")
        
        # Load model and data processor
        model_path = self.data_loader.data_config.get('model_file_path')
        processor_path = self.data_loader.data_config.get('data_processor_file_path')
        
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
        
        if not processor_path or not os.path.exists(processor_path):
            raise ValueError(f"Data processor file not found: {processor_path}")
        
        print(f"Loading model from {model_path}...")
        self.model = LGBMModel.load(model_path)
        
        print(f"Loading data processor from {processor_path}...")
        self.data_processor = DataProcessor.load(processor_path)
        
        # Get features (and target if available)
        y, X = self.data_loader.get_target_and_features(self.modeling_config)
        
        # Process data
        print("Processing data...")
        X_processed = self.data_processor.transform(X)
        
        # Make predictions
        print("Making predictions...")
        predictions = self.model.predict(X_processed)
        
        # Calculate feature importance if requested
        calculate_importance = self.data_loader.data_config.get('calculate_importance', False)
        feature_importance = None
        
        if calculate_importance:
            print("Calculating feature importance...")
            if self.model.importance_extraction_method == 'shap':
                feature_importance = self.model.feature_importance.get('shap', None)
                if not feature_importance:
                    feature_importance = self.model._get_shap_importance(X_processed)
            elif self.model.importance_extraction_method == 'treeinterpreter':
                feature_importance = self.model.feature_importance.get('treeinterpreter', None)
                if not feature_importance:
                    feature_importance = self.model._get_treeinterpreter_importance(X_processed)
        
        # Save results
        print("Saving prediction results...")
        prediction_results = self._save_prediction_results(X, y, X_processed, predictions, feature_importance)
        
        return {
            'success': True,
            'output_dir': self.output_dir,
            'predictions': prediction_results
        }
    
    def _save_training_results(self, X_orig: pd.DataFrame, y: pd.Series, 
                              X_processed: pd.DataFrame, training_results: Dict[str, Any],
                              feature_importance: Dict[str, Any]) -> None:
        """
        Save training results to the output directory.
        
        Args:
            X_orig: Original features DataFrame.
            y: Target Series.
            X_processed: Processed features DataFrame.
            training_results: Dictionary with training metrics and results.
            feature_importance: Dictionary with feature importance information.
        """
        # Save model
        model_file = os.path.join(self.output_dir, f"{self.modeling_config['target_variable']}_{self.modeling_config['problem_type']}_{self.modeling_config['models'][0]}.pkl")
        self.model.save(model_file)
        
        # Save data processor
        processor_file = os.path.join(self.output_dir, 'deep_insights_input', 
                                     f"{self.modeling_config['target_variable']}_{self.modeling_config['problem_type']}_{self.modeling_config['models'][0]}_data_processor.pkl")
        self.data_processor.save(processor_file)
        
        # Save training configuration
        config_file = os.path.join(self.output_dir, 'training_config.json')
        training_config = {
            'modeling_config': self.modeling_config,
            'data_config': self.data_loader.data_config,
            'training_results': training_results
        }
        with open(config_file, 'w') as file:
            json.dump(training_config, file, indent=2)
        
        # Create predictions with original data
        preds = self.model.predict(X_processed)
        
        # For classification, also create binary predictions
        if self.modeling_config['problem_type'] == 'classification':
            binary_preds = (preds > 0.5).astype(int)
            
            # Save original data + predictions
            results_df = pd.concat([
                X_orig.reset_index(drop=True),
                y.reset_index(drop=True),
                pd.Series(preds, name='prediction_proba'),
                pd.Series(binary_preds, name='prediction_class')
            ], axis=1)
        else:
            # Save original data + predictions for regression
            results_df = pd.concat([
                X_orig.reset_index(drop=True),
                y.reset_index(drop=True),
                pd.Series(preds, name='prediction')
            ], axis=1)
        
        # Save original data + predictions
        orig_preds_file = os.path.join(self.output_dir, 'predictions', 'original_data_with_predictions.csv')
        results_df.to_csv(orig_preds_file, index=False)
        
        # Save transformed data + predictions
        if self.modeling_config['problem_type'] == 'classification':
            transformed_df = pd.concat([
                X_processed.reset_index(drop=True),
                y.reset_index(drop=True),
                pd.Series(preds, name='prediction_proba'),
                pd.Series(binary_preds, name='prediction_class')
            ], axis=1)
        else:
            transformed_df = pd.concat([
                X_processed.reset_index(drop=True),
                y.reset_index(drop=True),
                pd.Series(preds, name='prediction')
            ], axis=1)
        
        transformed_preds_file = os.path.join(self.output_dir, 'predictions', 'transformed_data_with_predictions.csv')
        transformed_df.to_csv(transformed_preds_file, index=False)
        
        # Save feature importance
        if feature_importance:
            # Native feature importance
            native_importance = feature_importance.get('native', {})
            native_importance_df = pd.DataFrame({
                'feature': list(native_importance.keys()),
                'importance': list(native_importance.values())
            })
            native_importance_df = native_importance_df.sort_values('importance', ascending=False)
            
            native_importance_file = os.path.join(self.output_dir, 'deep_insights_input', 'native_feature_importance.csv')
            native_importance_df.to_csv(native_importance_file, index=False)
            
            # SHAP or TreeInterpreter importance
            if 'shap' in feature_importance:
                # Save SHAP overall importance
                shap_importance = feature_importance['shap']['importance']
                shap_importance_df = pd.DataFrame({
                    'feature': list(shap_importance.keys()),
                    'importance': list(shap_importance.values())
                })
                shap_importance_df = shap_importance_df.sort_values('importance', ascending=False)
                
                shap_importance_file = os.path.join(self.output_dir, 'deep_insights_input', 'shap_feature_importance.csv')
                shap_importance_df.to_csv(shap_importance_file, index=False)
                
                # Save SHAP values for each sample
                if 'values' in feature_importance['shap'] and 'sample_indices' in feature_importance['shap']:
                    shap_values = feature_importance['shap']['values']
                    sample_indices = feature_importance['shap']['sample_indices']
                    
                    # Create DataFrame with SHAP values
                    shap_values_df = pd.DataFrame(
                        shap_values,
                        columns=self.model.feature_names,
                        index=sample_indices
                    )
                    
                    # Save SHAP values
                    shap_values_file = os.path.join(self.output_dir, 'deep_insights_input', 'shap_values.csv')
                    shap_values_df.to_csv(shap_values_file)
            
            elif 'treeinterpreter' in feature_importance:
                # Save TreeInterpreter overall importance
                ti_importance = feature_importance['treeinterpreter']['importance']
                ti_importance_df = pd.DataFrame({
                    'feature': list(ti_importance.keys()),
                    'importance': list(ti_importance.values())
                })
                ti_importance_df = ti_importance_df.sort_values('importance', ascending=False)
                
                ti_importance_file = os.path.join(self.output_dir, 'deep_insights_input', 'treeinterpreter_feature_importance.csv')
                ti_importance_df.to_csv(ti_importance_file, index=False)
                
                # Save contribution values for each sample
                if 'contributions' in feature_importance['treeinterpreter'] and 'sample_indices' in feature_importance['treeinterpreter']:
                    contributions = feature_importance['treeinterpreter']['contributions']
                    sample_indices = feature_importance['treeinterpreter']['sample_indices']
                    
                    # Create DataFrame with contribution values
                    contributions_df = pd.DataFrame(
                        contributions,
                        columns=self.model.feature_names,
                        index=sample_indices
                    )
                    
                    # Save contribution values
                    contributions_file = os.path.join(self.output_dir, 'deep_insights_input', 'treeinterpreter_contributions.csv')
                    contributions_df.to_csv(contributions_file)
    
    def _save_prediction_results(self, X_orig: pd.DataFrame, y: Optional[pd.Series], 
                                X_processed: pd.DataFrame, predictions: np.ndarray,
                                feature_importance: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Save prediction results to the output directory.
        
        Args:
            X_orig: Original features DataFrame.
            y: Target Series (if available).
            X_processed: Processed features DataFrame.
            predictions: Array of predictions.
            feature_importance: Dictionary with feature importance information.
            
        Returns:
            Dictionary with prediction results.
        """
        # Get the unique ID column name from data config
        uid_column = self.data_loader.data_config.get('uid_column_name', None)
        
        # Prepare results
        if uid_column and uid_column in X_orig.columns:
            results_df = pd.DataFrame({
                uid_column: X_orig[uid_column]
            })
        else:
            results_df = pd.DataFrame({
                'index': range(len(predictions))
            })
        
        # Add predictions to results
        if self.model.problem_type == 'classification':
            results_df['prediction_proba'] = predictions
            results_df['prediction_class'] = (predictions > 0.5).astype(int)
        else:
            results_df['prediction'] = predictions
        
        # Add target if available
        if y is not None:
            target_name = self.model.target_variable
            results_df[target_name] = y.reset_index(drop=True)
        
        # Save predictions to the specified output path
        output_path = self.data_loader.data_config.get('predictions_output_path')
        if not output_path:
            # Default output path
            output_path = os.path.join(self.output_dir, 'predictions', 'predictions.csv')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save predictions
        results_df.to_csv(output_path, index=False)
        
        # Save original data + predictions
        orig_preds_file = os.path.join(self.output_dir, 'predictions', 'original_data_with_predictions.csv')
        
        if self.model.problem_type == 'classification':
            full_results_df = pd.concat([
                X_orig.reset_index(drop=True),
                pd.Series(predictions, name='prediction_proba'),
                pd.Series((predictions > 0.5).astype(int), name='prediction_class')
            ], axis=1)
        else:
            full_results_df = pd.concat([
                X_orig.reset_index(drop=True),
                pd.Series(predictions, name='prediction')
            ], axis=1)
        
        if y is not None:
            full_results_df[self.model.target_variable] = y.reset_index(drop=True)
        
        full_results_df.to_csv(orig_preds_file, index=False)
        
        # Save transformed data + predictions
        transformed_preds_file = os.path.join(self.output_dir, 'predictions', 'transformed_data_with_predictions.csv')
        
        if self.model.problem_type == 'classification':
            transformed_df = pd.concat([
                X_processed.reset_index(drop=True),
                pd.Series(predictions, name='prediction_proba'),
                pd.Series((predictions > 0.5).astype(int), name='prediction_class')
            ], axis=1)
        else:
            transformed_df = pd.concat([
                X_processed.reset_index(drop=True),
                pd.Series(predictions, name='prediction')
            ], axis=1)
        
        if y is not None:
            transformed_df[self.model.target_variable] = y.reset_index(drop=True)
        
        transformed_df.to_csv(transformed_preds_file, index=False)
        
        # Save feature importance if available
        if feature_importance:
            # Create directory for deep insights if it doesn't exist
            deep_insights_dir = os.path.join(self.output_dir, 'deep_insights_input')
            os.makedirs(deep_insights_dir, exist_ok=True)
            
            # Save importance information
            if 'importance' in feature_importance:
                importance = feature_importance['importance']
                importance_df = pd.DataFrame({
                    'feature': list(importance.keys()),
                    'importance': list(importance.values())
                })
                importance_df = importance_df.sort_values('importance', ascending=False)
                
                if self.model.importance_extraction_method == 'shap':
                    importance_file = os.path.join(deep_insights_dir, 'shap_feature_importance.csv')
                else:
                    importance_file = os.path.join(deep_insights_dir, 'treeinterpreter_feature_importance.csv')
                
                importance_df.to_csv(importance_file, index=False)
            
            # Save detailed contributions
            if self.model.importance_extraction_method == 'shap' and 'values' in feature_importance:
                shap_values = feature_importance['values']
                sample_indices = feature_importance.get('sample_indices', range(len(shap_values)))
                
                shap_values_df = pd.DataFrame(
                    shap_values,
                    columns=self.model.feature_names,
                    index=sample_indices
                )
                
                shap_values_file = os.path.join(deep_insights_dir, 'shap_values.csv')
                shap_values_df.to_csv(shap_values_file)
            
            elif self.model.importance_extraction_method == 'treeinterpreter' and 'contributions' in feature_importance:
                contributions = feature_importance['contributions']
                sample_indices = feature_importance.get('sample_indices', range(len(contributions)))
                
                contributions_df = pd.DataFrame(
                    contributions,
                    columns=self.model.feature_names,
                    index=sample_indices
                )
                
                contributions_file = os.path.join(deep_insights_dir, 'treeinterpreter_contributions.csv')
                contributions_df.to_csv(contributions_file)
        
        return {
            'output_path': output_path,
            'num_predictions': len(predictions)
        } 