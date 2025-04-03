"""Command-line entry point for the ML Pipeline."""

import argparse
import sys
from bb_ml_pipeline.ml_pipeline import ML_Pipeline


def main():
    """Run the ML Pipeline from the command line."""
    parser = argparse.ArgumentParser(description='ML Pipeline for training and prediction with LGBM models.')
    parser.add_argument('--modeling-config', '-m', type=str, required=True,
                        help='Path to the modeling configuration JSON file.')
    parser.add_argument('--data-config', '-d', type=str, required=True,
                        help='Path to the data configuration JSON file.')
    
    args = parser.parse_args()
    
    try:
        # Initialize and run the pipeline
        pipeline = ML_Pipeline(
            modeling_config=args.modeling_config,
            data_config=args.data_config
        )
        
        results = pipeline.run()
        
        # Print summary of results
        if results.get('success', False):
            print("\nML Pipeline completed successfully!")
            print(f"Results saved to: {results.get('output_dir', 'Unknown')}")
            
            if 'training_results' in results:
                training_results = results['training_results']
                print("\nTraining Results:")
                print(f"  CV Mean Validation Score: {training_results.get('cv_mean_val_score', 'N/A'):.4f}")
                print(f"  CV Standard Deviation: {training_results.get('cv_std_val_score', 'N/A'):.4f}")
                
                if 'test_roc_auc' in training_results:
                    print(f"  Test ROC AUC: {training_results.get('test_roc_auc', 'N/A'):.4f}")
                if 'test_accuracy' in training_results:
                    print(f"  Test Accuracy: {training_results.get('test_accuracy', 'N/A'):.4f}")
                if 'test_f1' in training_results:
                    print(f"  Test F1: {training_results.get('test_f1', 'N/A'):.4f}")
                if 'test_rmse' in training_results:
                    print(f"  Test RMSE: {training_results.get('test_rmse', 'N/A'):.4f}")
                if 'test_r2' in training_results:
                    print(f"  Test R²: {training_results.get('test_r2', 'N/A'):.4f}")
            
            if 'predictions' in results:
                prediction_results = results['predictions']
                print("\nPrediction Results:")
                print(f"  Number of predictions: {prediction_results.get('num_predictions', 'N/A')}")
                print(f"  Predictions saved to: {prediction_results.get('output_path', 'N/A')}")
        
        return 0
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main()) 