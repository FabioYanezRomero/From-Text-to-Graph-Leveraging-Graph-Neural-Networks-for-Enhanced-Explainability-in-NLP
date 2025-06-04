#!/usr/bin/env python3
"""
Model Selector

This module finds the best checkpoint from fine-tuned models based on F1-score.
"""

import os
import json
import glob
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def find_best_checkpoint(results_dir, dataset_name, metric='f1-score', split='test'):
    """
    Find the best checkpoint for a given dataset based on F1-score or other metrics
    
    Parameters:
    - results_dir: Directory containing fine-tuned model results
    - dataset_name: Name of the dataset (e.g., 'stanfordnlp/sst2', 'setfit/ag_news')
    - metric: Metric to use for selecting the best model (default: 'f1-score')
    - split: Data split to use for evaluation (default: 'test')
    
    Returns:
    - Path to the best model checkpoint
    - Best F1-score
    """
    # Extract dataset provider and name
    provider, name = dataset_name.split('/')
    
    # Find all training runs for this dataset
    dataset_dirs = glob.glob(os.path.join(results_dir, provider, f"{name}_*"))
    
    if not dataset_dirs:
        logger.warning(f"No training runs found for dataset {dataset_name}")
        return None, None
    
    best_score = -1
    best_model_path = None
    best_run_dir = None
    best_epoch = None
    
    # Iterate through all training runs
    for run_dir in dataset_dirs:
        logger.info(f"Examining training run: {os.path.basename(run_dir)}")
        
        # Find all classification reports for the specified split
        report_files = glob.glob(os.path.join(run_dir, f"classification_report_{split}_epoch*.json"))
        
        # Check if model files exist
        model_files = glob.glob(os.path.join(run_dir, "model_epoch_*.pt"))
        
        if not report_files or not model_files:
            logger.warning(f"No classification reports or model files found in {run_dir}")
            continue
        
        # Iterate through all epochs
        for report_file in report_files:
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                
                # Extract epoch number from filename
                epoch = int(os.path.basename(report_file).split('_')[-1].split('.')[0].replace('epoch', ''))
                
                # For multi-class classification, use macro avg F1-score
                # For binary classification, use weighted avg F1-score
                if 'macro avg' in report and metric in report['macro avg']:
                    score = report['macro avg'][metric]
                elif 'weighted avg' in report and metric in report['weighted avg']:
                    score = report['weighted avg'][metric]
                else:
                    logger.warning(f"Metric {metric} not found in report {report_file}")
                    continue
                
                logger.info(f"Epoch {epoch}: {metric} = {score:.4f}")
                
                # Check if this is the best score so far
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_run_dir = run_dir
                    
                    # Construct path to the corresponding model file
                    best_model_path = os.path.join(run_dir, f"model_epoch_{epoch}.pt")
                    
                    # Check if the model file exists
                    if not os.path.exists(best_model_path):
                        logger.warning(f"Model file {best_model_path} does not exist")
                        best_model_path = None
                        continue
            
            except Exception as e:
                logger.error(f"Error processing report file {report_file}: {e}")
                continue
    
    if best_model_path:
        logger.info(f"Best model found: {best_model_path}")
        logger.info(f"Best {metric}: {best_score:.4f} (Epoch {best_epoch})")
        
        # Also check for config.json to get model name
        config_path = os.path.join(best_run_dir, "config.json")
        model_name = None
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    model_name = config.get('model_name')
            except Exception as e:
                logger.error(f"Error reading config file {config_path}: {e}")
        
        return best_model_path, best_score, model_name
    else:
        logger.warning(f"No valid model found for dataset {dataset_name}")
        return None, None, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Find the best checkpoint for a given dataset')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing fine-tuned model results')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (e.g., stanfordnlp/sst2, setfit/ag_news)')
    parser.add_argument('--metric', type=str, default='f1-score', help='Metric to use for selecting the best model')
    parser.add_argument('--split', type=str, default='test', help='Data split to use for evaluation')
    
    args = parser.parse_args()
    
    best_model_path, best_score, model_name = find_best_checkpoint(
        args.results_dir,
        args.dataset_name,
        args.metric,
        args.split
    )
    
    if best_model_path:
        print(f"Best model path: {best_model_path}")
        print(f"Best {args.metric}: {best_score:.4f}")
        if model_name:
            print(f"Model name: {model_name}")
    else:
        print(f"No valid model found for dataset {args.dataset_name}")
