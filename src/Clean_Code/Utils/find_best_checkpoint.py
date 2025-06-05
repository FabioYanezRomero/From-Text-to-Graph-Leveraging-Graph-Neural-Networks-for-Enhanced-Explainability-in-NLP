#!/usr/bin/env python3
"""
Utility to find the best model checkpoint based on a specified metric.
"""

import os
import json
import argparse
import glob
from pathlib import Path

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Find the best model checkpoint based on a metric')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing training results')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (e.g., setfit/ag_news)')
    parser.add_argument('--metric', type=str, default='f1-score', help='Metric to use for selecting the best checkpoint')
    return parser.parse_args()

def find_best_checkpoint(results_dir, dataset_name, metric='f1-score'):
    """Find the best model checkpoint based on the specified metric"""
    # Parse dataset name to get provider and dataset parts
    if '/' in dataset_name:
        provider, dataset = dataset_name.split('/', 1)
    else:
        provider = ''
        dataset = dataset_name
    
    # Check for the dataset directory structure
    # First try the direct path
    dataset_path = os.path.join(results_dir, dataset_name.replace('/', '_'))
    
    # If that doesn't exist, try the provider/dataset structure
    if not os.path.exists(dataset_path) and provider:
        dataset_path = os.path.join(results_dir, provider)
        if os.path.exists(dataset_path):
            # Look for directories starting with the dataset name
            possible_dirs = [d for d in os.listdir(dataset_path) 
                           if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith(dataset)]
            if not possible_dirs:
                print(f"No results found for dataset {dataset} in {dataset_path}")
                return None, None
        else:
            print(f"No results found for provider {provider} in {results_dir}")
            return None, None
    elif not os.path.exists(dataset_path):
        print(f"No results found for dataset {dataset_name} in {results_dir}")
        return None, None
    
    # Find all training run directories
    if provider and os.path.exists(os.path.join(results_dir, provider)):
        # We're using the provider/dataset structure
        run_dirs = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith(dataset)]
        run_dirs = [os.path.join(dataset_path, d) for d in run_dirs]
    else:
        # We're using the flat structure
        run_dirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
    
    print(f"Found {len(run_dirs)} training runs for {dataset_name}")
    
    best_score = -1
    best_checkpoint = None
    
    for run_dir in run_dirs:
        print(f"Examining training run: {run_dir}")
        
        # Look for classification reports - check both naming patterns
        report_files = glob.glob(os.path.join(run_dir, "classification_report_epoch_*.json")) or \
                      glob.glob(os.path.join(run_dir, "classification_report_test_epoch*.json"))
        
        if not report_files:
            print(f"No classification reports found in {run_dir}")
            continue
        
        # Check if model files exist
        model_files = glob.glob(os.path.join(run_dir, "model_epoch_*.pt"))
        if not model_files:
            print(f"No model files found in {run_dir}")
            continue
        
        # Process each report file
        for report_file in report_files:
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                
                # Extract epoch number from filename
                filename = os.path.basename(report_file)
                if 'test_epoch' in filename:
                    # Format: classification_report_test_epoch4.json
                    epoch = int(filename.replace('classification_report_test_epoch', '').replace('.json', ''))
                else:
                    # Format: classification_report_epoch_4.json
                    epoch = int(filename.split('_')[-1].split('.')[0])
                
                # Get the metric value (use macro avg for classification metrics)
                if 'macro avg' in report and metric in report['macro avg']:
                    score = report['macro avg'][metric]
                elif metric in report:
                    score = report[metric]
                else:
                    print(f"Metric {metric} not found in report {report_file}")
                    continue
                
                print(f"Epoch {epoch}: {metric} = {score:.4f}")
                
                # Check if this is the best score so far
                if score > best_score:
                    best_score = score
                    model_file = os.path.join(run_dir, f"model_epoch_{epoch}.pt")
                    if os.path.exists(model_file):
                        best_checkpoint = model_file
                    else:
                        print(f"Model file not found: {model_file}")
            except Exception as e:
                print(f"Error processing report {report_file}: {str(e)}")
    
    return best_checkpoint, best_score

def main():
    """Main entry point"""
    args = parse_args()
    
    best_checkpoint, best_score = find_best_checkpoint(
        args.results_dir,
        args.dataset_name,
        args.metric
    )
    
    if best_checkpoint:
        print(f"Best checkpoint: {best_checkpoint}")
        print(f"Best {args.metric}: {best_score:.4f}")
    else:
        print(f"No valid checkpoint found for {args.dataset_name}")
        exit(1)

if __name__ == "__main__":
    main()
