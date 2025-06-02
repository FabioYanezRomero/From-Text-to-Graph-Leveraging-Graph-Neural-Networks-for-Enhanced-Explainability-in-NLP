#!/usr/bin/env python3
"""
Graph Generation Script

This script is the main entry point for generating graphs from text datasets.
It can be run directly to generate syntactic, semantic, and constituency graphs
for specified datasets.
"""

import sys
import os

# Add the parent directory to the path to allow importing from the Graph_Generation package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Graph_Generation.graph_generator import process_datasets
from Graph_Generation.config import (
    GRAPH_TYPES, DEFAULT_MODELS, AVAILABLE_MODELS,
    DEFAULT_DATASETS, DEFAULT_SUBSETS, DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE
)

import argparse
import json

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate graphs from text datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        default=DEFAULT_DATASETS,
        help="Datasets to process (e.g., stanfordnlp/sst2 SetFit/ag_news)"
    )
    
    parser.add_argument(
        "--subsets", 
        nargs="+", 
        default=DEFAULT_SUBSETS,
        help="Subsets to process (e.g., train test validation)"
    )
    
    # Graph type arguments
    parser.add_argument(
        "--graph_types", 
        nargs="+", 
        choices=GRAPH_TYPES,
        default=GRAPH_TYPES,
        help="Types of graphs to generate"
    )
    
    # Model arguments
    parser.add_argument(
        "--syntactic_model", 
        choices=AVAILABLE_MODELS["syntactic"],
        default=DEFAULT_MODELS["syntactic"],
        help="Model for syntactic graph generation"
    )
    
    parser.add_argument(
        "--semantic_model", 
        choices=AVAILABLE_MODELS["semantic"],
        default=DEFAULT_MODELS["semantic"],
        help="Model for semantic graph generation"
    )
    
    parser.add_argument(
        "--constituency_model", 
        choices=AVAILABLE_MODELS["constituency"],
        default=DEFAULT_MODELS["constituency"],
        help="Model for constituency graph generation"
    )
    
    # Processing arguments
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for processing"
    )
    
    parser.add_argument(
        "--device", 
        default=DEFAULT_DEVICE,
        help="Device to run on (e.g., cuda:0, cpu)"
    )
    
    # Config file
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to JSON configuration file (overrides command-line arguments)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()
    
    # Load config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Override arguments with config file values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Create models dictionary
    models = {
        "syntactic": args.syntactic_model,
        "semantic": args.semantic_model,
        "constituency": args.constituency_model
    }
    
    # Process datasets
    process_datasets(
        datasets=args.datasets,
        subsets=args.subsets,
        graph_types=args.graph_types,
        models=models,
        batch_size=args.batch_size,
        device=args.device
    )

if __name__ == "__main__":
    main()
