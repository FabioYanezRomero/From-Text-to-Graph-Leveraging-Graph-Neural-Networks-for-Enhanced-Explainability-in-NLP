#!/usr/bin/env python3
import argparse
import logging
import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from Code.Dataset_processing.mixer import GraphMixer, GraphType
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_argparser() -> argparse.ArgumentParser:
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Generate and process graphs from text data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        help='Path to the input dataset (supported formats: csv, tsv, json, jsonl)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='processed_data/graphs',
        help='Directory to save the generated graphs'
    )
    parser.add_argument(
        '--graph-types',
        nargs='+',
        choices=['syntactic', 'semantic', 'constituency'],
        default=['syntactic', 'semantic'],
        help='Types of graphs to generate'
    )
    parser.add_argument(
        '--model', 
        type=str,
        default='en_core_web_sm',
        help='Spacy model to use for text processing'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of texts to process in each batch'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    return parser

def process_dataset(args: argparse.Namespace) -> None:
    """Process the dataset and generate graphs based on command line arguments"""
    try:
        # Prepare configuration
        config = {
            'dataset_path': args.dataset,
            'output_dir': args.output_dir,
            'graph_types': args.graph_types,
            'model': args.model,
            'batch_size': args.batch_size,
            'log_level': logging.DEBUG if args.debug else logging.INFO
        }
        
        # Initialize the graph processor
        from Code.Dataset_processing.main import DatasetProcessor
        processor = DatasetProcessor(config)
        
        # Process the dataset
        processor.process_dataset()
        
        logger.info("Dataset processing completed successfully")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

def main():
    """Main entry point"""
    try:
        # Parse command line arguments
        parser = setup_argparser()
        args = parser.parse_args()
        
        # Set debug logging if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Process the dataset
        process_dataset(args)
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()