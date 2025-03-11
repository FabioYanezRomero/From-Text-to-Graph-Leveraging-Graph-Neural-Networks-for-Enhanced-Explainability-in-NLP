import argparse
import logging
import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import networkx as nx
from mixer import GraphMixer, GraphType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetProcessor:
    """Main class for processing datasets and generating graphs"""
    
    def __init__(self, config: Dict):
        """
        Initialize the dataset processor
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.dataset_path = Path(config['dataset_path'])
        self.graph_types = {GraphType.from_string(t) for t in config['graph_types']}
        self.batch_size = config.get('batch_size', 100)
        
        # Set log level from config if provided
        if 'log_level' in config:
            logging.getLogger().setLevel(config['log_level'])
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.graph_generators = self._initialize_graph_generators(config['model'])
        self.mixer = GraphMixer()

    def _initialize_graph_generators(self, model) -> Dict:
        """Initialize the appropriate graph generators based on config"""
        generators = {}
        
        for graph_type in self.graph_types:
            try:
                if graph_type == GraphType.SYNTACTIC:
                    from sintactic import SyntacticGraphGenerator
                    generators[graph_type] = SyntacticGraphGenerator(model)
                elif graph_type == GraphType.SEMANTIC:
                    from semantic import SemanticGraphGenerator
                    generators[graph_type] = SemanticGraphGenerator(model)
                elif graph_type == GraphType.CONSTITUENCY:
                    from constituency import ConstituencyGraphGenerator
                    generators[graph_type] = ConstituencyGraphGenerator(model)
            except ImportError as e:
                logger.error(f"Failed to initialize {graph_type.value} generator: {str(e)}")
                continue
        
        if not generators:
            raise ValueError("No graph generators could be initialized")
        
        return generators

    def process_dataset(self):
        """Process the dataset and generate graphs"""
        try:
            # Read and validate dataset
            dataset = self._read_dataset()
            if dataset is None or dataset.empty:
                logger.error("Failed to read dataset or dataset is empty")
                return

            total_entries = len(dataset)
            logger.info(f"Processing {total_entries} entries")

            # Process in batches
            for start_idx in range(0, total_entries, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_entries)
                batch = dataset.iloc[start_idx:end_idx]
                
                logger.info(f"Processing batch {start_idx//self.batch_size + 1}, entries {start_idx+1} to {end_idx}")
                self._process_batch(batch, start_idx)

        except Exception as e:
            logger.error(f"Fatal error during dataset processing: {str(e)}")
            raise

    def _process_batch(self, batch: pd.DataFrame, start_idx: int):
        """Process a batch of dataset entries"""
        for i, (idx, row) in enumerate(batch.iterrows()):
            try:
                # Extract text from the appropriate column (adjust based on your dataset structure)
                text = row['text'] if 'text' in row else row.iloc[0]
                
                current_idx = start_idx + i
                logger.info(f"Processing entry {current_idx + 1}")
                
                # Generate individual graphs
                graphs = []
                for graph_type, generator in self.graph_generators.items():
                    try:
                        graph = generator.get_graph([text])  # Updated method name
                        if graph is not None:
                            graphs.append(graph)
                    except Exception as e:
                        logger.error(f"Error generating {graph_type.value} graph for entry {current_idx}: {str(e)}")
                        continue
                
                # Mix graphs if we have any
                if graphs:
                    mixed_graph = self.mixer.mix(graphs, {t.value for t in self.graph_types})  # Convert to strings
                    self._save_graph(mixed_graph, current_idx)
                else:
                    logger.warning(f"No graphs generated for entry {current_idx}")
            
            except Exception as e:
                logger.error(f"Error processing entry {current_idx}: {str(e)}")
                continue

    def _read_dataset(self) -> Optional[pd.DataFrame]:
        """Read and validate the dataset"""
        try:
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
            
            # Read dataset based on file extension
            file_extension = self.dataset_path.suffix.lower()
            if file_extension == '.csv':
                df = pd.read_csv(self.dataset_path)
            elif file_extension == '.tsv':
                df = pd.read_csv(self.dataset_path, sep='\t')
            elif file_extension == '.json':
                df = pd.read_json(self.dataset_path)
            elif file_extension == '.jsonl':
                df = pd.read_json(self.dataset_path, lines=True)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error reading dataset: {str(e)}")
            return None

    def _save_graph(self, graph: nx.MultiDiGraph, idx: int):
        """Save the generated graph"""
        try:
            output_path = self.output_dir / f"graph_{idx}.pkl"
            nx.write_gpickle(graph, output_path)
            logger.debug(f"Saved graph to {output_path}")
        
        except Exception as e:
            logger.error(f"Error saving graph {idx}: {str(e)}")


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ['dataset_path', 'output_dir', 'graph_types']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ValueError(f"Missing required config fields: {missing_fields}")
        
        # Validate graph types
        valid_types = {t.value for t in GraphType}
        invalid_types = [t for t in config['graph_types'] if t not in valid_types]
        if invalid_types:
            raise ValueError(f"Invalid graph types in config: {invalid_types}. Valid types are: {valid_types}")
            
        return config
    
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Process dataset and generate graphs')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration JSON file')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    args = parser.parse_args()

    try:
        # Set debug logging if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Load configuration
        config = load_config(args.config)
        
        # Initialize and run processor
        processor = DatasetProcessor(config)
        processor.process_dataset()
        
        logger.info("Dataset processing completed successfully")
    
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()