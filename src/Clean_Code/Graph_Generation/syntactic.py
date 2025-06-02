"""
Syntactic Graph Generator Module
This module provides functionality to create syntactic graphs from sentences.
It leverages the supar Parser to generate dependency trees and converts them
into directed graphs where nodes represent words and edges represent syntactic relations.
"""
from typing import List, Dict, Union, Any, Optional
import networkx as nx
import torch
from supar import Parser
from .base_generator import BaseGraphGenerator


# Available Supar dependency parsing models
SUPAR_DEP_MODELS = [
    'dep-biaffine-en',
    'dep-biaffine-roberta-en',
    'dep-crf2o-en',
    'dep-crf2o-roberta-en'
]


class SyntacticGraphGenerator(BaseGraphGenerator):
    """
    Creates syntactic graphs from sentences.
    This class processes sentences using a syntactic parser and converts
    the parsed trees into directed graphs. Each node in the graph represents
    a word from the sentence, and edges represent syntactic dependencies.
    """
    def __init__(self, model: str, device: str = 'cuda:0'):
        """
        Initialize the syntactic parser with a given model.
        Args:
            model (str): Name of the syntactic parser model to load (e.g., 'dep-biaffine-roberta-en').
            device (str, optional): Device to run the parser on. Defaults to 'cuda:0'.
        Raises:
            RuntimeError: If the specified device is not available.
        """
        super().__init__(model, device)
        if model not in SUPAR_DEP_MODELS:
            raise ValueError(f"Unknown model: {model}. Available models: {SUPAR_DEP_MODELS}")
        self.property = 'syntactic'  # Corrected spelling from 'sintactic'
        
        # Verify device availability for better error handling
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please use 'cpu' instead or check CUDA installation.")
        
        if device.startswith('cuda'):
            torch.cuda.set_device(device)
        
        try:
            self.sint = Parser.load(model)
        except Exception as e:
            raise RuntimeError(f"Failed to load syntactic parser model '{model}': {e}")
    
    def _parse(self, sentences: List[str]):
        """
        Parse sentences using the syntactic parser.
        Args:
            sentences (List[str]): List of sentences to parse.
        Returns:
            The parsed syntactic trees.
        """
        return self.sint.predict(sentences, verbose=False, lang='en')
    
    def _build_graph(self, trees, ids: Optional[List[str]] = None) -> List[nx.DiGraph]:
        """
        Build graphs from the parsed syntactic trees.
        Args:
            trees: The parsed syntactic trees.
            ids (Optional[List[str]], optional): Optional list of IDs to assign to graphs. Defaults to None.
        Returns:
            List[nx.DiGraph]: List of directed graphs representing the syntactic structure.
        """
        graph_list = []
        
        for i, tree in enumerate(trees):
            graph = nx.DiGraph()
            
            # Add nodes (words) to the graph
            for j in range(len(tree.values[1])):
                graph.add_node(j+1, label=tree.values[1][j])
            
            # Add edges (dependencies) to the graph
            for j in range(len(tree.values[6])):
                parent = int(tree.values[6][j])
                if parent == 0:  # Skip root nodes (they have no parent)
                    continue
                else:
                    child = int(tree.values[0][j])
                    relation = tree.values[7][j]
                    graph.add_edge(parent, child, label=relation)
            
            # Add graph metadata
            graph.graph['model'] = self.model
            graph.graph['property'] = self.property
            if ids and i < len(ids):
                graph.graph['id'] = ids[i]
                
            graph_list.append(graph)
            
        return graph_list
    
    def get_graph(self, sentences: List[str], ids: Optional[List[str]] = None) -> List[nx.DiGraph]:
        """
        Get the syntactic graph for each sentence in the list.
        Args:
            sentences (List[str]): List of sentences to process.
            ids (Optional[List[str]], optional): List of IDs to assign to each graph. Defaults to None.
        Returns:
            List[nx.DiGraph]: List of syntactic graphs.
            
        Raises:
            ValueError: If ids are provided but don't match the number of sentences.
        """
        if ids is not None and len(ids) != len(sentences):
            raise ValueError(f"Number of ids ({len(ids)}) must match number of sentences ({len(sentences)})")
        
        try:
            syntactic_trees = self._parse(sentences)
            graphs = self._build_graph(syntactic_trees, ids)
            return graphs
        except Exception as e:
            raise RuntimeError(f"Error generating syntactic graphs: {e}")
