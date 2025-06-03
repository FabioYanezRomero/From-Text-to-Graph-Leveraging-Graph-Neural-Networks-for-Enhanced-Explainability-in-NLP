"""
Constituency Graph Generator Module

This module provides functionality to create constituency graphs from sentences.
It leverages the Stanza NLP library to generate constituency trees and converts them
into directed graphs where nodes represent constituents and words.
"""

from typing import List, Dict, Union, Any, Optional
import os
import networkx as nx
import torch
import torch.cuda
import stanza
from .base_generator import BaseGraphGenerator

# Dictionary mapping constituency labels to more descriptive phrases
PHRASE_MAPPER = {
    'S': '«SENTENCE»',
    'NP': '«NOUN PHRASE»',
    'VP': '«VERB PHRASE»',
    'PP': '«PREPOSITIONAL PHRASE»',
    'ADJP': '«ADJECTIVE PHRASE»',
    'ADVP': '«ADVERB PHRASE»',
    'SBAR': '«SUBORDINATE CLAUSE»',
    'PRT': '«PARTICLE»',
    'INTJ': '«INTERJECTION»',
    'CONJP': '«CONJUCTION PHRASE»',
    'LST': '«LIST MARKER»',
    'UCP': '«UNLIKE COORDINATED PHRASE»',
    'PRN': '«PARENTETICAL»',
    'FRAG': '«FRAGMENT»',
    'SINV': '«INVERTED SENTENCE»',
    'SBARQ': '«SUBORDINATE CLAUSE QUESTION»',
    'SQ': '«QUESTION»',
    'WHADJP': '«WH-ADJECTIVE PHRASE»',
    'WHAVP': '«WH-ADVERB PHRASE»',
    'WHNP': '«WH-NOUN PHRASE»',
    'WHPP': '«WH-PREPOSITIONAL PHRASE»',
    'RRC': '«REDUCED RELATIVE CLAUSE»',
    'NX': '«NOUN PHRASE (NO HEAD)»',
    'WHADVP': '«WH-ADVERB PHRASE»',
    'QP': '«QUANTIFIER PHRASE»',
    'NAC': '«NOT A CONSTITUENT»',
    'X': '«UNKNOWN»'
}


class ConstituencyGraphGenerator(BaseGraphGenerator):
    """
    Creates constituency graphs from sentences.

    This class processes sentences using a Stanza constituency parser and converts
    the parsed trees into directed graphs. Each node in the graph represents
    either a constituent phrase or a word from the sentence.

    Attributes:
        model (str): Name or configuration of the constituency parser model.
        property (str): Property type, always set to 'constituency'.
        nlp: The loaded Stanza pipeline.
        device (str): The device to run the parser on (CPU or CUDA).
    """

    def __init__(self, model: str = 'default_accurate', device: str = 'cuda:0'):
        """
        Initialize the constituency parser with Stanza.

        Args:
            model (str, optional): Package name for Stanza. Defaults to 'default_accurate'.
            device (str, optional): Device to run the parser on. Defaults to 'cuda:0'.

        Raises:
            RuntimeError: If the specified device is not available.
        """
        super().__init__(model, device)
        self.property = 'constituency'
        
        # Verify device availability for better error handling
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please use 'cpu' instead or check CUDA installation.")
        
        self.device = device
        if device.startswith('cuda'):
            torch.cuda.set_device(device)
        
        # Set up Stanza configuration
        stanza_device = 'gpu' if device.startswith('cuda') else 'cpu'
        
        # Download the model if it doesn't exist
        try:
            stanza.download('en')
        except Exception as e:
            print(f"Warning: Could not download Stanza model: {e}")
            print("If the model is already downloaded, you can ignore this warning.")
        
        # Initialize the Stanza pipeline with transformer-enhanced parsing
        try:
            self.nlp = stanza.Pipeline(
                lang='en',
                processors='tokenize,pos,constituency',
                package=model,  # default_accurate uses transformer backbone
                use_gpu=(stanza_device == 'gpu')
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Stanza pipeline: {e}")

    def _parse(self, sentences: List[str]):
        """
        Parse sentences using the Stanza constituency parser.

        Args:
            sentences (List[str]): List of sentences to parse.

        Returns:
            The parsed constituency trees.
        """
        # Process all sentences as a batch
        doc = self.nlp('\n\n'.join(sentences))
        
        # Extract constituency trees from each sentence
        trees = []
        for sent in doc.sentences:
            if hasattr(sent, 'constituency') and sent.constituency:
                trees.append(sent.constituency)
            else:
                raise RuntimeError(f"Constituency parsing failed for sentence: {sent.text}")
        
        return trees

    def _tree_to_list(self, tree) -> Union[str, List]:
        """
        Convert the parsed constituency tree into a nested list structure.

        Args:
            tree: A Stanza constituency tree or subtree.

        Returns:
            Union[str, List]: A string for leaf nodes or a list for non-leaf nodes.
        """
        # Handle leaf nodes (words)
        if tree.is_leaf():
            return tree.label
        
        # Handle non-leaf nodes (constituents)
        return [tree.label] + [self._tree_to_list(child) for child in tree.children]

    def _build_graph(self, graph: nx.DiGraph, node_list: List, sentence: str, parent_id: str = '', graph_id: Optional[str] = None) -> nx.DiGraph:
        """
        Add edges and nodes to the graph from the node list.

        Args:
            graph (nx.DiGraph): The graph to build.
            node_list (List): The nested list representing the constituency tree.
            sentence (str): The original sentence.
            parent_id (str, optional): ID of the parent node. Defaults to ''.
            graph_id (str, optional): ID for the graph. Defaults to None.

        Returns:
            nx.DiGraph: The constructed graph.
        """
        parent = node_list[0] + parent_id
        if parent not in graph:
            # Add parent node with appropriate label
            label = PHRASE_MAPPER.get(str(parent), parent)
            graph.add_node(str(parent), label=label)
        
        children = node_list[1:]
        for i, child in enumerate(children):
            if isinstance(child, list):
                # Process non-leaf nodes (constituents)
                child_id = parent_id + str(i)
                child_label = child[0]
                node_id = str(child_label) + str(child_id)
                
                graph.add_node(node_id, label=child_label)
                graph.nodes[node_id]['label'] = PHRASE_MAPPER.get(child_label, child_label)
                graph.add_edge(parent, node_id, label="constituency relation")
                
                self._build_graph(graph, child, sentence, child_id)
            else:
                # Process leaf nodes (words)
                try:
                    # Use the index of the word in the sentence as the node ID
                    counter = sentence.index(child)
                    graph.add_node(counter, label=child)
                    graph.nodes[counter]['label'] = PHRASE_MAPPER.get(child, child)
                    graph.add_edge(parent, counter, label="constituency relation")
                except ValueError:
                    # Word not found in the sentence
                    continue
        
        # Add graph metadata
        graph.graph['model'] = self.model
        graph.graph['property'] = self.property
        if graph_id:
            graph.graph['id'] = graph_id
        
        return graph
    
    def _remove_nodes_and_reconnect(self, graph: nx.DiGraph) -> None:
        """
        Remove nodes that start with '_' and reconnect their parents to their children.

        Args:
            graph (nx.DiGraph): The graph to modify.
        """
        nodes_to_remove = [node for node in graph.nodes() if str(node).startswith('_')]
        for node in nodes_to_remove:
            # Get the parents and children of the node
            parents = list(graph.predecessors(node))
            children = list(graph.successors(node))
            
            # Connect each parent node to each child node
            for parent in parents:
                for child in children:
                    # Add an edge from parent to child
                    graph.add_edge(parent, child, label="constituency relation")
            
            # Remove the node
            graph.remove_node(node)

    def get_graph(self, sentences: List[str], ids: List[str] = None) -> List[nx.DiGraph]:
        """
        Generate constituency graphs for a list of sentences.

        Args:
            sentences (List[str]): List of sentences to process.
            ids (List[str], optional): List of IDs to assign to each graph. Defaults to None.

        Returns:
            List[nx.DiGraph]: List of constituency graphs.
            
        Raises:
            ValueError: If ids are provided but don't match the number of sentences.
        """
        if ids is not None and len(ids) != len(sentences):
            raise ValueError(f"Number of ids ({len(ids)}) must match number of sentences ({len(sentences)})")
        
        try:
            # Parse the sentences
            constituency_trees = self._parse(sentences)
            
            # Create a graph for each sentence
            graphs = []
            for i, tree in enumerate(constituency_trees):
                # Convert tree to nested list
                tree_list = self._tree_to_list(tree)
                
                # Create a new graph
                graph = nx.DiGraph()
                
                # Build the graph from the tree list
                graph = self._build_graph(
                    graph=graph,
                    node_list=tree_list,
                    sentence=sentences[i],
                    graph_id=ids[i] if ids else None
                )
                
                # Remove nodes that start with '_' and reconnect their parents to their children
                self._remove_nodes_and_reconnect(graph)
                
                graphs.append(graph)
            
            return graphs
            
        except Exception as e:
            raise RuntimeError(f"Error generating constituency graphs: {e}")
