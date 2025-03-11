"""
Constituency Graph Generator Module

This module provides functionality to create constituency graphs from sentences.
It leverages the supar Parser to generate constituency trees and converts them
into directed graphs where nodes represent constituents and words.
"""

from typing import List, Dict, Union, Any, Optional
import networkx as nx
import matplotlib.pyplot as plt
import pickle as pkl
import torch
from supar import Parser

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


class ConstituencyGraphGenerator:
    """
    Creates constituency graphs from sentences.

    This class processes sentences using a constituency parser and converts
    the parsed trees into directed graphs. Each node in the graph represents
    either a constituent phrase or a word from the sentence.

    Attributes:
        model (str): Name of the constituency parser model.
        property (str): Property type, always set to 'constituency'.
        cons: The loaded constituency parser.
        device (str): The device to run the parser on (CPU or CUDA).
    """

    def __init__(self, model: str, device: str = 'cuda:0'):
        """
        Initialize the constituency parser with a given model.

        Args:
            model (str): Name of the constituency parser model to load.
            device (str, optional): Device to run the parser on. Defaults to 'cuda:0'.

        Raises:
            RuntimeError: If the specified device is not available.
        """
        self.model = model
        self.property = 'constituency'
        
        # Verify device availability for better error handling
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please use 'cpu' instead or check CUDA installation.")
        
        self.device = device
        if device.startswith('cuda'):
            torch.cuda.set_device(device)
        
        try:
            self.cons = Parser.load(model)
        except Exception as e:
            raise RuntimeError(f"Failed to load constituency parser model '{model}': {e}")

    def _parse(self, sentences: List[str]):
        """
        Parse sentences using the constituency parser.

        Args:
            sentences (List[str]): List of sentences to parse.

        Returns:
            The parsed constituency trees.
        """
        return self.cons.predict(sentences, verbose=False, lang='en')

    def _tree_to_list(self, tree) -> Union[str, List]:
        """
        Convert the parsed constituency tree into a nested list structure.

        Args:
            tree: A constituency tree or subtree.

        Returns:
            Union[str, List]: A string for leaf nodes or a list for non-leaf nodes.
        """
        if isinstance(tree, str):  # base case: leaf node
            return tree
        return [tree.label()] + [self._tree_to_list(child) for child in tree]

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
        
        constituency_trees = self._parse(sentences)
        graph_list = []
        
        for i, sentence in enumerate(sentences):
            try:
                tree = self._tree_to_list(constituency_trees[i].values[2][0])
                graph = nx.DiGraph()
                graph_id = ids[i] if ids else None
                
                graph = self._build_graph(graph, tree, sentence, graph_id=graph_id)
                self._remove_nodes_and_reconnect(graph)
                graph_list.append(graph)
            except Exception as e:
                print(f"Error processing sentence {i}: '{sentence[:30]}...'. Error: {e}")
                # Add an empty graph to maintain the same length as the input
                graph = nx.DiGraph()
                graph.graph['error'] = str(e)
                graph_list.append(graph)
                
        return graph_list

    def draw_graph(self, graph: nx.DiGraph, figsize: tuple = (12, 8)) -> None:
        """
        Draw the constituency graph.

        Args:
            graph (nx.DiGraph): The constituency graph to draw.
            figsize (tuple, optional): Size of the figure. Defaults to (12, 8).
        """
        plt.figure(figsize=figsize)
        labels = nx.get_node_attributes(graph, 'label')
        pos = nx.kamada_kawai_layout(graph)
        nx.draw(graph, pos, with_labels=True, labels=labels, 
                node_color='lightblue', node_size=1500, 
                font_weight='bold', font_size=8, 
                edge_color='gray')
        plt.title("Constituency Graph")
        plt.show()

    def save_graph(self, graph: Union[nx.DiGraph, List[nx.DiGraph]], folder: str, filename: str) -> None:
        """
        Save the constituency graph(s) to a file in pickle format.

        Args:
            graph (Union[nx.DiGraph, List[nx.DiGraph]]): The graph or list of graphs to save.
            folder (str): Destination folder path.
            filename (str): Name of the file without extension.
            
        Raises:
            IOError: If there's an error saving the file.
        """
        try:
            with open(f'{folder}/{filename}.pkl', 'wb') as f:
                pkl.dump(graph, f)
            print(f"Graph(s) successfully saved to {folder}/{filename}.pkl")
        except IOError as e:
            raise IOError(f"Failed to save graph to {folder}/{filename}.pkl: {e}")