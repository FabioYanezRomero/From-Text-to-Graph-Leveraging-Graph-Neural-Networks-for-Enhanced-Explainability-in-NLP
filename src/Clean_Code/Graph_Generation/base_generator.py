"""
Base Graph Generator Module
This module provides a base class for all graph generators to implement.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Any
import networkx as nx
import pickle as pkl
import matplotlib.pyplot as plt

class BaseGraphGenerator(ABC):
    """
    Abstract base class for all graph generators.
    
    This class defines the common interface that all graph generators must implement,
    ensuring consistency across different graph generation approaches.
    
    Attributes:
        model (str): Name of the model used for graph generation.
        property (str): Type of graph property (e.g., 'syntactic', 'semantic').
        device (str): Device to run computation on ('cpu', 'cuda:0', etc.)
    """
    def __init__(self, model: str, property: str = None, device: str = 'cuda:0'): # type: ignore
        """
        Initialize the graph generator with a model and device.
        
        Args:
            model (str): Name of the model to use.
            device (str, optional): Device to run the model on. Defaults to 'cuda:0'.
        """
        self.model = model
        self.property = property  # To be set by child classes
        self.device = device
    
    @abstractmethod
    def _parse(self, sentences: List[str]) -> Any:
        """
        Parse the input sentences using the appropriate parser.
        
        Args:
            sentences (List[str]): List of sentences to parse.
        
        Returns:
            Any: The parsed representation of the sentences.
        """
        pass
    
    @abstractmethod
    def _build_graph(self, parsed_data: Any, ids: Optional[List[str]] = None) -> List[nx.DiGraph]:
        """
        Build graphs from the parsed data.
        
        Args:
            parsed_data (Any): The parsed representation of the sentences.
            ids (Optional[List[str]]): Optional list of IDs to assign to the graphs.
        
        Returns:
            List[nx.DiGraph]: List of directed graphs.
        """
        pass
    
    @abstractmethod
    def get_graph(self, sentences: List[str], ids: Optional[List[str]] = None) -> List[nx.DiGraph]:
        """
        Generate graphs for a list of sentences.
        
        Args:
            sentences (List[str]): List of sentences to process.
            ids (Optional[List[str]]): Optional list of IDs to assign to the graphs.
        
        Returns:
            List[nx.DiGraph]: List of generated graphs.
        """
        pass
    
    def draw_graph(self, graph: nx.DiGraph, figsize: tuple = (12, 8)) -> None:
        """
        Draw the graph for visualization.
        
        Args:
            graph (nx.DiGraph): The graph to draw.
            figsize (tuple, optional): Size of the figure. Defaults to (12, 8).
        """
        plt.figure(figsize=figsize)
        labels = nx.get_node_attributes(graph, 'label')
        pos = nx.kamada_kawai_layout(graph)
        
        nx.draw(graph, pos, with_labels=True, labels=labels,
                node_color='lightblue', node_size=1500,
                font_weight='bold', font_size=8,
                edge_color='gray')
                
        edge_labels = nx.get_edge_attributes(graph, 'label')
        if edge_labels:
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
            
        plt.title(f"{self.property.capitalize()} Graph")
        plt.show()
    
    def save_graph(self, graph: Union[nx.DiGraph, List[nx.DiGraph]], folder: str, filename: str) -> None:
        """
        Save the graph(s) to a file in pickle format.
        
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
            print(f"Graph(s) successfully saved to {folder}/{self.property}/{filename}.pkl")
        except IOError as e:
            raise IOError(f"Failed to save graph to {folder}/{self.property}/{filename}.pkl: {e}")
