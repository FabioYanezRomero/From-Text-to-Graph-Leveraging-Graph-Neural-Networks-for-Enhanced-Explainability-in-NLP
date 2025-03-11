import networkx as nx
import matplotlib.pyplot as plt
import pickle as pkl
import torch
from abc import ABC, abstractmethod

class BaseGraphGenerator(ABC):
    """
    Abstract base class for graph generators.
    
    This class defines the common interface that all graph generators should implement.
    Subclasses should implement the _parse and _build_graph methods.
    """
    
    def __init__(self, model: str, device: str = 'cuda:0'):
        """
        Initialize the graph generator with a model and device.
        
        Args:
            model (str): The model to be used for parsing.
            device (str): The device to be used for computations. Default is 'cuda:0'.
        """
        self.model = model
        self.device = device
        torch.cuda.set_device(device)
    
    @abstractmethod
    def _parse(self, sentence):
        """
        Parse the sentence using an appropriate parser.
        
        Args:
            sentence: The sentence to parse.
            
        Returns:
            The parsed result.
        """
        pass
    
    @abstractmethod
    def _build_graph(self, parsed_result):
        """
        Build a graph from the parsed result.
        
        Args:
            parsed_result: The result from _parse method.
            
        Returns:
            A networkx graph.
        """
        pass
    
    @abstractmethod
    def get_graph(self, sentence):
        """
        Get the graph representation of a sentence.
        
        Args:
            sentence: The sentence to create a graph from.
            
        Returns:
            A networkx graph.
        """
        pass
    
    def draw_graph(self, graph):
        """
        Draw the graph.
        
        Args:
            graph: The networkx graph to draw.
        """
        labels = nx.get_node_attributes(graph, 'label')
        nx.draw_kamada_kawai(graph, with_labels=True, font_weight='bold')
        plt.show()
    
    def save_graph(self, graph, folder, filename):
        """
        Save the graph to a file.
        
        Args:
            graph: The networkx graph to save.
            folder: The folder to save the graph in.
            filename: The filename to save the graph as.
        """
        with open(f'{folder}/{filename}.pkl', 'wb') as f:
            pkl.dump(graph, f)