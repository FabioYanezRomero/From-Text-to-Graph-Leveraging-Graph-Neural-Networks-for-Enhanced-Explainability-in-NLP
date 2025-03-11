import networkx as nx
from enum import Enum
from typing import List, Set, Union


class GraphType(Enum):
    """Enum defining the supported graph types"""
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"
    CONSTITUENCY = "constituency"
    KNOWLEDGE = "knowledge"
    
    @classmethod
    def from_string(cls, type_str: str) -> "GraphType":
        """Convert string to GraphType, case-insensitive"""
        try:
            return next(t for t in cls if t.value == type_str.lower())
        except StopIteration:
            raise ValueError(f"Unknown graph type: {type_str}")


class GraphMixer:
    """
    This class will mix different types of graphs i.e. syntactic, semantic, constituency or
    knowledge graphs into a single graph based on the user's choice and the individual graphs provided.    
    """
    def __init__(self):
        """Initialize the mixer"""
        self.supported_types = {t.value for t in GraphType}

    def mix(self, graph_list: List[nx.MultiDiGraph], types_to_mix: Set[Union[str, GraphType]]) -> nx.MultiDiGraph:
        """
        Mix the graphs based on the specified types.
        
        Args:
            graph_list: List of graphs to potentially mix
            types_to_mix: Set of GraphTypes or strings indicating which graph types to include
            
        Returns:
            A merged NetworkX MultiDiGraph containing all the specified graph types
        """
        # Convert string types to GraphType enums
        requested_types = {
            GraphType.from_string(t) if isinstance(t, str) else t 
            for t in types_to_mix
        }
        
        # Validate requested types
        if not all(t.value in self.supported_types for t in requested_types):
            invalid = [t for t in requested_types if t.value not in self.supported_types]
            raise ValueError(f"Unsupported graph type(s): {invalid}")

        # Initialize the merged graph
        G = nx.MultiDiGraph()
        G.graph['property'] = []
        G.graph['model'] = []
        
        # Keep track of processed types to ensure no duplicates
        processed_types = set()
        
        # Process each input graph
        for graph in graph_list:
            graph = nx.MultiDiGraph(graph)
            graph_type = graph.graph.get('property')
            
            # Skip if this graph type wasn't requested
            if not any(t.value == graph_type for t in requested_types):
                continue
                
            # Check for duplicate graph types
            if graph_type in processed_types:
                raise ValueError(f"Multiple graphs of type {graph_type} detected. Only one graph per type is allowed.")
                
            # Merge the graph
            G = nx.compose(G, graph)
            G.graph['property'].append(graph_type)
            G.graph['model'].append(graph.graph.get('model'))
            processed_types.add(graph_type)
            
        # Verify all requested types were found
        missing_types = {t.value for t in requested_types} - processed_types
        if missing_types:
            raise ValueError(f"The following requested graph types were not found in the input: {missing_types}")
            
        return G