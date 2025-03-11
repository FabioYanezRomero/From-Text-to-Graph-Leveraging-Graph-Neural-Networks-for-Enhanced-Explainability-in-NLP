from supar import Parser
import networkx as nx
import matplotlib.pyplot as plt
import pickle as pkl
import torch
from .base_generator import BaseGraphGenerator

PHRASE_MAPPER = {
    'S' : '«SENTENCE»',
    'NP' : '«NOUN PHRASE»',
    'VP' : '«VERB PHRASE»',
    'PP' : '«PREPOSITIONAL PHRASE»',
    'ADJP' : '«ADJECTIVE PHRASE»',
    'ADVP' : '«ADVERB PHRASE»',
    'SBAR' : '«SUBORDINATE CLAUSE»',
    'PRT' : '«PARTICLE»',
    'INTJ' : '«INTERJECTION»',
    'CONJP' : '«CONJUCTION PHRASE»',
    'LST' : '«LIST MARKER»',
    'UCP' : '«UNLIKE COORDINATED PHRASE»',
    'PRN' : '«PARENTETICAL»',
    'FRAG' : '«FRAGMENT»',
    'SINV' : '«INVERTED SENTENCE»',
    'SBARQ' : '«SUBORDINATE CLAUSE QUESTION»',
    'SQ' : '«QUESTION»',
    'WHADJP' : '«WH-ADJECTIVE PHRASE»',
    'WHAVP' : '«WH-ADVERB PHRASE»',
    'WHNP' : '«WH-NOUN PHRASE»',
    'WHPP' : '«WH-PREPOSITIONAL PHRASE»',
    'X' : '«UNKNOWN»'
}

class constituency_graph_generator(BaseGraphGenerator):
    """
    This class is used to create a constituency graph from a sentence. 
    
    It first parses the sentence using a constituency parser, then converts the parsed tree into a list. Technically, the tree is
    a tuple that needs to be preprocessed for handle the data.
    
    The list is then converted into a graph, and edges are added to the graph from the node list. Here we remove intermediate nodes
    that add no value to the graph. The labels for the nodes in the graph are taken from a predefined PHRASE_MAPPER dictionary. If a label is 
    not found in the dictionary, the original label is used.
    Notice that ONLY THREE METHODS SHOULD BE USED IN THIS CLASS: get_graph, draw_graph, and save_graph
        · The get_graph method returns the constituency graph of a sentence. The draw_graph method draws the constituency graph.
        · The graph can be drawn using the Kamada-Kawai layout algorithm. 
        · The graph can be saved to a file in pickle format. The filename and folder for the file are provided as parameters to 
          the save_graph method. The graph can be loaded later using the pickle.load function. 
    """
    def __init__(self, model: str, device: str = 'cuda:0'):
        """
        Initialize the constituency parser with a given model.
        """
        super().__init__(model, device)
        self.property = 'constituency'
        self.cons = Parser.load(model)

    def _parse(self, sentence: str):
        """
        Parse the sentence using the constituency parser.
        """
        return self.cons.predict(sentence, verbose=False)[0]

    def _tree_to_list(self, tree):
        """
        Convert the parsed tree into a list.
        """
        if isinstance(tree, str):  # base case: leaf node
            return tree
        return [tree.label()] + [self._tree_to_list(child) for child in tree]

    def _build_graph(self, parsed_result, sentence=None):
        """
        Build a graph from the parsed tree and sentence.
        
        Args:
            parsed_result: The result from _parse method
            sentence: The original sentence (needed for constituency parsing)
            
        Returns:
            A networkx graph
        """
        if sentence is None:
            raise ValueError("Constituency parsing requires the original sentence")
            
        graph = nx.DiGraph()
        graph = self._add_edges_from_nodelist(graph, parsed_result, sentence)
        
        # Add graph metadata
        graph.graph['model'] = self.model
        graph.graph['property'] = self.property
        
        # Remove nodes and reconnect as needed
        self._remove_nodes_and_reconnect(graph)
        
        return graph

    def _add_edges_from_nodelist(self, graph, node_list, sentence, parent_id=''):
        """
        Add edges to the graph from the node list.
        """
        parent = node_list[0] + parent_id
        if not parent in graph:
            graph.add_node(str(parent), label=PHRASE_MAPPER.get(str(parent), str(parent)))
            
        children = node_list[1:]
        for i, child in enumerate(children):
            if isinstance(child, list):
                child_id = parent_id + str(i)
                graph.add_node(str(child[0]) + str(child_id), label=child[0])
                graph.nodes[str(child[0]) + str(child_id)]['label'] = PHRASE_MAPPER.get(graph.nodes[str(child[0]) + str(child_id)]['label'], child[0])
                graph.add_edge(parent, child[0] + child_id)
                self._add_edges_from_nodelist(graph, child, sentence, child_id)
            else:
                counter = sentence.index(child)
                graph.add_node(counter, label=child)
                graph.nodes[counter]['label'] = PHRASE_MAPPER.get(child, child)
                graph.add_edge(parent, counter)
        
        return graph
    
    def _remove_nodes_and_reconnect(self, graph):
        """
        Remove nodes starting with '_' and reconnect their parents and children.
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
                    graph.add_edge(parent, child)
            
            # Remove the node
            graph.remove_node(node)

    def get_graph(self, sentence: list):
        """
        Get the constituency graph of a sentence.
        """
        constituency_tree = self._parse(sentence)
        tree = self._tree_to_list(constituency_tree.values[2][0])
        return self._build_graph(tree, sentence)

    # draw_graph and save_graph methods are inherited from BaseGraphGenerator