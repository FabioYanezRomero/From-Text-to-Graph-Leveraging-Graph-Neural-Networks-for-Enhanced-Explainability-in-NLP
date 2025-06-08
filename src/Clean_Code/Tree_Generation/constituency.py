"""
Constituency Tree Generator Module

This module provides functionality to create constituency trees from sentences.
It leverages the Stanza NLP library to generate constituency trees and converts them
into directed graphs where nodes represent constituents and words.
"""

from typing import List, Dict, Union, Any, Optional
import os
import networkx as nx
import torch
import torch.cuda
import stanza
from .base_generator import BaseTreeGenerator

# Dictionary mapping constituency labels to more descriptive phrases
PHRASE_MAPPER = {
    # POS TAGS
    'CC': '«COORDINATING CONJUNCTION»',
    'CD': '«CARDINAL NUMBER»',
    'DT': '«DETERMINER»',
    'EX': '«EXISTENTIAL THERE»',
    'FW': '«FOREIGN WORD»',
    'IN': '«PREPOSITION OR SUBORDINATING CONJUNCTION»',
    'JJ': '«ADJECTIVE»',
    'JJR': '«ADJECTIVE, COMPARATIVE»',
    'JJS': '«ADJECTIVE, SUPERLATIVE»',
    'LS': '«LIST MARKER»',
    'MD': '«MODAL VERB»',
    'NN': '«NOUN, SINGULAR OR MASS»',
    'NNS': '«NOUN, PLURAL»',
    'NNP': '«PROPER NOUN, SINGULAR»',
    'NNPS': '«PROPER NOUN, PLURAL»',
    'PDT': '«PREDETERMINER»',
    'POS': '«POSSESSIVE ENDING»',
    'PRP': '«PERSONAL PRONOUN»',
    'PRP$': '«POSSESSIVE PRONOUN»',
    'RB': '«ADVERB»',
    'RBR': '«ADVERB, COMPARATIVE»',
    'RBS': '«ADVERB, SUPERLATIVE»',
    'RP': '«PARTICLE»',
    'SYM': '«SYMBOL»',
    'TO': '«TO»',
    'UH': '«INTERJECTION»',
    'VB': '«VERB, BASE FORM»',
    'VBD': '«VERB, PAST TENSE»',
    'VBG': '«VERB, GERUND OR present participle»',
    'VBN': '«VERB, past participle»',
    'VBP': '«VERB, non-3rd person singular present»',
    'VBZ': '«VERB, 3rd person singular present»',
    'WDT': '«WH-DETERMINER»',
    'WP': '«WH-PRONOUN»',
    'WP$': '«WH-POSSESSIVE PRONOUN»',
    'WRB': '«WH-ADVERB»',
    # CONSTITUENCY TAGS
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


class ConstituencyTreeGenerator(BaseTreeGenerator):
    """
    Creates constituency trees from sentences.

    This class processes sentences using a Stanza constituency parser and converts
    the parsed trees into directed graphs. Each node in the graph represents
    either a constituent phrase or a word from the sentence.

    Attributes:
        model (str): Name or configuration of the constituency parser model.
        property (str): Property type, always set to 'constituency'.
        nlp: The loaded Stanza pipeline.
        device (str): The device to run the parser on (CPU or CUDA).
    """

    def __init__(self, device: str = 'cuda:0'):
        """
        Initialize the constituency parser with Stanza.

        Args:
            device (str, optional): Device to run the parser on. Defaults to 'cuda:0'.

        Raises:
            RuntimeError: If the specified device is not available.
        """
        super().__init__(property='constituency', device=device)
        
        # Verify device availability for better error handling
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please use 'cpu' instead or check CUDA installation.")
        
        self.device = device
        if device.startswith('cuda'):
            torch.cuda.set_device(device)
        
        # Set up Stanza configuration
        stanza_device = 'gpu' if device.startswith('cuda') else 'cpu'
        
        # Download the transformer-based models explicitly
        try:
            print("Downloading transformer-based Stanza English models...")
            # Download with transformer-based constituency parser
            # Using 'default_accurate' which uses BERT for best accuracy
            stanza.download('en', package='default_accurate', processors={
                'tokenize': 'default',
                'pos': 'default',
                'constituency': 'default_accurate'  # Uses BERT for best accuracy
            })
            print("Transformer-based Stanza models downloaded successfully.")
        except Exception as e:
            print(f"Warning: Could not download transformer-based Stanza model: {e}")
            print("Falling back to default models...")
            try:
                stanza.download('en', package='default')
            except Exception as e2:
                print(f"Error downloading default models: {e2}")
        
        # Initialize the Stanza pipeline with BERT-based constituency parser
        #print("Initializing Stanza pipeline with BERT-based constituency parser...")
        #from transformers import AutoModel, AutoTokenizer

        # Use the model name that matches your Stanza model (e.g., 'google/electra-large-discriminator')
        #AutoModel.from_pretrained('google/electra-large-discriminator')
        #AutoTokenizer.from_pretrained('google/electra-large-discriminator')
        
        self.nlp = stanza.Pipeline(
            lang='en',
            processors={
                'tokenize': 'default',
                'pos': 'default',
                'constituency': 'default_accurate'  # Uses BERT for best accuracy
            },
            package='default_accurate',  # Specify the package for BERT-based model
            use_gpu=(stanza_device == 'gpu'),
            download_method=stanza.DownloadMethod.NONE,  # We already downloaded the models
            tokenize_pretokenized=False,
            tokenize_no_ssplit=False,
            pos_batch_size=1000,
            constituency_batch_size=1000,
            constituency_pretagged=True  # Use POS tags from the POS tagger
        )
        print("Stanza pipeline with BERT-based constituency parser initialized successfully.")
        

    def _parse(self, sentences: List[str]):
        """
        Parse sentences using the Stanza constituency parser.
        If a sentence is split into multiple sentences, they are combined into a single parse.

        Args:
            sentences (List[str]): List of sentences to parse.

        Returns:
            List: List of parsed constituency trees, one per input sentence.
        """
        trees = []
        for i, sentence in enumerate(sentences):
            try:
                # Process the current sentence
                doc = self.nlp(sentence)
                
                if not doc.sentences:
                    raise ValueError("No sentences returned by parser")
                    
                # If we get multiple sentences, create a combined parse
                if len(doc.sentences) > 1:
                    # Create a new root node for the combined parse
                    combined_parse = ['ROOT', ['S']]  # Start with ROOT -> S structure
                    
                    for sent in doc.sentences:
                        if hasattr(sent, 'constituency') and sent.constituency:
                            # Get the parse tree as a string
                            parse_str = str(sent.constituency)
                            # Parse it into a nested list structure
                            sent_parse = self._tree_to_list(parse_str)
                            # If the parse starts with ['ROOT', ...], take the children of ROOT
                            if (isinstance(sent_parse, list) and len(sent_parse) > 1 and 
                                isinstance(sent_parse[0], str) and sent_parse[0] == 'ROOT'):
                                # Add all children of ROOT (skipping ROOT itself)
                                combined_parse[1].extend(sent_parse[1:])
                            else:
                                # Otherwise add the entire parse
                                combined_parse[1].append(sent_parse)
                                
                    trees.append(combined_parse)
                else:
                    # Single sentence case
                    sent = doc.sentences[0]
                    if hasattr(sent, 'constituency') and sent.constituency:
                        trees.append(sent.constituency)
                    else:
                        raise ValueError("No constituency parse available")
                        
            except Exception as e:
                print(f"Error processing sentence {i}: {e}")
                print(f"Sentence content: {sentence}")
                # Fallback: create a simple flat structure
                words = sentence.split()[:100]  # Limit to 100 words
                flat_tree = ['ROOT', ['S'] + [['WORD', word] for word in words]]
                trees.append(flat_tree)
                continue
                
        if len(trees) != len(sentences):
            raise RuntimeError(f"Parse count mismatch: expected {len(sentences)}, got {len(trees)}")
            
        return trees

    def _tree_to_list(self, tree) -> Union[str, List]:
        """
        Convert the parsed constituency tree into a nested list structure.

        Args:
            tree: A Stanza constituency tree, subtree, or list.

        Returns:
            Union[str, List]: A string for leaf nodes or a list for non-leaf nodes.
        """
        # If tree is already a list, return it as is
        if isinstance(tree, list):
            return tree
            
        # If tree is a string, it's already in string format
        if isinstance(tree, str):
            try:
                # Use a stack-based iterative parser
                stack = []
                current = []
                i = 0
                n = len(tree)
                
                while i < n:
                    if tree[i] == ' ':
                        i += 1
                        continue
                        
                    if tree[i] == '(':
                        # Push current context to stack and start new one
                        stack.append(current)
                        current = []
                        i += 1
                    elif tree[i] == ')':
                        # Pop the last item from stack as parent
                        if current:
                            if stack:
                                parent = stack.pop()
                                parent.append(current)
                                current = parent
                            else:
                                # This is the root
                                if len(current) == 1:
                                    return current[0]
                                return current
                        i += 1
                    else:
                        # Read token
                        j = i
                        while j < n and tree[j] not in '() ':
                            j += 1
                        token = tree[i:j].strip()
                        if token:
                            current.append(token)
                        i = j
                        
                # Handle any remaining context
                if stack:
                    while stack:
                        parent = stack.pop()
                        if current:
                            parent.append(current)
                        current = parent
                        
                return current[0] if len(current) == 1 else current
                
            except Exception as e:
                print(f"Error parsing tree string: {e}")
                # Fall back to a simple flat structure
                words = [w for w in tree.split() if w not in '()']
                return ['ROOT', ['S'] + [['WORD', word] for word in words]]

        # Original logic for Stanza Tree objects
        if tree.is_leaf():
            return tree.label
        
        # Handle non-leaf nodes (constituents)
        return [tree.label] + [self._tree_to_list(child) for child in tree.children]

        # Fallback for unknown types
        return str(tree)

    def _build_graph(self, graph: nx.DiGraph, node_list: List, sentence: str, parent_id: str = '', graph_id: str = None, node_id_counter=None, parent_nid=None) -> nx.DiGraph:
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
        # Setup id counter if not provided
        if node_id_counter is None:
            node_id_counter = {'val': 0}

        # Assign a unique numerical id to this node
        nid = node_id_counter['val']
        node_id_counter['val'] += 1

        parent = node_list[0] + parent_id
        # Add parent node with id and label if not already present
        if parent not in graph:
            # Always use PHRASE_MAPPER for constituent node labels if available
            label_key = node_list[0]
            mapped_label = PHRASE_MAPPER.get(label_key, label_key)
            graph.add_node(parent, id=nid, label=f"{nid}: {mapped_label}")
        else:
            # If already present, update with id if not set
            if 'id' not in graph.nodes[parent]:
                graph.nodes[parent]['id'] = nid
                # Update label with mapping if needed
                label_key = node_list[0]
                mapped_label = PHRASE_MAPPER.get(label_key, label_key)
                graph.nodes[parent]['label'] = f"{nid}: {mapped_label}"

        children = node_list[1:]
        for i, child in enumerate(children):
            if isinstance(child, list):
                # Process non-leaf nodes (constituents)
                child_id = parent_id + str(i)
                child_label = child[0]
                node_key = str(child_label) + str(child_id)

                # Assign id and label
                child_nid = node_id_counter['val']
                node_id_counter['val'] += 1
                # Always use PHRASE_MAPPER for constituent node labels if available
                mapped_child_label = PHRASE_MAPPER.get(child_label, child_label)
                graph.add_node(node_key, id=child_nid, label=f"{child_nid}: {mapped_child_label}")
                graph.add_edge(parent, node_key, label="constituency relation")
                self._build_graph(graph, child, sentence, child_id, graph_id, node_id_counter, parent_nid=nid)
            else:
                # Process leaf nodes (words)
                try:
                    counter = node_id_counter['val']
                    node_id_counter['val'] += 1
                    # Always use PHRASE_MAPPER for leaf (word) labels if available
                    mapped_leaf_label = PHRASE_MAPPER.get(child, child)
                    graph.add_node(counter, id=counter, label=f"{counter}: {mapped_leaf_label}")
                    graph.add_edge(parent, counter, label="constituency relation")
                except Exception:
                    continue
        # Add graph metadata
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
        Generate constituency trees for a list of sentences.

        Args:
            sentences (List[str]): List of sentences to process.
            ids (List[str], optional): List of IDs to assign to each graph. Defaults to None.

        Returns:
            List[nx.DiGraph]: List of constituency trees.
            
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
                    graph_id=ids[i] if ids else None,
                    node_id_counter={'val': 0}  # reset id counter for each graph
                )
                
                # Remove nodes that start with '_' and reconnect their parents to their children
                self._remove_nodes_and_reconnect(graph)
                
                graphs.append(graph)
            
            return graphs
            
        except Exception as e:
            raise RuntimeError(f"Error generating constituency trees: {e}")
